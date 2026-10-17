"""Weekly hierarchical environment with STOP and up to N joint actions."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any, Callable

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # Lets contract/unit tests run before optional RL deps are installed.
    gym = None
    spaces = None

from .candidate_proposals import CandidateGenerator, CandidatePool, ProposalContext
from .observation import ObservationEncoder
from .reward import PortfolioComparison, RewardConfig, compare_portfolio
from .schemas import BusinessState, CandidateAction
from .simulator_contract import NextWeekSimulator


class InvalidEnvironmentAction(ValueError):
    pass


class _FallbackDiscrete:
    def __init__(self, n: int) -> None:
        self.n = n


class _FallbackSpace:
    def __init__(self, **_kwargs: Any) -> None:
        pass


_EnvironmentBase = gym.Env if gym is not None else object


@dataclass(frozen=True)
class EnvironmentConfig:
    planning_horizon_weeks: int = 8
    max_new_actions_per_week: int = 3
    minimum_actions_before_stop: int = 0
    require_positive_marginal_net_profit: bool = False
    minimum_marginal_net_profit_usd: float = 0.0
    seed: int = 42

    def __post_init__(self) -> None:
        if self.planning_horizon_weeks < 1 or self.max_new_actions_per_week < 1:
            raise ValueError("Planning horizon and max actions must be positive")
        if self.minimum_actions_before_stop < 0:
            raise ValueError("minimum_actions_before_stop must be nonnegative")
        if self.minimum_actions_before_stop > self.max_new_actions_per_week:
            raise ValueError("minimum_actions_before_stop cannot exceed max_new_actions_per_week")
        if not isinstance(self.require_positive_marginal_net_profit, bool):
            raise ValueError("require_positive_marginal_net_profit must be boolean")
        if (
            not isfinite(self.minimum_marginal_net_profit_usd)
            or self.minimum_marginal_net_profit_usd < 0
        ):
            raise ValueError("minimum_marginal_net_profit_usd must be finite and nonnegative")


ContextProvider = Callable[[BusinessState], ProposalContext]
ScenarioProvider = Callable[[BusinessState, int], Any]
StateTransition = Callable[[BusinessState, Any, tuple[CandidateAction, ...]], BusinessState]


class WeeklyDecisionEnv(_EnvironmentBase):
    """MaskablePPO-compatible environment; candidate index 0 is always STOP."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        initial_state_provider: Callable[[], BusinessState],
        context_provider: ContextProvider,
        scenario_provider: ScenarioProvider,
        simulator: NextWeekSimulator,
        candidate_generator: CandidateGenerator,
        observation_encoder: ObservationEncoder,
        reward_config: RewardConfig,
        config: EnvironmentConfig | None = None,
        state_transition: StateTransition | None = None,
    ) -> None:
        self.initial_state_provider = initial_state_provider
        self.context_provider = context_provider
        self.scenario_provider = scenario_provider
        self.simulator = simulator
        self.candidate_generator = candidate_generator
        self.encoder = observation_encoder
        self.reward_config = reward_config
        self.config = config or EnvironmentConfig()
        self.state_transition = state_transition or getattr(simulator, "advance_state", None)
        if self.state_transition is None:
            raise ValueError("A state_transition is required so weekly state actually changes")
        candidate_limit = self.encoder.schema.max_candidates
        if candidate_limit != self.candidate_generator.proposals_per_step:
            raise ValueError("Observation max_candidates must equal generator proposals_per_step")
        if spaces is not None:
            self.action_space = spaces.Discrete(candidate_limit + 1)
            self.observation_space = spaces.Dict(
                {
                    "state": spaces.Box(-np.inf, np.inf, (self.encoder.state_width,), np.float32),
                    "candidates": spaces.Box(
                        -np.inf,
                        np.inf,
                        (candidate_limit, self.encoder.candidate_width),
                        np.float32,
                    ),
                    "candidate_mask": spaces.Box(0.0, 1.0, (candidate_limit,), np.float32),
                }
            )
        else:
            self.action_space = _FallbackDiscrete(candidate_limit + 1)
            self.observation_space = _FallbackSpace()
        self._episode_seed = self.config.seed
        self._state: BusinessState | None = None
        self._context: ProposalContext | None = None
        self._pool: CandidatePool | None = None
        self._selected: list[CandidateAction] = []
        self._week_index = 0
        self._profitability_rejected_count = 0
        self._candidate_marginal_net_profit_usd: tuple[float, ...] = ()

    def _filter_profitable_candidates(self, pool: CandidatePool) -> CandidatePool:
        """Keep only candidates with positive marginal portfolio net profit.

        Each candidate is evaluated jointly with the actions already selected
        for the week. The paired simulator scenario and seed are identical for
        the current and augmented portfolio.
        """

        assert self._state is not None
        if not self.config.require_positive_marginal_net_profit:
            self._profitability_rejected_count = 0
            self._candidate_marginal_net_profit_usd = ()
            return pool
        scenario = self.scenario_provider(self._state, self._week_index)
        seed = self._episode_seed + self._week_index
        current = compare_portfolio(
            self.simulator,
            self._state,
            tuple(self._selected),
            scenario,
            seed,
            self.reward_config,
        )
        kept: list[CandidateAction] = []
        marginals: list[float] = []
        for candidate in pool.candidates:
            augmented = compare_portfolio(
                self.simulator,
                self._state,
                tuple((*self._selected, candidate)),
                scenario,
                seed,
                self.reward_config,
            )
            marginal = (
                augmented.net_incremental_profit_usd
                - current.net_incremental_profit_usd
            )
            if marginal > self.config.minimum_marginal_net_profit_usd:
                kept.append(candidate)
                marginals.append(marginal)
        self._profitability_rejected_count = len(pool.candidates) - len(kept)
        self._candidate_marginal_net_profit_usd = tuple(marginals)
        return CandidatePool(
            candidates=tuple(kept),
            coverage=pool.coverage,
            rejected_count=pool.rejected_count + self._profitability_rejected_count,
        )

    def _refresh(self) -> None:
        assert self._state is not None
        self._context = self.context_provider(self._state)
        generated_pool = self.candidate_generator.generate(
            self._state,
            self._context,
            tuple(self._selected),
            step_number=self._week_index * self.config.max_new_actions_per_week + len(self._selected),
        )
        self._pool = self._filter_profitable_candidates(generated_pool)

    def _observation(self):
        assert self._state is not None and self._context is not None and self._pool is not None
        return self.encoder.encode(self._state, self._pool, self._context, len(self._selected))

    def action_masks(self) -> np.ndarray:
        if self._pool is None:
            raise RuntimeError("reset() must be called before action_masks()")
        # Keep an emergency STOP when no feasible candidate remains. Without
        # this escape hatch a strict minimum could create an all-false mask.
        stop_allowed = (
            len(self._selected) >= self.config.minimum_actions_before_stop
            or not self._pool.candidates
        )
        masks = [stop_allowed]
        masks.extend([True] * len(self._pool.candidates))
        masks.extend([False] * (self.encoder.schema.max_candidates - len(self._pool.candidates)))
        return np.asarray(masks, dtype=bool)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        if gym is not None:
            super().reset(seed=seed)
        del options
        self._episode_seed = self.config.seed if seed is None else seed
        self._state = self.initial_state_provider()
        self._selected = []
        self._week_index = 0
        self._refresh()
        return self._observation(), {
            "week_start": self._state.week_start.isoformat(),
            "candidate_coverage": self._pool.coverage if self._pool else (),
            "profitability_rejected_count": self._profitability_rejected_count,
            "candidate_marginal_net_profit_usd": self._candidate_marginal_net_profit_usd,
            "simulator_support": getattr(self.simulator, "label", "action_conditioned"),
        }

    def _advance_week(self) -> tuple[float, PortfolioComparison]:
        assert self._state is not None
        scenario = self.scenario_provider(self._state, self._week_index)
        comparison = compare_portfolio(
            self.simulator,
            self._state,
            tuple(self._selected),
            scenario,
            self._episode_seed + self._week_index,
            self.reward_config,
        )
        self._state = self.state_transition(self._state, comparison.policy, tuple(self._selected))
        self._selected = []
        self._week_index += 1
        return comparison.reward, comparison

    def step(self, action: int):
        if self._state is None or self._pool is None:
            raise RuntimeError("reset() must be called before step()")
        if isinstance(action, bool) or not isinstance(action, (int, np.integer)):
            raise InvalidEnvironmentAction("Action must be an integer candidate index")
        action = int(action)
        masks = self.action_masks()
        if action < 0 or action >= len(masks) or not masks[action]:
            raise InvalidEnvironmentAction(f"Candidate index {action} is masked or out of range")
        stopped = action == 0
        if not stopped:
            self._selected.append(self._pool.candidates[action - 1])
        reached_limit = len(self._selected) >= self.config.max_new_actions_per_week
        comparison = None
        reward = 0.0
        if stopped or reached_limit:
            reward, comparison = self._advance_week()
        terminated = self._week_index >= self.config.planning_horizon_weeks
        if not terminated:
            self._refresh()
        info = {
            "calendar_advanced": bool(stopped or reached_limit),
            "selected_action_count": len(comparison.new_actions) if comparison else len(self._selected),
            "portfolio_comparison": comparison,
            "profitability_rejected_count": self._profitability_rejected_count,
            "candidate_marginal_net_profit_usd": self._candidate_marginal_net_profit_usd,
        }
        return self._observation() if not terminated else self._terminal_observation(), reward, terminated, False, info

    def _terminal_observation(self):
        assert self._state is not None
        empty_pool = CandidatePool((), (), 0)
        context = self.context_provider(self._state)
        return self.encoder.encode(self._state, empty_pool, context, 0)
