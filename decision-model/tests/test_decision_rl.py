from __future__ import annotations

import json
import sys
import tempfile
import unittest
from datetime import date
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from decision_rl.candidate_proposals import CandidateGenerator, ProposalContext
from decision_rl.config import load_config
from decision_rl.environment import EnvironmentConfig, InvalidEnvironmentAction, WeeklyDecisionEnv
from decision_rl.observation import ObservationEncoder, ObservationSchema
from decision_rl.registry import RegistryValidationError, load_decisions
from decision_rl.reporting import build_portfolio_report
from decision_rl.reward import RewardConfig, compare_portfolio
from decision_rl.schemas import BusinessState, CandidateAction, IntegerDomain, ProductState, Target
from decision_rl.simulator_contract import audit_action_response
from decision_rl.split_guard import Purpose, Split, SplitAccessError, SplitGuard
from decision_rl.synthetic_simulator import SyntheticResponseProfile, SyntheticSimulator
from decision_rl.validity import ActionValidator


REGISTRY_PATH = ROOT / "config" / "decisions.json"


def product_state() -> BusinessState:
    return BusinessState(
        week_start=date(2026, 7, 6),
        products=(
            ProductState(
                branch_id="Garden",
                product_id="Iced Latte",
                category_id="Cold Coffee",
                expected_units=100.0,
                listed_price_usd=10.0,
                unit_cost_usd=4.0,
                discount_rate=0.0,
                prepared_units=120,
            ),
        ),
        remaining_action_budget_usd=1000.0,
    )


def price_evidence(change: int = 7) -> dict[str, object]:
    return {
        "branch_id": "Garden",
        "product_id": "Iced Latte",
        "current_price": 10.0,
        "recommended_change_pct": change,
        "within_store_performance": {"periods": 8},
        "evidence_period": "train-window",
        "confidence_or_uncertainty": {"status": "toy"},
    }


def price_action(change: int = 7) -> CandidateAction:
    return CandidateAction(
        decision_id="price_change",
        scope="product",
        target=Target(branch_id="Garden", product_id="Iced Latte"),
        parameters={"change_pct": change},
        evidence=price_evidence(change),
    )


def domains() -> dict[tuple[str, str], IntegerDomain]:
    return {
        ("price_change", "change_pct"): IntegerDomain(-25, 35, "toy-simulator-support"),
        ("promotion", "intensity_pct"): IntegerDomain(0, 100, "toy-simulator-support"),
        ("promotion", "duration_weeks"): IntegerDomain(1, 8, "operating-horizon"),
    }


class RegistryTests(unittest.TestCase):
    def test_registry_is_editable_and_requires_integer_parameters(self) -> None:
        original = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
        next(item for item in original if item["id"] == "price_change")["enabled"] = False
        original.append(
            {
                "id": "test_family",
                "label": "Temporary test action",
                "enabled": True,
                "scopes": ["company"],
                "parameters": [{"name": "amount", "dtype": "int", "unit": "units", "validation": "positive"}],
                "simulator_capability": "test_capability",
                "interaction_group": "test",
                "executor": "test_family",
                "future_metadata": {"preserved": True},
            }
        )
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "decisions.json"
            path.write_text(json.dumps(original), encoding="utf-8")
            loaded = load_decisions(path)
            self.assertFalse(loaded.get("price_change").enabled)
            self.assertTrue(loaded.get("test_family").metadata["future_metadata"]["preserved"])
            original[-1]["parameters"][0]["dtype"] = "float"
            path.write_text(json.dumps(original), encoding="utf-8")
            with self.assertRaisesRegex(RegistryValidationError, "dtype must be 'int'"):
                load_decisions(path)


class CandidateAndValidityTests(unittest.TestCase):
    def test_arbitrary_integer_is_valid_without_a_static_grid(self) -> None:
        registry = load_decisions(REGISTRY_PATH)
        validator = ActionValidator(registry, {"action_conditioned_price_and_demand"}, domains())
        result = validator.validate(price_action(7), product_state())
        self.assertTrue(result.valid, result.reasons)
        self.assertFalse(validator.validate(price_action(-101), product_state()).valid)

    def test_generator_uses_runtime_domain_and_masks_unsupported_families(self) -> None:
        registry = load_decisions(REGISTRY_PATH)
        runtime_domains = {
            ("price_change", "change_pct"): IntegerDomain(7, 7, "bounded experiment"),
        }
        validator = ActionValidator(registry, {"action_conditioned_price_and_demand"}, {})
        generator = CandidateGenerator(registry, validator, proposals_per_step=8, seed=11)
        context = ProposalContext(
            targets_by_scope={"product": (Target(branch_id="Garden", product_id="Iced Latte"),)},
            integer_domains=runtime_domains,
            sourced_parameter_values={
                ("price_change", "change_pct", Target(branch_id="Garden", product_id="Iced Latte")): (7,)
            },
            evidence_by_target={
                ("price_change", "product", Target(branch_id="Garden", product_id="Iced Latte")): price_evidence(7)
            },
        )
        pool = generator.generate(product_state(), context)
        self.assertEqual(len(pool.candidates), 1)
        self.assertEqual(pool.candidates[0].parameters["change_pct"], 7)
        self.assertEqual({item.decision_id for item in pool.candidates}, {"price_change"})

    def test_revenue_only_generator_excludes_review_actions(self) -> None:
        registry = load_decisions(REGISTRY_PATH)
        runtime_domains = {
            ("price_change", "change_pct"): IntegerDomain(7, 7, "bounded experiment"),
        }
        simulator = SyntheticSimulator(SyntheticResponseProfile())
        validator = ActionValidator(registry, simulator.supported_capabilities, {})
        generator = CandidateGenerator(
            registry,
            validator,
            proposals_per_step=8,
            seed=11,
            revenue_changing_only=True,
        )
        target = Target(branch_id="Garden", product_id="Iced Latte")
        context = ProposalContext(
            targets_by_scope={"product": (target,)},
            integer_domains=runtime_domains,
            sourced_parameter_values={
                ("price_change", "change_pct", target): (7,),
            },
            evidence_by_target={
                ("price_change", "product", target): price_evidence(7),
                ("branch_price_variance_review", "product", target): {
                    "observation": {"listed_unit_price_usd": 10.0},
                    "model_expectation": {"expected_listed_unit_price_usd": 9.0},
                    "residual_signals": {"price_residual_pct": 11.1},
                    "prediction_quality": {"complete": True},
                },
            },
        )
        pool = generator.generate(product_state(), context)
        self.assertEqual({item.decision_id for item in pool.candidates}, {"price_change"})
        self.assertTrue(registry.get("price_change").metadata["changes_revenue"])
        self.assertFalse(
            registry.get("branch_price_variance_review").metadata["changes_revenue"]
        )


class SimulatorAndRewardTests(unittest.TestCase):
    def setUp(self) -> None:
        self.simulator = SyntheticSimulator(SyntheticResponseProfile(price_elasticity=-1.0))
        self.state = product_state()

    def test_price_action_changes_price_and_demand_and_reconciles(self) -> None:
        action = price_action(10)
        audit = audit_action_response(
            self.simulator,
            self.state,
            action,
            "action_conditioned_price_and_demand",
            {"id": "same", "units_multiplier": 1.0},
            7,
        )
        self.assertTrue(audit.passed)
        forecast = self.simulator.simulate_next_week(self.state, [action], {"id": "same"}, 7)
        row = forecast.rows[0]
        self.assertAlmostEqual(row.predicted_units, 100 / 1.1)
        self.assertAlmostEqual(row.predicted_effective_price_usd, 11.0)
        self.assertAlmostEqual(row.predicted_revenue_usd, row.predicted_units * 11.0)
        self.assertAlmostEqual(row.predicted_gross_profit_usd, row.predicted_revenue_usd - row.predicted_cogs_usd)
        self.assertIn("synthetic_only", forecast.simulator_support)

    def test_paired_no_action_is_exactly_zero(self) -> None:
        comparison = compare_portfolio(
            self.simulator,
            self.state,
            (),
            {"id": "paired", "units_multiplier": 1.1},
            99,
            RewardConfig(scale=100.0),
        )
        self.assertAlmostEqual(comparison.expected_revenue_change_usd, 0.0)
        self.assertAlmostEqual(comparison.net_incremental_profit_usd, 0.0)
        self.assertAlmostEqual(comparison.reward, 0.0)
        self.assertEqual(comparison.baseline, comparison.policy)

    def test_negative_net_profit_receives_asymmetric_penalty(self) -> None:
        costly_action = CandidateAction(
            decision_id="price_change",
            scope="product",
            target=Target(branch_id="Garden", product_id="Iced Latte"),
            parameters={"change_pct": 7},
            evidence=price_evidence(7),
            implementation_cost_usd=10_000.0,
        )
        normal = compare_portfolio(
            self.simulator,
            self.state,
            (costly_action,),
            {"id": "penalty"},
            5,
            RewardConfig(scale=100.0),
        )
        strong = compare_portfolio(
            self.simulator,
            self.state,
            (costly_action,),
            {"id": "penalty"},
            5,
            RewardConfig(scale=100.0, negative_net_profit_penalty_multiplier=25.0),
        )
        self.assertLess(strong.reward, normal.reward)
        self.assertAlmostEqual(strong.reward, normal.reward * 25.0)

    def test_reporting_reconciles_product_and_company_totals(self) -> None:
        comparison = compare_portfolio(
            self.simulator,
            self.state,
            (price_action(10),),
            {"id": "paired"},
            5,
            RewardConfig(scale=100.0),
        )
        report = build_portfolio_report(
            comparison,
            load_decisions(REGISTRY_PATH),
            split_id="toy-train-only",
            simulator_version="synthetic-test-v1",
        )
        product = report["product_forecasts"][0]
        self.assertAlmostEqual(
            product["delta"]["predicted_revenue_usd"],
            report["company_totals"]["delta_expected_revenue_usd"],
        )
        self.assertEqual(report["validation_status"], "synthetic_unverified_intervention_effects")

    def test_joint_actions_are_simulated_together_and_duration_persists(self) -> None:
        promotion = CandidateAction(
            decision_id="promotion",
            scope="product",
            target=Target(branch_id="Garden", product_id="Iced Latte"),
            parameters={"intensity_pct": 20, "duration_weeks": 2},
        )
        joint = self.simulator.simulate_next_week(
            self.state,
            [price_action(10), promotion],
            {"id": "joint"},
            3,
        )
        separately_summed_units = (
            self.simulator.simulate_next_week(self.state, [price_action(10)], {"id": "joint"}, 3).rows[0].predicted_units
            + self.simulator.simulate_next_week(self.state, [promotion], {"id": "joint"}, 3).rows[0].predicted_units
            - self.state.products[0].expected_units
        )
        self.assertNotAlmostEqual(joint.rows[0].predicted_units, separately_summed_units)
        next_state = self.simulator.advance_state(self.state, joint, (promotion,))
        self.assertEqual(next_state.active_actions[0].parameters["duration_weeks"], 1)
        second_week = self.simulator.simulate_next_week(
            next_state,
            list(next_state.active_actions),
            {"id": "second"},
            4,
        )
        self.assertAlmostEqual(second_week.rows[0].predicted_units, 110.0)


class EnvironmentTests(unittest.TestCase):
    def test_stop_advances_week_and_mask_padding_is_invalid(self) -> None:
        registry = load_decisions(REGISTRY_PATH)
        context = ProposalContext(
            targets_by_scope={"product": (Target(branch_id="Garden", product_id="Iced Latte"),)},
            integer_domains={
                ("price_change", "change_pct"): IntegerDomain(7, 7, "toy"),
            },
            sourced_parameter_values={
                ("price_change", "change_pct", Target(branch_id="Garden", product_id="Iced Latte")): (7,)
            },
            evidence_by_target={
                ("price_change", "product", Target(branch_id="Garden", product_id="Iced Latte")): price_evidence(7)
            },
        )
        simulator = SyntheticSimulator(SyntheticResponseProfile(price_elasticity=-1.0))
        validator = ActionValidator(registry, simulator.supported_capabilities, {})
        generator = CandidateGenerator(registry, validator, proposals_per_step=4, seed=3)
        encoder = ObservationEncoder(ObservationSchema(1, 4, 2))
        env = WeeklyDecisionEnv(
            initial_state_provider=product_state,
            context_provider=lambda _state: context,
            scenario_provider=lambda _state, week: {"id": f"week-{week}"},
            simulator=simulator,
            candidate_generator=generator,
            observation_encoder=encoder,
            reward_config=RewardConfig(scale=100.0),
            config=EnvironmentConfig(planning_horizon_weeks=2, max_new_actions_per_week=3),
        )
        observation, _info = env.reset(seed=7)
        self.assertEqual(observation["candidates"].shape, (4, encoder.candidate_width))
        self.assertEqual(env.action_masks().tolist(), [True, True, False, False, False])
        with self.assertRaises(InvalidEnvironmentAction):
            env.step(4)
        _observation, reward, terminated, truncated, info = env.step(0)
        self.assertEqual(reward, 0.0)
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertTrue(info["calendar_advanced"])

        env.reset(seed=7)
        _observation, reward, terminated, _truncated, info = env.step(1)
        self.assertEqual(reward, 0.0)
        self.assertFalse(terminated)
        self.assertFalse(info["calendar_advanced"])
        self.assertEqual(env.action_masks().tolist(), [True, False, False, False, False])
        _observation, reward, _terminated, _truncated, info = env.step(0)
        self.assertGreater(reward, 0.0)
        self.assertTrue(info["calendar_advanced"])

        strict_env = WeeklyDecisionEnv(
            initial_state_provider=product_state,
            context_provider=lambda _state: context,
            scenario_provider=lambda _state, week: {"id": f"strict-week-{week}"},
            simulator=simulator,
            candidate_generator=generator,
            observation_encoder=encoder,
            reward_config=RewardConfig(scale=100.0),
            config=EnvironmentConfig(
                planning_horizon_weeks=2,
                max_new_actions_per_week=10,
                minimum_actions_before_stop=3,
            ),
        )
        strict_env.reset(seed=7)
        self.assertFalse(strict_env.action_masks()[0])
        strict_env.step(1)
        # The sole candidate now conflicts with the selected portfolio, so the
        # emergency STOP prevents an invalid all-false mask before the minimum.
        self.assertTrue(strict_env.action_masks()[0])

    def test_repository_config_sets_three_before_stop_and_ten_maximum(self) -> None:
        config = load_config(ROOT / "config" / "ppo_decision.json")
        self.assertEqual(config.minimum_actions_before_stop, 3)
        self.assertEqual(config.max_new_actions_per_week, 10)


class SplitGuardTests(unittest.TestCase):
    def test_test_sentinel_cannot_be_used_for_training_and_test_is_once(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            paths = {}
            for split in Split:
                path = root / f"{split.value}.txt"
                path.write_text("TEST_SENTINEL" if split is Split.TEST else split.value, encoding="utf-8")
                paths[split] = path
            guard = SplitGuard(paths)
            with self.assertRaises(SplitAccessError):
                guard.read_bytes(Split.TEST, Purpose.PPO_TRAINING)
            with self.assertRaises(SplitAccessError):
                guard.read_bytes(Split.TEST, Purpose.FINAL_FROZEN_EVALUATION)
            guard.freeze_for_test()
            self.assertEqual(
                guard.read_bytes(Split.TEST, Purpose.FINAL_FROZEN_EVALUATION),
                b"TEST_SENTINEL",
            )
            with self.assertRaises(SplitAccessError):
                guard.read_bytes(Split.TEST, Purpose.FINAL_FROZEN_EVALUATION)
            with self.assertRaises(SplitAccessError):
                guard.read_bytes(Split.TRAIN, Purpose.PPO_TRAINING)


if __name__ == "__main__":
    unittest.main()
