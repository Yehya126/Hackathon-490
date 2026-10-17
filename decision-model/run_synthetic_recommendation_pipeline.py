"""Train the 3-to-10 action policy in an explicitly synthetic sandbox."""

from __future__ import annotations

import json
import os
import sys
from datetime import timedelta
from pathlib import Path
from typing import Any, Callable

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "src"))

from decision_rl.candidate_proposals import CandidateGenerator, ProposalContext
from decision_rl.environment import EnvironmentConfig, WeeklyDecisionEnv
from decision_rl.observation import ObservationEncoder, ObservationSchema
from decision_rl.registry import load_decisions
from decision_rl.reward import RewardConfig
from decision_rl.schemas import BusinessState, ProductState, Target
from decision_rl.synthetic_simulator import SyntheticResponseProfile, SyntheticSimulator
from decision_rl.validity import ActionValidator

from run_training_pipeline import load_split, reward_scale_from_train


REGISTRY_PATH = HERE / "config" / "decisions.json"
ASSUMPTIONS_PATH = HERE / "config" / "synthetic_demo_assumptions.json"
OUT = HERE / "outputs" / "synthetic_demo"


def build_state(lines: pd.DataFrame, split: str) -> BusinessState:
    data = lines.copy()
    data["Date"] = pd.to_datetime(data["Date"])
    weeks = max(1.0, data["Date"].nunique() / 7.0)
    products = []
    for (branch, product, category), group in data.groupby(
        ["Branch", "Product", "Category"], observed=True
    ):
        products.append(
            ProductState(
                branch_id=str(branch),
                product_id=str(product),
                category_id=str(category),
                expected_units=float(group["Quantity"].sum() / weeks),
                listed_price_usd=float(group["Unit Price (USD)"].median()),
                unit_cost_usd=float(group["Unit Cost (USD)"].median()),
                discount_rate=float(group["Discount Rate"].median()),
                features={"source_rows": float(len(group))},
            )
        )
    return BusinessState(
        week_start=data["Date"].max().date() + timedelta(days=1),
        products=tuple(products),
        company_features={
            "historical_revenue_usd": float(data["Line Revenue (USD)"].sum()),
            "historical_gross_profit_usd": float(data["Line Gross Profit (USD)"].sum()),
            "source_days": float(data["Date"].nunique()),
        },
        provenance={"split": split, "rows": len(data)},
    )


def _training_combinations(limit_per_branch: int = 10) -> list[dict[str, Any]]:
    path = ROOT / "Marketing_Expert" / "artifacts" / "validation_review" / "product_candidate_model.json"
    model = json.loads(path.read_text(encoding="utf-8"))
    result = []
    for branch in ("Garden", "Harbour", "Uptown"):
        candidates = []
        for combination in model["scopes"][branch]["combinations"]:
            rules = [rule for rule in combination["rules"] if rule["lift"] > 1]
            if combination["count"] < 10 or not rules:
                continue
            rule = max(rules, key=lambda item: (item["lift"], item["confidence"]))
            score = combination["support"] * (combination["price"] - combination["cost"])
            candidates.append((score, combination, rule))
        for _score, combination, rule in sorted(candidates, reverse=True, key=lambda x: x[0])[:limit_per_branch]:
            result.append(
                {
                    "branch": branch,
                    "products": tuple(combination["products"]),
                    "support": combination["support"],
                    "confidence": rule["confidence"],
                    "lift": rule["lift"],
                    "gross_profit_usd": combination["price"] - combination["cost"],
                    "rule_direction": f"{' + '.join(rule['antecedent'])} -> {rule['target']}",
                    "consequent_product": rule["target"],
                    "assessment": "TRAIN candidate for synthetic optimization",
                }
            )
    return result


def _validated_combinations() -> list[dict[str, Any]]:
    path = ROOT / "Marketing_Expert" / "artifacts" / "validation_review" / "product_comparison.csv"
    data = pd.read_csv(path)
    valid = data[
        data["assessment"].eq("Promising for review (not proven)")
        & data["lift_train"].gt(1)
        & data["lift_validation"].gt(1)
    ]
    result = []
    for _, row in valid.iterrows():
        products = tuple(str(row["combination"]).split(" + "))
        direction = str(row["rule_direction"])
        consequent = direction.split(" -> ")[-1]
        branches = ("Garden", "Harbour", "Uptown") if row["branch"] == "Overall" else (row["branch"],)
        for branch in branches:
            result.append(
                {
                    "branch": branch,
                    "products": products,
                    "support": float(row["support_validation"]),
                    "confidence": float(row["confidence_validation"]),
                    "lift": float(row["lift_validation"]),
                    "gross_profit_usd": float(row["gross_profit_usd"]),
                    "rule_direction": direction,
                    "consequent_product": consequent,
                    "assessment": "Promising on TRAIN and VALIDATION; not proven",
                }
            )
    return result


def build_context(
    state: BusinessState,
    combinations: list[dict[str, Any]],
    assumptions: dict[str, Any],
) -> ProposalContext:
    branch_targets: set[Target] = set()
    evidence: dict[tuple[str, str, Target], dict[str, Any]] = {}
    costs: dict[tuple[str, str, Target], float] = {}
    for item in combinations:
        target = Target(branch_id=item["branch"], bundle_product_ids=item["products"])
        branch_targets.add(target)
        score = min(1.0, max(0.0, (item["lift"] - 1) * 5 + item["confidence"]))
        base_evidence = {**item, "learning_value_score": score}
        for decision_id, cost_name in (
            ("bundle_create", "bundle_implementation_cost_usd"),
            ("cross_sell", "cross_sell_implementation_cost_usd"),
        ):
            key = (decision_id, "branch", target)
            evidence[key] = base_evidence
            costs[key] = float(assumptions[cost_name])
    return ProposalContext(
        targets_by_scope={
            "branch": tuple(sorted(branch_targets)),
        },
        integer_domains={},
        evidence_by_target=evidence,
        implementation_costs_usd=costs,
    )


def make_environment(
    state: BusinessState,
    context: ProposalContext,
    scale: float,
    assumptions: dict[str, Any],
    seed: int,
) -> WeeklyDecisionEnv:
    profile = SyntheticResponseProfile(
        price_elasticity=float(assumptions["price_elasticity"]),
        bundle_uptake_rate=float(assumptions["bundle_uptake_rate"]),
        bundle_substitution_rate=float(assumptions["bundle_substitution_rate"]),
        cross_sell_acceptance_rate=float(assumptions["cross_sell_acceptance_rate"]),
        cross_sell_cannibalization_rate=float(assumptions["cross_sell_cannibalization_rate"]),
    )
    simulator = SyntheticSimulator(profile)
    registry = load_decisions(REGISTRY_PATH)
    validator = ActionValidator(registry, simulator.supported_capabilities, {})
    generator = CandidateGenerator(
        registry,
        validator,
        proposals_per_step=64,
        seed=seed,
        revenue_changing_only=True,
    )
    encoder = ObservationEncoder(
        ObservationSchema(
            max_products=len(state.products),
            max_candidates=64,
            max_parameters_per_action=2,
            company_feature_names=("historical_revenue_usd", "historical_gross_profit_usd", "source_days"),
            product_feature_names=("source_rows",),
            candidate_evidence_names=("learning_value_score", "support", "confidence", "lift", "gross_profit_usd"),
        )
    )
    return WeeklyDecisionEnv(
        initial_state_provider=lambda: state,
        context_provider=lambda _state: context,
        scenario_provider=lambda _state, week: {"id": f"{state.provenance['split']}-{week}"},
        simulator=simulator,
        candidate_generator=generator,
        observation_encoder=encoder,
        reward_config=RewardConfig(
            scale=scale,
            # Evidence determines candidate eligibility, but cannot create reward.
            # The PPO objective is financial once candidates enter this environment.
            evidence_utility_weight=0.0,
            action_count_penalty=float(assumptions["action_count_penalty"]),
            negative_net_profit_penalty_multiplier=float(
                assumptions["negative_net_profit_penalty_multiplier"]
            ),
        ),
        config=EnvironmentConfig(
            planning_horizon_weeks=1,
            max_new_actions_per_week=10,
            minimum_actions_before_stop=3,
            seed=seed,
        ),
    )


def env_factory(state, context, scale, assumptions, seed) -> Callable[[], WeeklyDecisionEnv]:
    return lambda: make_environment(state, context, scale, assumptions, seed)


def evaluate(model, state, context, scale, assumptions) -> dict[str, Any]:
    from sb3_contrib.common.maskable.utils import get_action_masks

    env = make_environment(state, context, scale, assumptions, 42)
    observation, _ = env.reset(seed=42)
    done = False
    while not done:
        action, _ = model.predict(
            observation,
            action_masks=get_action_masks(env),
            deterministic=True,
        )
        observation, reward, done, _truncated, info = env.step(int(action))
    comparison = info["portfolio_comparison"]
    return {
        "actions": [item.as_dict() for item in comparison.new_actions],
        "action_count": len(comparison.new_actions),
        "synthetic_baseline_revenue_usd": comparison.baseline.total_revenue,
        "synthetic_policy_revenue_usd": comparison.policy.total_revenue,
        "synthetic_revenue_change_usd": comparison.expected_revenue_change_usd,
        "synthetic_baseline_gross_profit_usd": comparison.baseline.total_gross_profit,
        "synthetic_policy_gross_profit_usd": comparison.policy.total_gross_profit,
        "synthetic_gross_profit_change_usd": comparison.expected_gross_profit_change_usd,
        "synthetic_implementation_costs_usd": comparison.implementation_costs_usd,
        "synthetic_net_incremental_profit_usd": comparison.net_incremental_profit_usd,
        "evidence_utility": comparison.evidence_utility,
        "reward": float(reward),
        "simulator_support": comparison.policy.simulator_support,
    }


def main() -> None:
    from sb3_contrib import MaskablePPO
    from stable_baselines3.common.vec_env import SubprocVecEnv

    assumptions = json.loads(ASSUMPTIONS_PATH.read_text(encoding="utf-8"))
    train = load_split("train")
    validation = load_split("validation")
    train_state = build_state(train, "train")
    validation_state = build_state(validation, "validation")
    scale = reward_scale_from_train(train)
    train_context = build_context(train_state, _training_combinations(), assumptions)
    validation_combinations = _validated_combinations()
    validation_context = build_context(validation_state, validation_combinations, assumptions)

    workers = min(4, max(1, (os.cpu_count() or 2) // 2))
    vec = SubprocVecEnv(
        [env_factory(train_state, train_context, scale, assumptions, 100 + i) for i in range(workers)],
        start_method="spawn",
    )
    model = MaskablePPO(
        "MultiInputPolicy",
        vec,
        seed=42,
        gamma=0.99,
        n_steps=128,
        batch_size=128,
        n_epochs=6,
        learning_rate=3e-4,
        verbose=1,
        device="cpu",
    )
    model.learn(total_timesteps=8192)
    OUT.mkdir(parents=True, exist_ok=True)
    checkpoint = OUT / "maskable_ppo_synthetic_3_to_10"
    model.save(str(checkpoint))
    vec.close()

    validation_result = evaluate(model, validation_state, validation_context, scale, assumptions)

    # Final synthetic scenario uses the frozen validated combinations and reads TEST once.
    test = load_split("test")
    test_state = build_state(test, "test")
    test_context = build_context(test_state, validation_combinations, assumptions)
    test_result = evaluate(model, test_state, test_context, scale, assumptions)
    report = {
        "status": "synthetic_demo_only_unvalidated_intervention_effects",
        "parallel_rollout_workers": workers,
        "training_timesteps": 8192,
        "reward_scale_train_only_usd": scale,
        "minimum_actions_before_stop": 3,
        "maximum_actions_per_week": 10,
        "assumptions": assumptions,
        "training_candidate_source": "Marketing_Expert TRAIN-only product_candidate_model",
        "validation_candidate_source": "Marketing_Expert TRAIN-to-VALIDATION comparison",
        "action_eligibility": "Only registry decisions with changes_revenue=true",
        "validated_marketing_combinations": len(validation_combinations),
        "validation": validation_result,
        "final_test_synthetic": test_result,
        "limitations": [
            "All intervention effects and implementation costs are synthetic assumptions.",
            "Review-only actions are excluded from the PPO candidate pool.",
            "The reported uplift is simulated and is not an observed TEST improvement.",
            "A controlled experiment is required before business deployment.",
        ],
        "checkpoint": str(checkpoint.with_suffix(".zip")),
    }
    report_path = OUT / "synthetic_policy_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
