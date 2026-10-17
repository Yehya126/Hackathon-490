"""Train/validate/final-test the currently supportable decision policy.

The supplied downstream artifacts contain TEST/full-data information and are
therefore audited only after the policy is frozen. With no validated
action-response capability, STOP is the only feasible action in this run.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from dataclasses import asdict
from datetime import date, timedelta
from pathlib import Path
from typing import Callable

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from decision_rl.candidate_proposals import CandidateGenerator, ProposalContext
from decision_rl.environment import EnvironmentConfig, WeeklyDecisionEnv
from decision_rl.observation import ObservationEncoder, ObservationSchema
from decision_rl.registry import load_decisions
from decision_rl.reward import RewardConfig
from decision_rl.schemas import BusinessState, ProductState
from decision_rl.synthetic_simulator import SyntheticSimulator
from decision_rl.validity import ActionValidator


DATA = ROOT / "data"
OUT = Path(__file__).resolve().parent / "outputs"
REGISTRY = Path(__file__).resolve().parent / "config" / "decisions.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_split(name: str) -> pd.DataFrame:
    manifest = json.loads((DATA / "split_manifest.json").read_text(encoding="utf-8"))
    path = DATA / f"{name}.csv"
    expected = manifest["splits"][name]["sha256"]
    if sha256(path) != expected:
        raise ValueError(f"{name} fingerprint does not match data/split_manifest.json")
    return pd.read_csv(path)


def build_state(lines: pd.DataFrame, split: str) -> BusinessState:
    data = lines.copy()
    data["Date"] = pd.to_datetime(data["Date"])
    days = max(1, int(data["Date"].nunique()))
    weeks = days / 7.0
    grouped = data.groupby(["Branch", "Product", "Category"], observed=True)
    products = []
    for (branch, product, category), group in grouped:
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
            "source_days": float(days),
        },
        provenance={"split": split, "rows": len(data)},
    )


def reward_scale_from_train(train: pd.DataFrame) -> float:
    data = train.copy()
    data["Date"] = pd.to_datetime(data["Date"])
    weekly = data.groupby(data["Date"].dt.to_period("W"))["Line Gross Profit (USD)"].sum()
    value = float(weekly.median())
    if value <= 0:
        raise ValueError("TRAIN-derived reward scale must be positive")
    return value


def make_environment(state: BusinessState, reward_scale: float, seed: int) -> WeeklyDecisionEnv:
    registry = load_decisions(REGISTRY)
    # Deliberately expose no intervention capability: none of the supplied
    # artifacts models an action-conditioned response.
    validator = ActionValidator(registry, supported_capabilities=(), domains={})
    generator = CandidateGenerator(registry, validator, proposals_per_step=64, seed=seed)
    encoder = ObservationEncoder(
        ObservationSchema(
            max_products=len(state.products),
            max_candidates=64,
            max_parameters_per_action=2,
            company_feature_names=(
                "historical_revenue_usd",
                "historical_gross_profit_usd",
                "source_days",
            ),
            product_feature_names=("source_rows",),
        )
    )
    empty_context = ProposalContext(targets_by_scope={}, integer_domains={})
    return WeeklyDecisionEnv(
        initial_state_provider=lambda: state,
        context_provider=lambda _state: empty_context,
        scenario_provider=lambda _state, week: {"id": f"{state.provenance['split']}-week-{week}"},
        simulator=SyntheticSimulator(),
        candidate_generator=generator,
        observation_encoder=encoder,
        reward_config=RewardConfig(scale=reward_scale),
        config=EnvironmentConfig(
            planning_horizon_weeks=8,
            max_new_actions_per_week=10,
            minimum_actions_before_stop=3,
            seed=seed,
        ),
    )


def env_factory(state: BusinessState, reward_scale: float, seed: int) -> Callable[[], WeeklyDecisionEnv]:
    return lambda: make_environment(state, reward_scale, seed)


def evaluate_policy(model, state: BusinessState, reward_scale: float) -> dict[str, object]:
    from sb3_contrib.common.maskable.utils import get_action_masks

    env = make_environment(state, reward_scale, 42)
    observation, _ = env.reset(seed=42)
    actions = []
    total_reward = 0.0
    terminated = False
    while not terminated:
        mask = get_action_masks(env)
        action, _ = model.predict(observation, action_masks=mask, deterministic=True)
        observation, reward, terminated, _truncated, info = env.step(int(action))
        total_reward += float(reward)
        comparison = info.get("portfolio_comparison")
        if comparison is not None:
            actions.extend(item.as_dict() for item in comparison.new_actions)
    return {"total_reward": total_reward, "actions": actions, "recommended": "stop" if not actions else "actions"}


def audit_supplied_artifacts() -> dict[str, object]:
    branch_path = ROOT / "branch-analysis-model" / "outputs" / "decision_model_input.json"
    marketing_path = ROOT / "Marketing_Expert" / "artifacts" / "full_data" / "product_model.json"
    merchandise_path = ROOT / "Merchandise" / "outputs" / "summary.json"
    branch = json.loads(branch_path.read_text(encoding="utf-8"))
    marketing = json.loads(marketing_path.read_text(encoding="utf-8"))
    merchandise = json.loads(merchandise_path.read_text(encoding="utf-8"))
    screened_path = ROOT / "Marketing_Expert" / "artifacts" / "validation_review" / "product_screened.csv"
    screened = pd.read_csv(screened_path)
    approved = int(screened["assessment"].astype(str).eq("Include for owner review").sum())
    return {
        "branch_analysis": {
            "path": str(branch_path),
            "sha256": sha256(branch_path),
            "data_window": branch.get("data_window"),
            "action_conditioned": False,
        },
        "marketing_expert": {
            "path": str(marketing_path.parent),
            "model_sha256": sha256(marketing_path),
            "training_scope": marketing.get("training_scope"),
            "provenance": marketing.get("provenance"),
            "validated_owner_review_candidates": approved,
            "action_conditioned": False,
        },
        "merchandise": {
            "path": str(merchandise_path.parent),
            "summary_sha256": sha256(merchandise_path),
            "test_period": merchandise.get("test_period"),
            "model": merchandise.get("model"),
            "business_as_usual_only": True,
            "action_conditioned": False,
        },
    }


def main() -> None:
    from sb3_contrib import MaskablePPO
    from stable_baselines3.common.vec_env import SubprocVecEnv

    OUT.mkdir(parents=True, exist_ok=True)
    train = load_split("train")
    validation = load_split("validation")
    scale = reward_scale_from_train(train)
    train_state = build_state(train, "train")
    validation_state = build_state(validation, "validation")

    worker_count = min(4, max(1, (os.cpu_count() or 2) // 2))
    vector_env = SubprocVecEnv(
        [env_factory(train_state, scale, 42 + rank) for rank in range(worker_count)],
        start_method="spawn",
    )
    model = MaskablePPO(
        "MultiInputPolicy",
        vector_env,
        seed=42,
        gamma=0.99,
        n_steps=64,
        batch_size=64,
        n_epochs=4,
        learning_rate=3e-4,
        verbose=1,
        device="cpu",
    )
    model.learn(total_timesteps=1024)
    checkpoint = OUT / "checkpoints" / "maskable_ppo_stop_only"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(checkpoint))
    vector_env.close()

    validation_result = evaluate_policy(model, validation_state, scale)

    # Freeze before the first and only raw TEST load in this process.
    test = load_split("test")
    test_state = build_state(test, "test")
    test_result = evaluate_policy(model, test_state, scale)
    actual_revenue = float(test["Line Revenue (USD)"].sum())
    actual_profit = float(test["Line Gross Profit (USD)"].sum())
    artifacts = audit_supplied_artifacts()

    report = {
        "status": "completed_stop_only_no_validated_intervention",
        "parallel_rollout_workers": worker_count,
        "training_timesteps": 1024,
        "reward_scale_train_only_usd": scale,
        "checkpoint": str(checkpoint.with_suffix(".zip")),
        "validation": validation_result,
        "final_test": {
            **test_result,
            "actual_business_as_usual_revenue_usd": actual_revenue,
            "actual_business_as_usual_gross_profit_usd": actual_profit,
            "policy_revenue_usd": actual_revenue,
            "policy_gross_profit_usd": actual_profit,
            "revenue_improvement_usd": 0.0,
            "gross_profit_improvement_usd": 0.0,
            "improvement_pct": 0.0,
            "interpretation": "STOP was the only supported action, so policy and baseline are identical.",
        },
        "supplied_artifact_audit": artifacts,
        "limitations": [
            "No supplied model estimates action-conditioned demand or causal intervention effects.",
            "Branch decision input is TEST-derived and Marketing full_data includes former TEST; both were audited only after policy freeze.",
            "Merchandise forecasts business-as-usual units and cannot score bundle/cross-sell/price interventions.",
            "The zero improvement is an exact no-action comparison, not evidence that candidate interventions have zero effect.",
        ],
    }
    destination = OUT / "frozen_test_report" / "decision_policy_evaluation.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
