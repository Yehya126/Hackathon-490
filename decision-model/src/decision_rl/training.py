"""Optional MaskablePPO entrypoint. Nothing trains on import or by default."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


class TrainingDisabledError(RuntimeError):
    pass


@dataclass(frozen=True)
class TrainingRequest:
    enabled: bool = False
    timesteps: int = 0
    seed: int = 42
    checkpoint_path: Path | None = None


def train_maskable_ppo(
    environment_factory: Callable[[], Any],
    request: TrainingRequest,
    **ppo_kwargs: Any,
) -> Any:
    """Train only after the caller explicitly enables a fully wired request."""

    if not request.enabled:
        raise TrainingDisabledError("Training is disabled until input sources and simulator support are configured")
    if request.timesteps < 1:
        raise ValueError("Training timesteps must be positive")
    try:
        from sb3_contrib import MaskablePPO
    except ImportError as error:
        raise RuntimeError("Install the decision-model RL dependencies before training") from error
    environment = environment_factory()
    model = MaskablePPO("MultiInputPolicy", environment, seed=request.seed, **ppo_kwargs)
    model.learn(total_timesteps=request.timesteps)
    if request.checkpoint_path is not None:
        request.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(request.checkpoint_path))
    return model
