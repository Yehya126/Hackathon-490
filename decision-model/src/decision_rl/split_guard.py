"""Explicit split-role enforcement and append-only access records."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Callable, TypeVar


class SplitAccessError(PermissionError):
    pass


class Split(str, Enum):
    TRAIN = "train"
    VALIDATE = "validation"
    TEST = "test"


class Purpose(str, Enum):
    FIT_PREPROCESSOR = "fit_preprocessor"
    FIT_SIMULATOR = "fit_simulator"
    PPO_TRAINING = "ppo_training"
    VALIDATION_SELECTION = "validation_selection"
    FINAL_FROZEN_EVALUATION = "final_frozen_evaluation"
    METADATA_AUDIT = "metadata_audit"


_ALLOWED = {
    Purpose.FIT_PREPROCESSOR: {Split.TRAIN},
    Purpose.FIT_SIMULATOR: {Split.TRAIN},
    Purpose.PPO_TRAINING: {Split.TRAIN},
    Purpose.VALIDATION_SELECTION: {Split.VALIDATE},
    Purpose.FINAL_FROZEN_EVALUATION: {Split.TEST},
    Purpose.METADATA_AUDIT: {Split.TRAIN, Split.VALIDATE, Split.TEST},
}


@dataclass(frozen=True)
class AccessRecord:
    timestamp_utc: str
    split: str
    purpose: str
    path: str
    sha256: str


T = TypeVar("T")


class SplitGuard:
    def __init__(
        self,
        paths: dict[Split, str | Path],
        audit_log: str | Path | None = None,
    ) -> None:
        if set(paths) != set(Split):
            raise ValueError("SplitGuard requires exactly train, validation and test paths")
        self.paths = {split: Path(path).resolve() for split, path in paths.items()}
        self.audit_log = Path(audit_log).resolve() if audit_log is not None else None
        self._frozen = False
        self._test_accessed = False

    def freeze_for_test(self) -> None:
        """Freeze the process before its one permitted final TEST access."""

        self._frozen = True

    def _authorize(self, split: Split, purpose: Purpose) -> None:
        if split not in _ALLOWED[purpose]:
            raise SplitAccessError(f"{purpose.value} cannot read the {split.value} split")
        if purpose is Purpose.FINAL_FROZEN_EVALUATION:
            if not self._frozen:
                raise SplitAccessError("Final TEST evaluation requires a frozen setup")
            if self._test_accessed:
                raise SplitAccessError("TEST has already been accessed for final evaluation")
            self._test_accessed = True
        elif self._frozen and purpose is not Purpose.METADATA_AUDIT:
            raise SplitAccessError("No fitting, training, or selection is allowed after freezing for TEST")

    def read_bytes(self, split: Split, purpose: Purpose) -> bytes:
        self._authorize(split, purpose)
        path = self.paths[split]
        payload = path.read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        if self.audit_log is not None:
            self.audit_log.parent.mkdir(parents=True, exist_ok=True)
            record = AccessRecord(
                timestamp_utc=datetime.now(timezone.utc).isoformat(),
                split=split.value,
                purpose=purpose.value,
                path=str(path),
                sha256=digest,
            )
            with self.audit_log.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(asdict(record), sort_keys=True) + "\n")
        return payload

    def load(self, split: Split, purpose: Purpose, loader: Callable[[Path], T]) -> T:
        """Authorize a path-based loader and log the exact pre-load fingerprint."""

        self.read_bytes(split, purpose)
        return loader(self.paths[split])


def verify_manifest_file(path: str | Path, expected_sha256: str) -> None:
    actual = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    if actual != expected_sha256:
        raise ValueError(f"Split fingerprint mismatch for {Path(path)}: {actual} != {expected_sha256}")
