"""Create one canonical, reproducible train/validation/test split."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(PROJECT_ROOT))

from data_utils import load_data, split_data


SPLIT_NAMES = ("train", "validation", "test")
DERIVED_COLUMNS = ("Transaction Timestamp",)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    data = load_data(PROJECT_ROOT / "stories_transactions.csv")
    splits = split_data(data)
    manifest: dict[str, object] = {
        "source": "stories_transactions.csv",
        "strategy": "chronological transaction-level 70/15/15 split",
        "splits": {},
    }

    # Write each canonical split once. Every model reads this shared directory.
    data_dir = PROJECT_ROOT / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    for split_name in SPLIT_NAMES:
        frame = getattr(splits, split_name).drop(columns=list(DERIVED_COLUMNS), errors="ignore")
        output = data_dir / f"{split_name}.csv"
        frame.to_csv(output, index=True, index_label="Source Row ID", date_format="%Y-%m-%d")
        manifest["splits"][split_name] = {
            "rows": len(frame),
            "transactions": frame["Transaction ID"].nunique(),
            "first_date": frame["Date"].min().strftime("%Y-%m-%d"),
            "last_date": frame["Date"].max().strftime("%Y-%m-%d"),
            "sha256": sha256(output),
        }

    manifest_text = json.dumps(manifest, indent=2) + "\n"
    (data_dir / "split_manifest.json").write_text(manifest_text, encoding="utf-8")

    for split_name in SPLIT_NAMES:
        path = data_dir / f"{split_name}.csv"
        expected = manifest["splits"][split_name]["sha256"]
        if sha256(path) != expected:
            raise RuntimeError(f"Checksum mismatch: {path}")

    print(f"Created canonical splits in {data_dir}")


if __name__ == "__main__":
    main()

