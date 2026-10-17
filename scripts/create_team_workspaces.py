"""Create identical, reproducible dataset splits for each team member."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(PROJECT_ROOT))

from data_utils import load_data, split_data


MEMBERS = ("mohammad", "yehya", "aya", "PricingOptimization")
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

    # Write each canonical split once, then copy its exact bytes for the other
    # members so checksums—not merely transaction memberships—are identical.
    first_member = PROJECT_ROOT / MEMBERS[0] / "data"
    first_member.mkdir(parents=True, exist_ok=True)
    for split_name in SPLIT_NAMES:
        frame = getattr(splits, split_name).drop(columns=list(DERIVED_COLUMNS), errors="ignore")
        output = first_member / f"{split_name}.csv"
        frame.to_csv(output, index=True, index_label="Source Row ID", date_format="%Y-%m-%d")
        manifest["splits"][split_name] = {
            "rows": len(frame),
            "transactions": frame["Transaction ID"].nunique(),
            "first_date": frame["Date"].min().strftime("%Y-%m-%d"),
            "last_date": frame["Date"].max().strftime("%Y-%m-%d"),
            "sha256": sha256(output),
        }

    for member in MEMBERS[1:]:
        member_data = PROJECT_ROOT / member / "data"
        member_data.mkdir(parents=True, exist_ok=True)
        for split_name in SPLIT_NAMES:
            source = first_member / f"{split_name}.csv"
            destination = member_data / source.name
            destination.write_bytes(source.read_bytes())

    manifest_text = json.dumps(manifest, indent=2) + "\n"
    for member in MEMBERS:
        (PROJECT_ROOT / member / "split_manifest.json").write_text(manifest_text, encoding="utf-8")

    for member in MEMBERS:
        for split_name in SPLIT_NAMES:
            path = PROJECT_ROOT / member / "data" / f"{split_name}.csv"
            expected = manifest["splits"][split_name]["sha256"]
            if sha256(path) != expected:
                raise RuntimeError(f"Checksum mismatch: {path}")

    print(f"Created identical splits for: {', '.join(MEMBERS)}")


if __name__ == "__main__":
    main()

