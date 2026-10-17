"""Train and evaluate the decision-neutral Stories cost/price predictor."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.pipeline import PredictorEngine


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--data",
        default=str(Path(__file__).resolve().parents[1] / "data"),
        help="Shared directory containing train.csv, validation.csv, and test.csv",
    )
    p.add_argument("--output", default=str(Path(__file__).parent / "outputs"))
    p.add_argument("--non-strict", action="store_true", help="Report invalid rows as warnings instead of stopping")
    return p


def main() -> None:
    args = parser().parse_args()
    result = PredictorEngine(args.output).run(args.data, strict=not args.non_strict)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
