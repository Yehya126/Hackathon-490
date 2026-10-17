"""Explicit commands: train, export, validate, evaluate-test. No automatic fitting."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pandas as pd
from mohammad.model import DEFAULTS, combination_table, evaluate_table, fit_model

WORKSPACE = Path(__file__).resolve().parent


def read_split(name):
    path = WORKSPACE / 'data' / f'{name}.csv'
    manifest = json.loads((WORKSPACE / 'split_manifest.json').read_text())
    if hashlib.sha256(path.read_bytes()).hexdigest() != manifest['splits'][name]['sha256']:
        raise ValueError(f'{name} differs from the shared split manifest')
    return pd.read_csv(path)


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    train = commands.add_parser('train')
    for name, default in DEFAULTS.items():
        train.add_argument(f'--min-{name}', type=int if name == 'count' else float, default=default)
    train.add_argument('--min-branch-transactions', type=int, default=200)
    train.add_argument('--max-length', type=int, default=3)
    for command in ['export', 'validate', 'evaluate-test']:
        commands.add_parser(command)
    args = parser.parse_args()
    output = WORKSPACE / 'artifacts'
    artifact = output / 'model.json'
    if args.command == 'train':
        if artifact.exists():
            raise ValueError('Archive existing artifacts before a new experiment')
        limits = {name: getattr(args, f'min_{name}') for name in DEFAULTS}
        if (not 0 < limits['support'] <= 1 or not 0 <= limits['confidence'] <= 1
                or limits['count'] < 1 or limits['lift'] <= 1 or limits['profit'] < 0):
            raise ValueError('Invalid thresholds: lift must exceed 1; profit must be nonnegative')
        model = fit_model(read_split('train'), limits['support'], args.min_branch_transactions, args.max_length)
        model['thresholds'] = limits
        model['split_manifest'] = json.loads((WORKSPACE / 'split_manifest.json').read_text())
        save(artifact, model)
    else:
        model = json.loads(artifact.read_text(encoding='utf-8'))
        if model.get('version') != 2:
            raise ValueError('This command requires a combination-table model (version 2)')
        if model['split_manifest'] != json.loads((WORKSPACE / 'split_manifest.json').read_text()):
            raise ValueError('Splits changed since training')
    if args.command in ('train', 'export'):
        table = combination_table(model)
        table.to_csv(output / 'combinations.csv', index=False, float_format='%.6f')
        (output / 'combinations.html').write_text(table.to_html(index=False, float_format=lambda v: f'{v:.4f}'), encoding='utf-8')
        print(f'Exported {len(table)} combinations to {output}')
    else:
        split = 'validation' if args.command == 'validate' else 'test'
        report = output / f'{split}_results.json'
        if split == 'test' and report.exists():
            raise ValueError('Final test evaluation already exists')
        save(report, evaluate_table(model, read_split(split)))
        print(f'Saved combination stability report to {report}')


if __name__ == '__main__':
    main()
