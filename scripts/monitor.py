"""
Monitoring rule of phase 6: fit the reference fences, then check batches.

    python -m scripts.monitor fit --predictions reports/fase5/predict_all.csv \\
        --splits datasets/interim/splits.csv --out reports/fase6/monitor_reference.json

    python -m scripts.monitor check --predictions <predict.py csv> \\
        --reference reports/fase6/monitor_reference.json --name <label> \\
        [--split test --splits datasets/interim/splits.csv]

Each check appends one row to --summary (reports/fase6/monitor_summary.csv).
See src/deploy/monitor.py for the rule and its deviation from the plan.
"""
import argparse
from pathlib import Path

import pandas as pd

from src.deploy.monitor import (
    MIN_N, THRESHOLD, box_features, check_batch, false_alarm_prob, fit_reference, load_reference,
    outside_by_feature, reference_rows, save_reference, split_rows,
)
from src.train.loop import PROJECT_ROOT

DEFAULT_SUMMARY = PROJECT_ROOT / 'reports' / 'fase6' / 'monitor_summary.csv'
SUMMARY_COLUMNS = ['name', 'n', 'n_dropped', 'n_out', 'frac_out', 'status', 'alert', 'out_width', 'out_height']
FALSE_ALARM_NS = (20, 50, 153)


def fit(args) -> dict:
    preds = reference_rows(pd.read_csv(args.predictions), pd.read_csv(args.splits))
    feats = box_features(preds)
    ref = fit_reference(feats)
    ref['n_dropped'] = feats.attrs['n_dropped']
    save_reference(ref, args.out)

    print(f"reference: n={ref['n']} (dropped {ref['n_dropped']}), k={ref['k']}")
    for f, b in ref['features'].items():
        print(f"  {f:6s} q1={b['q1']:.1f} q3={b['q3']:.1f} fences=[{b['lo']:.1f}, {b['hi']:.1f}]")
    print(f"base_rate={ref['base_rate']:.4f}")
    print(f"P(false alarm) at threshold {THRESHOLD}:")
    for n in FALSE_ALARM_NS:
        print(f"  n={n:4d}  {false_alarm_prob(ref['base_rate'], n, THRESHOLD):.4f}")
    print(f"-> {args.out}")
    return ref


def check(args) -> dict:
    preds = pd.read_csv(args.predictions)
    if args.split:
        if not args.splits:
            raise SystemExit("--split needs --splits")
        preds = split_rows(preds, pd.read_csv(args.splits), args.split)
    ref = load_reference(args.reference)
    feats = box_features(preds)
    res = check_batch(feats, ref, THRESHOLD, MIN_N)
    by_feat = outside_by_feature(feats, ref).sum()
    row = {'name': args.name, 'n_dropped': feats.attrs['n_dropped'], **res,
           'out_width': int(by_feat['width']), 'out_height': int(by_feat['height'])}

    summary = Path(args.summary)
    summary.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).reindex(columns=SUMMARY_COLUMNS).to_csv(
        summary, mode='a', header=not summary.exists(), index=False)
    print(f"{args.name}: n={res['n']} (dropped {row['n_dropped']}), out={res['n_out']} "
          f"({res['frac_out']:.3f}; width {row['out_width']}, height {row['out_height']}) "
          f"-> {res['status']}")
    return row


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Box-size monitoring rule.")
    sub = parser.add_subparsers(dest='cmd', required=True)

    p = sub.add_parser('fit', help="fences from the non-test predictions")
    p.add_argument('--predictions', type=Path, required=True)
    p.add_argument('--splits', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.set_defaults(func=fit)

    p = sub.add_parser('check', help="apply the rule to one batch")
    p.add_argument('--predictions', type=Path, required=True)
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--name', required=True)
    p.add_argument('--summary', type=Path, default=DEFAULT_SUMMARY)
    p.add_argument('--split', help="keep only this split of --predictions")
    p.add_argument('--splits', type=Path)
    p.set_defaults(func=check)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == '__main__':
    main()
