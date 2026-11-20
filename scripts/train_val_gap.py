import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from scripts.cv import CLASSES, RUNS_DIR
from src.train.loop import (
    KEY,
    PROJECT_ROOT,
    evaluate,
    get_device,
    load_data,
    predict_checkpoint,
    volume_boxes,
)
from src.train.sampler import train_slices

REPORTS_DIR = PROJECT_ROOT / 'reports' / 'fase3'
# The only source of tau: the one the refit and test use
REFIT_CFG = PROJECT_ROOT / 'configs' / 'train' / 'm1_refit.yaml'
GAP_LIMIT = 0.1


def train_predictions(run_dir: Path, slices: pd.DataFrame, device: torch.device) -> pd.DataFrame:
    """
    Predictions of run_dir/best.pt on its own training slices (every dev fold
    but its val_fold), cached in run_dir/preds_train.csv so a rerun resumes.
    """
    cache = Path(run_dir) / 'preds_train.csv'
    if cache.exists():
        return pd.read_csv(cache)
    ck = torch.load(Path(run_dir) / 'best.pt', map_location='cpu', weights_only=False)
    train = train_slices(slices, ck['cfg']['val_fold'])
    predict_checkpoint(ck, train, device).to_csv(cache, index=False)
    # Read back so a first run and a resumed one score the same float64 values
    return pd.read_csv(cache)


def run_gap(run_dir: Path, preds_train: pd.DataFrame, volumes: pd.DataFrame, tau: float) -> dict:
    """Train and val IoU of one CV run, both at the same fixed tau, and their gap."""
    run_dir = Path(run_dir)
    cfg = yaml.safe_load((run_dir / 'config.yaml').read_text())
    fold, seed = run_dir.name.rsplit('_s', 1)
    preds_val = pd.read_csv(run_dir / 'preds_val.csv')

    results = {}
    for name, preds in (('train', preds_train), ('val', preds_val)):
        gt = volume_boxes(volumes.merge(preds[KEY].drop_duplicates(), on=KEY))
        results[name] = (len(gt), evaluate(
            preds, gt, np.asarray(cfg['taus']), tau=tau, **cfg.get('aggregation', {}),
        ))

    (n_train, train), (n_val, val) = results['train'], results['val']
    row = {
        'run': run_dir.name, 'fold': fold, 'seed': int(seed), 'tau': tau,
        'n_train': n_train, 'n_val': n_val,
        'iou_train': train['iou'], 'iou_val': val['iou'], 'gap': train['iou'] - val['iou'],
    }
    for c in CLASSES[1:]:
        t, v = train.get(f'iou_class_{c}', np.nan), val.get(f'iou_class_{c}', np.nan)
        row.update({f'iou_train_class_{c}': t, f'iou_val_class_{c}': v, f'gap_class_{c}': t - v})
    return row


def summarize_gap(table: pd.DataFrame) -> pd.DataFrame:
    """One row per class: means over runs of train IoU, val IoU and gap, and the gap's range."""
    rows = []
    for c in CLASSES:
        suffix = '' if c == 'all' else f'_class_{c}'
        g = table[[f'iou_train{suffix}', f'iou_val{suffix}', f'gap{suffix}']].dropna()
        g.columns = ['iou_train', 'iou_val', 'gap']
        gap = g['gap'].mean()
        rows.append({
            'class': str(c), 'n_runs': len(g),
            'iou_train': g['iou_train'].mean(), 'iou_val': g['iou_val'].mean(), 'gap': gap,
            'gap_min': g['gap'].min(), 'gap_max': g['gap'].max(),
            'meets': bool(gap < GAP_LIMIT),
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train/val IoU gap of the CV runs at the refit's fixed tau.")
    parser.add_argument('--runs-dir', type=Path, default=RUNS_DIR)
    parser.add_argument('--runs', nargs='+', help="Run names, e.g. fold_0_s42; default every run in --runs-dir")
    parser.add_argument('--device', help="cpu or mps; default picks MPS when available")
    args = parser.parse_args()

    refit_cfg = yaml.safe_load(REFIT_CFG.read_text())
    tau = refit_cfg['tau']
    runs = args.runs or sorted(p.name for p in args.runs_dir.iterdir() if (p / 'best.pt').exists())

    dev = get_device(args.device)
    slices, volumes = load_data(refit_cfg)

    rows = []
    for name in runs:
        print(f"=== {name}: train vs val IoU at tau={tau:g}", flush=True)
        run_dir = args.runs_dir / name
        rows.append(run_gap(run_dir, train_predictions(run_dir, slices, dev), volumes, tau))

    table = pd.DataFrame(rows)
    summary = summarize_gap(table)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(REPORTS_DIR / 'train_val_gap.csv', index=False)
    summary.to_csv(REPORTS_DIR / 'train_val_gap_summary.csv', index=False)

    print(f"\nTrain - val IoU over {len(table)} runs (meets: gap < {GAP_LIMIT}):")
    print(summary.round(4).to_string(index=False))
    print(f"\nSaved {REPORTS_DIR}/train_val_gap*.csv")


if __name__ == '__main__':
    main()
