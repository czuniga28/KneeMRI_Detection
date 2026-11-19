import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from scripts.cv import CLASSES, _std
from src.eval.baseline import fixed_box_test
from src.eval.bootstrap import bootstrap_mean, paired_bootstrap
from src.models.net import SliceNet
from src.train.dataset import SliceDataset
from src.train.loop import (
    KEY,
    PROJECT_ROOT,
    _command,
    evaluate,
    get_device,
    load_data,
    predict,
    volume_boxes,
)

REFIT_ROOT = PROJECT_ROOT / 'runs' / 'refit'
REPORTS_DIR = PROJECT_ROOT / 'reports' / 'fase4'
CV_SUMMARY = PROJECT_ROOT / 'reports' / 'fase3' / 'cv_summary.csv'
# Relative to PROJECT_ROOT, where _command runs git
PREREGISTRATION = 'docs/entregable3/preregistro_test.md'


def check_not_evaluated(reports_dir: Path, force: bool) -> None:
    """Exit if test was already scored: a second look is no longer a held-out estimate."""
    done = Path(reports_dir) / 'test_volumes.csv'
    if done.exists() and not force:
        sys.exit(f"{done} already exists; test has been evaluated. Pass --force to overwrite it.")


def check_preregistered() -> None:
    """Exit unless the preregistration is tracked by git and has no uncommitted changes."""
    tracked = _command(['git', 'ls-files', '--error-unmatch', PREREGISTRATION])
    if tracked.startswith('unavailable'):
        sys.exit(f"{PREREGISTRATION} is not committed; commit the preregistration before scoring test.")
    changes = _command(['git', 'status', '--porcelain', '--', PREREGISTRATION])
    if changes:
        sys.exit(f"{PREREGISTRATION} has uncommitted changes ({changes}); commit them before scoring test.")


def common_config(cfgs: dict[int, dict]) -> dict:
    """
    The config every seed shares; only 'seed' may differ.

    Returns:
        The first seed's config.

    Raises:
        ValueError: Naming the keys that differ, e.g. tau, aggregation or model.
    """
    seeds = list(cfgs)
    first = {k: v for k, v in cfgs[seeds[0]].items() if k != 'seed'}
    for seed in seeds[1:]:
        other = {k: v for k, v in cfgs[seed].items() if k != 'seed'}
        if other != first:
            keys = sorted(k for k in first.keys() | other.keys() if first.get(k) != other.get(k))
            raise ValueError(f"seeds {seeds[0]} and {seed} were refit with different configs: {keys}")
    return cfgs[seeds[0]]


def score_seed(
    cfg: dict, state_dict: dict, test: pd.DataFrame, gt: pd.DataFrame, device: torch.device,
) -> tuple[pd.DataFrame, dict]:
    """Predict test with one refit model and score it at the config's fixed tau."""
    # Weights are overwritten below, so skip the ImageNet download
    model = SliceNet(**{**cfg['model'], 'pretrained': False}).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    ds = SliceDataset(
        test, PROJECT_ROOT / cfg['data']['processed_dir'],
        size=cfg['image_size'], context=cfg.get('context', 1),
    )
    loader = DataLoader(ds, shuffle=False, batch_size=cfg['batch_size'], num_workers=cfg['num_workers'])
    preds = predict(model, loader, test, device, cfg['data']['width'], cfg['data']['height'])
    # Fixed tau from the CV: select_tau on test would tune on test
    result = evaluate(preds, gt, np.asarray(cfg['taus']), tau=cfg['tau'], **cfg.get('aggregation', {}))
    return preds, result


def volume_table(gt: pd.DataFrame, ious: dict[int, np.ndarray], m0: pd.DataFrame) -> pd.DataFrame:
    """
    One row per test volume: KEY, aclDiagnosis, iou_s<seed> for each seed, iou
    (their mean, the unit of analysis) and iou_m0.

    Per-volume IoUs are averaged, not the predictions: an ensemble would be a
    model the CV never validated.
    """
    table = gt[KEY + ['aclDiagnosis']].reset_index(drop=True)
    cols = [f'iou_s{seed}' for seed in ious]
    for seed, col in zip(ious, cols):
        table[col] = ious[seed]
    table['iou'] = table[cols].mean(axis=1)
    table = table.merge(m0[KEY + ['iou']].rename(columns={'iou': 'iou_m0'}), on=KEY,
                        how='left', validate='one_to_one')
    if table['iou_m0'].isna().any():
        raise ValueError("some test volumes have no M0 IoU")
    return table


def summarize_test(table: pd.DataFrame, seeds: list[int], n_boot: int = 2000) -> pd.DataFrame:
    """Same columns as cv_summary.csv minus fold_std, which has no meaning on test."""
    rows = []
    for c in CLASSES:
        g = table if c == 'all' else table[table['aclDiagnosis'] == c]
        if g.empty:
            continue
        ci = bootstrap_mean(g['iou'].to_numpy(), n=n_boot)
        gain = paired_bootstrap(g['iou'].to_numpy(), g['iou_m0'].to_numpy(), n=n_boot)
        rows.append({
            'class': str(c),
            'n': len(g),
            'iou': ci['mean'], 'ci_lo': ci['lo'], 'ci_hi': ci['hi'],
            'frac_ge_05': float((g['iou'] >= 0.5).mean()),
            'seed_std': _std(pd.Series([g[f'iou_s{s}'].mean() for s in seeds])),
            'delta_m0': gain['delta'], 'm0_lo': gain['lo'], 'm0_hi': gain['hi'],
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Score the refit models on test, once, at the CV's fixed tau.")
    parser.add_argument('--refit-root', type=Path, default=REFIT_ROOT)
    parser.add_argument('--seeds', nargs='+', type=int, default=[42, 0])
    parser.add_argument('--device', help="cpu or mps; default picks MPS when available")
    parser.add_argument('--force', action='store_true', help="Evaluate even if the outputs already exist")
    args = parser.parse_args()

    # Every guard runs before a model is loaded
    check_not_evaluated(REPORTS_DIR, args.force)
    check_preregistered()
    checkpoints = {
        seed: torch.load(args.refit_root / f's{seed}' / 'last.pt', map_location='cpu', weights_only=False)
        for seed in args.seeds
    }
    cfg = common_config({seed: ck['cfg'] for seed, ck in checkpoints.items()})

    dev = get_device(args.device)
    slices, volumes = load_data(cfg)
    test = slices[slices['split'] == 'test'].reset_index(drop=True)
    gt = volume_boxes(volumes[volumes['split'] == 'test'])

    ious, slice_rows = {}, []
    for seed, ck in checkpoints.items():
        print(f"=== s{seed}: scoring {len(gt)} test volumes at tau={cfg['tau']:g}", flush=True)
        preds, result = score_seed(cfg, ck['model'], test, gt, dev)
        preds.to_csv(args.refit_root / f's{seed}' / 'preds_test.csv', index=False)
        ious[seed] = result['ious']
        slice_rows.append({'seed': seed, 'auprc': result['auprc'], 'f1': result['f1']})

    table = volume_table(gt, ious, fixed_box_test(volumes))
    summary = summarize_test(table, args.seeds)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(REPORTS_DIR / 'test_volumes.csv', index=False)
    summary.to_csv(REPORTS_DIR / 'test_summary.csv', index=False)
    pd.DataFrame(slice_rows).to_csv(REPORTS_DIR / 'test_slices.csv', index=False)

    print("\nTest IoU of M1 (seed-averaged per volume), 95% bootstrap CI, gain over M0:")
    print(summary.round(4).to_string(index=False))
    if CV_SUMMARY.exists():
        cv = pd.read_csv(CV_SUMMARY)
        print("\nCV reference (OOF):")
        print(cv[cv['class'].astype(str) == 'all'].round(4).to_string(index=False))
    print(f"\nSaved {REPORTS_DIR}/test_*.csv")


if __name__ == '__main__':
    main()
