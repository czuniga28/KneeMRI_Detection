"""
Exploratory analysis of the simulated shifts (phase 6, designed after seeing
the monitoring results; the monitoring rule itself does not change).

    python -m scripts.shift_analysis

E1  Does the monitor's silence matter: how much IoU does the model lose?
E2  From which zoom does the monitor alert (dose-response)?
E3  Would other signals have caught what the rule misses? Descriptive only.

Inputs: reports/fase6/shift_*.csv (scripts/simulate_shift.py; shift_none is
required), the test ground truth, monitor_reference.json, the development rows
of reports/fase5/predict_all.csv and manifest.csv.

Outputs: reports/fase6/shift_volumes.csv, reports/fase6/shift_summary.csv,
reports/figures/monitor_dose_response.png. Nothing is written to
monitor_summary.csv: that table stays the preregistered one.
"""
import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.deploy.monitor import K, box_features, check_batch, load_reference, outside, reference_rows
from src.eval.bootstrap import bootstrap_mean, paired_bootstrap
from src.eval.metrics import BOX3D_COLUMNS, iou_1d, iou_2d, iou_3d
from src.eval.shift import gt_after_shift, scale_response
from src.train.loop import KEY, PROJECT_ROOT, volume_boxes

FASE6 = PROJECT_ROOT / 'reports' / 'fase6'
FIGURES = PROJECT_ROOT / 'reports' / 'figures'
INTERIM = PROJECT_ROOT / 'datasets' / 'interim'
PREDICT_ALL = PROJECT_ROOT / 'reports' / 'fase5' / 'predict_all.csv'
LOW_PMAX_PCT = 1
VOLUME_COLUMNS = ['shift', 'examId', 'seriesNo', 'aclDiagnosis', 'f', 'iou', 'iou_base', 'iou_2d', 'iou_z',
                  'b', 'gt_clipped', 'p_max', 'out_box', 'low_pmax', 'out_intensity', 'odd_shape']


def test_ground_truth(interim_dir: Path = INTERIM) -> pd.DataFrame:
    """volume_boxes of the test split plus volumeFilename and the raw height and width."""
    manifest = pd.read_csv(interim_dir / 'manifest.csv')
    splits = pd.read_csv(interim_dir / 'splits.csv')
    vols = splits[KEY + ['split']].merge(manifest, on=KEY, validate='one_to_one')
    vols = vols[vols['split'] == 'test'].reset_index(drop=True)
    gt = volume_boxes(vols)
    gt[['volumeFilename', 'height', 'width']] = vols[['volumeFilename', 'height', 'width']]
    return gt


def dev_signals(predict_all: Path = PREDICT_ALL, interim_dir: Path = INTERIM) -> dict:
    """Development-set references of the E3 signals (p_max is in-sample for the refit)."""
    splits = pd.read_csv(interim_dir / 'splits.csv')
    manifest = pd.read_csv(interim_dir / 'manifest.csv')
    dev_preds = reference_rows(pd.read_csv(predict_all), splits)
    dev_files = set(splits.loc[splits['split'] != 'test', 'volumeFilename'])
    dev_man = manifest[manifest['volumeFilename'].isin(dev_files)]
    q1, q3 = np.percentile(dev_man['intensity_max'], [25, 75])
    return {
        'pmax_floor': float(np.percentile(dev_preds['p_max'], LOW_PMAX_PCT)),
        'intensity_lo': float(q1 - K * (q3 - q1)),
        'intensity_hi': float(q3 + K * (q3 - q1)),
        'shapes': set(zip(dev_man['height'], dev_man['width'])),
    }


def score_shift(name: str, preds: pd.DataFrame, gt: pd.DataFrame, ref: dict, signals: dict) -> pd.DataFrame:
    """One row per volume of one shift; b is left for add_scale_response."""
    names = preds['file'].map(lambda p: Path(str(p)).name)
    if preds['error'].notna().any():
        raise ValueError(f"{name}: {int(preds['error'].notna().sum())} volumes failed; score them explicitly")
    table = gt.merge(preds.assign(volumeFilename=names), on='volumeFilename', how='inner',
                     suffixes=('_gt', ''), validate='one_to_one')
    if len(table) != len(gt):
        raise ValueError(f"{name}: {len(table)} of {len(gt)} test volumes matched")
    f = float(table['f'].iloc[0])
    gt_box, clipped = gt_after_shift(table[[c + '_gt' for c in BOX3D_COLUMNS]].to_numpy(),
                                     table['height_gt'].to_numpy(), table['width_gt'].to_numpy(), f)
    pred = table[BOX3D_COLUMNS].to_numpy(dtype=float)
    feats = box_features(table)
    lo, hi = signals['intensity_lo'], signals['intensity_hi']
    shapes = signals['shapes']
    return pd.DataFrame({
        'shift': name,
        'examId': table['examId'], 'seriesNo': table['seriesNo'], 'aclDiagnosis': table['aclDiagnosis'],
        'f': f,
        'iou': iou_3d(pred, gt_box),
        'iou_2d': iou_2d(pred[:, [0, 1, 3, 4]], gt_box[:, [0, 1, 3, 4]]),
        'iou_z': iou_1d(pred[:, [2, 5]], gt_box[:, [2, 5]]),
        'gt_clipped': clipped,
        'p_max': table['p_max'],
        'out_box': outside(feats, ref).to_numpy(),
        'low_pmax': table['p_max'] < signals['pmax_floor'],
        'out_intensity': (table['intensity_max'] < lo) | (table['intensity_max'] > hi),
        'odd_shape': [(h, w) not in shapes for h, w in zip(table['height'], table['width'])],
        '_box': list(pred), '_width': feats['width'].to_numpy(), '_height': feats['height'].to_numpy(),
    })


def add_baseline(volumes: pd.DataFrame) -> pd.DataFrame:
    """iou_base (IoU of shift_none) and b (scale response against shift_none, zoom only)."""
    base = volumes[volumes['shift'] == 'none'].set_index(KEY)
    if base.empty:
        raise ValueError("shift_none is required: run scripts.simulate_shift --shifts none")
    idx = pd.MultiIndex.from_frame(volumes[KEY])
    volumes['iou_base'] = base['iou'].reindex(idx).to_numpy()
    base_box = np.stack(base['_box'].reindex(idx).to_numpy())
    volumes['b'] = np.nan
    for f, rows in volumes.groupby('f').groups.items():
        if f != 1:
            pos = volumes.index.get_indexer(rows)
            volumes.loc[rows, 'b'] = scale_response(np.stack(volumes.loc[rows, '_box']), base_box[pos], f)
    return volumes


def summarize(volumes: pd.DataFrame, ref: dict) -> pd.DataFrame:
    rows = []
    for name, v in volumes.groupby('shift', sort=False):
        feats = pd.DataFrame({'width': v['_width'], 'height': v['_height']})
        res = check_batch(feats, ref)
        res = {k: res[k] for k in ('n', 'n_out', 'frac_out', 'alert')}
        iou = bootstrap_mean(v['iou'].to_numpy())
        delta = paired_bootstrap(v['iou'].to_numpy(), v['iou_base'].to_numpy())
        row = {
            'shift': name, 'f': v['f'].iloc[0], **res,
            'iou': iou['mean'], 'iou_lo': iou['lo'], 'iou_hi': iou['hi'],
            'delta_iou': delta['delta'], 'delta_lo': delta['lo'], 'delta_hi': delta['hi'],
            'frac_ge_05': float((v['iou'] >= 0.5).mean()),
            'iou_2d': v['iou_2d'].mean(), 'iou_z': v['iou_z'].mean(),
            'n_gt_clipped': int(v['gt_clipped'].sum()),
            'low_pmax': v['low_pmax'].mean(), 'out_intensity': v['out_intensity'].mean(),
            'odd_shape': v['odd_shape'].mean(),
        }
        if v['b'].notna().all():
            b = bootstrap_mean(v['b'].to_numpy())
            row.update(b_median=float(v['b'].median()), b_mean=b['mean'], b_lo=b['lo'], b_hi=b['hi'])
        rows.append(row)
    return pd.DataFrame(rows)


def dose_response_figure(summary: pd.DataFrame, out: Path) -> None:
    z = summary[summary['shift'].str.fullmatch(r'none|zoom\d{3}')].sort_values('f')
    base = float(z.loc[z['shift'] == 'none', 'iou'].iloc[0])
    ink, muted, series = '#0b0b0b', '#52514e', '#2a78d6'
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.6), constrained_layout=True)
    for ax in (a1, a2):
        ax.set_xscale('log')
        ax.set_xticks(z['f'], [f'{f:g}' for f in z['f']])
        ax.minorticks_off()
        ax.set_xlabel('factor de zoom f (escala log)', color=muted)
        ax.grid(axis='y', color='#e4e3df', linewidth=0.8)
        ax.spines[['top', 'right']].set_visible(False)
        ax.tick_params(colors=muted)
    a1.plot(z['f'], 100 * z['frac_out'], color=series, linewidth=2, marker='o', markersize=6)
    a1.axhline(10, color=muted, linewidth=1, linestyle='--')
    a1.text(z['f'].max(), 10.5, 'umbral 10 %', color=muted, fontsize=8, va='bottom', ha='right')
    a1.set_ylabel('% de cajas fuera de las cercas', color=ink)
    a1.set_title('Monitor (regla de tamaño de caja)', color=ink, fontsize=10, loc='left')
    a2.errorbar(z['f'], z['iou'], yerr=[z['iou'] - z['iou_lo'], z['iou_hi'] - z['iou']],
                color=series, linewidth=2, marker='o', markersize=6, capsize=3)
    a2.axhline(base, color=muted, linewidth=1, linestyle='--')
    a2.text(z['f'].min(), base, ' sin cambio', color=muted, fontsize=8, va='bottom')
    a2.set_ylabel('IoU 3D medio (IC 95 %)', color=ink)
    a2.set_title('Modelo (IoU contra el GT transformado)', color=ink, fontsize=10, loc='left')
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    plt.close(fig)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Exploratory analysis of the simulated shifts.")
    parser.add_argument('--shift-dir', type=Path, default=FASE6)
    parser.add_argument('--reference', type=Path, default=FASE6 / 'monitor_reference.json')
    parser.add_argument('--out-dir', type=Path, default=FASE6)
    parser.add_argument('--figure', type=Path, default=FIGURES / 'monitor_dose_response.png')
    args = parser.parse_args(argv)

    ref = load_reference(args.reference)
    gt = test_ground_truth()
    signals = dev_signals()
    # Only simulate_shift outputs: this script's own shift_summary/shift_volumes share the prefix
    files = sorted(p for p in args.shift_dir.glob('shift_*.csv')
                   if re.fullmatch(r'shift_(none|(zoom|gain|gamma|noise)\d{3})', p.stem))
    files = sorted(files, key=lambda p: p.stem != 'shift_none')  # none first
    volumes = pd.concat([
        score_shift(p.stem.removeprefix('shift_'), pd.read_csv(p), gt, ref, signals) for p in files
    ], ignore_index=True)
    volumes = add_baseline(volumes)
    summary = summarize(volumes, ref)
    volumes = volumes[VOLUME_COLUMNS]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    volumes.to_csv(args.out_dir / 'shift_volumes.csv', index=False)
    summary.to_csv(args.out_dir / 'shift_summary.csv', index=False)
    dose_response_figure(summary, args.figure)

    cols = ['shift', 'f', 'frac_out', 'alert', 'iou', 'delta_iou', 'delta_lo', 'delta_hi', 'frac_ge_05',
            'iou_2d', 'iou_z', 'b_median', 'n_gt_clipped', 'low_pmax', 'out_intensity', 'odd_shape']
    with pd.option_context('display.width', 200, 'display.max_columns', None):
        print(summary.reindex(columns=cols).round(3).to_string(index=False))
    print(f"-> {args.out_dir / 'shift_volumes.csv'}, {args.out_dir / 'shift_summary.csv'}, {args.figure}")


if __name__ == '__main__':
    main()
