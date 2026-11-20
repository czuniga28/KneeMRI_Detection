import argparse
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from matplotlib.patches import Rectangle
from torch import nn
from torch.utils.data import DataLoader

from scripts.cv import CLASSES, DEV_FOLDS, RUNS_DIR, run_name
from src.eval.bootstrap import bootstrap_spearman, paired_bootstrap
from src.eval.gradcam import box_area, energy_in_box, gradcam, pointing_hit
from src.models.net import SliceNet
from src.train.dataset import SliceDataset
from src.train.loop import KEY, PROJECT_ROOT, evaluate, get_device, load_data, volume_boxes

REPORTS_DIR = PROJECT_ROOT / 'reports' / 'fase3'
FIGS_DIR = PROJECT_ROOT / 'docs' / 'entregable3' / 'figs'
# Same tau as the refit and test, so the IoU here is the one reported
REFIT_CFG = PROJECT_ROOT / 'configs' / 'train' / 'm1_refit.yaml'
FOLDS = DEV_FOLDS
SEED = 42
RANDOM_SEED = 0
N_FIG = 4
PRED_COLUMNS = ['pred_x0', 'pred_y0', 'pred_x1', 'pred_y1']
SLICE_COLUMNS = ['examId', 'seriesNo', 'slice_idx', 'split', 'aclDiagnosis']


def positive_slices(slices: pd.DataFrame, fold: str) -> pd.DataFrame:
    """
    Slices of a development fold that contain the ACL.

    Grad-CAM is compared with the ground-truth box, so only positives have one.

    Raises:
        ValueError: If fold is not a development fold (e.g. 'test').
    """
    if fold not in DEV_FOLDS:
        raise ValueError(f"gradcam runs on development folds only, got {fold!r}")
    keep = (slices['split'] == fold) & slices['contains_acl'].astype(bool)
    return slices[keep].reset_index(drop=True)


def load_model(ck: dict, device: torch.device) -> nn.Module:
    # Weights are overwritten below, so skip the ImageNet download
    model = SliceNet(**{**ck['cfg']['model'], 'pretrained': False}).to(device)
    model.load_state_dict(ck['model'])
    return model.eval()


def _dataset(slices: pd.DataFrame, cfg: dict) -> SliceDataset:
    return SliceDataset(
        slices, PROJECT_ROOT / cfg['data']['processed_dir'],
        size=cfg['image_size'], augment=False, context=cfg.get('context', 1),
    )


def score_slices(model: nn.Module, slices: pd.DataFrame, cfg: dict, device: torch.device) -> pd.DataFrame:
    """
    Per positive slice: Grad-CAM energy inside the true box, its area, the
    pointing hit, p and the slice's predicted box (normalized).

    hit is NaN where the map is all zero, like energy.
    """
    model.eval()
    loader = DataLoader(_dataset(slices, cfg), batch_size=cfg['batch_size'], shuffle=False,
                        num_workers=cfg['num_workers'])
    layer = model.backbone.layer4
    rows = []
    for batch in loader:
        images = batch['image'].to(device)
        cams = gradcam(model, images, layer)
        with torch.no_grad():
            out = model(images)
        probs = torch.sigmoid(out['logit']).cpu().numpy()
        pred = out['box'].clamp(0, 1).cpu().numpy()
        for cam, box, p, pb in zip(cams, batch['box'].numpy(), probs, pred):
            energy = energy_in_box(cam, box)
            rows.append({
                'energy': energy, 'area': box_area(box),
                'hit': float(pointing_hit(cam, box)) if np.isfinite(energy) else np.nan,
                'p': float(p), **dict(zip(PRED_COLUMNS, pb.tolist())),
            })
    # shuffle=False: rows follow slices' order
    return pd.concat([slices[SLICE_COLUMNS].reset_index(drop=True), pd.DataFrame(rows)], axis=1)


def volume_ious(run_dir: Path, volumes: pd.DataFrame, tau: float) -> pd.DataFrame:
    """Val IoU per volume of one CV run at the fixed tau: KEY, aclDiagnosis, iou."""
    cfg = yaml.safe_load((Path(run_dir) / 'config.yaml').read_text())
    preds = pd.read_csv(Path(run_dir) / 'preds_val.csv')
    gt = volume_boxes(volumes.merge(preds[KEY].drop_duplicates(), on=KEY))
    res = evaluate(preds, gt, np.asarray(cfg['taus']), tau=tau, **cfg.get('aggregation', {}))
    return gt[KEY + ['aclDiagnosis']].assign(iou=res['ious'])


def aggregate_volumes(scores: pd.DataFrame, ious: pd.DataFrame) -> pd.DataFrame:
    """Mean energy, area and hit per (model, run, volume), joined with the trained model's IoU."""
    vols = scores.groupby(['model', 'run'] + KEY + ['aclDiagnosis'], as_index=False, sort=False).agg(
        n_slices=('energy', 'size'), energy=('energy', 'mean'), area=('area', 'mean'), hit=('hit', 'mean'),
    )
    return vols.merge(ious, on=KEY + ['aclDiagnosis'], how='left', validate='many_to_one')


def summarize(vols: pd.DataFrame) -> pd.DataFrame:
    """
    One row per (model, class).

    H1: energy > area (paired bootstrap over volumes; delta = energy - area).
    H2: Spearman rho between energy and IoU, with a bootstrap interval over volumes.
    """
    columns = ['model', 'class', 'n_volumes', 'energy', 'area', 'ratio', 'hit_rate',
               'delta', 'delta_lo', 'delta_hi', 'spearman_rho', 'spearman_lo', 'spearman_hi']
    rows = []
    for model, gm in vols.groupby('model', sort=False):
        for c in CLASSES:
            g = gm if c == 'all' else gm[gm['aclDiagnosis'] == c]
            g = g.dropna(subset=['energy'])
            row = {'model': model, 'class': str(c), 'n_volumes': len(g)}
            if len(g):
                boot = paired_bootstrap(g['energy'].to_numpy(), g['area'].to_numpy())
                energy, area = g['energy'].mean(), g['area'].mean()
                row.update({
                    'energy': energy, 'area': area, 'ratio': energy / area, 'hit_rate': g['hit'].mean(),
                    'delta': boot['delta'], 'delta_lo': boot['lo'], 'delta_hi': boot['hi'],
                })
                ok = g['iou'].notna()
                if ok.sum() >= 3:
                    rho = bootstrap_spearman(g.loc[ok, 'energy'].to_numpy(), g.loc[ok, 'iou'].to_numpy())
                    row.update({'spearman_rho': rho['rho'], 'spearman_lo': rho['lo'], 'spearman_hi': rho['hi']})
            rows.append(row)
    return pd.DataFrame(rows).reindex(columns=columns)


def paired_control(vols: pd.DataFrame) -> pd.DataFrame:
    """
    E(trained) - E(random) per class, paired over the volumes both models scored
    (the random control's fold), with a paired bootstrap interval.

    The summary's random rows cover one fold and the trained rows five, so only
    this paired comparison says whether training moved the energy into the box.
    """
    columns = ['class', 'n_volumes', 'energy_trained', 'energy_random', 'delta', 'delta_lo', 'delta_hi']
    trained = vols[vols['model'] == 'trained']
    random = vols[vols['model'] == 'random']
    m = trained.merge(random[KEY + ['energy']], on=KEY, suffixes=('_trained', '_random'), validate='one_to_one')
    m = m.dropna(subset=['energy_trained', 'energy_random'])
    rows = []
    for c in CLASSES:
        g = m if c == 'all' else m[m['aclDiagnosis'] == c]
        row = {'class': str(c), 'n_volumes': len(g)}
        if len(g):
            boot = paired_bootstrap(g['energy_trained'].to_numpy(), g['energy_random'].to_numpy())
            row.update({
                'energy_trained': g['energy_trained'].mean(), 'energy_random': g['energy_random'].mean(),
                'delta': boot['delta'], 'delta_lo': boot['lo'], 'delta_hi': boot['hi'],
            })
        rows.append(row)
    return pd.DataFrame(rows).reindex(columns=columns)


def central_slice(slices: pd.DataFrame, volume: pd.Series) -> int:
    """slice_idx of the positive slice whose centre is nearest the true box's z centre (lower on ties)."""
    zc = volume['roiZ'] + volume['roiDepth'] / 2
    s = np.sort(slices['slice_idx'].to_numpy())
    return int(s[np.argmin(np.abs(s + 0.5 - zc))])


def figure(
    group: pd.DataFrame,
    slices: pd.DataFrame,
    volumes: pd.DataFrame,
    scores: pd.DataFrame,
    models: dict[str, tuple[nn.Module, dict]],
    device: torch.device,
    path: Path,
    title: str,
) -> None:
    """One row per volume: centre channel, Grad-CAM overlay, true (green) and predicted (red) boxes."""
    fig, axes = plt.subplots(len(group), 3, figsize=(9, 3 * len(group)), squeeze=False)
    for ax_row, v in zip(axes, group.itertuples()):
        model, cfg = models[v.run]
        key = (slices['examId'] == v.examId) & (slices['seriesNo'] == v.seriesNo)
        pos = slices[key & slices['contains_acl'].astype(bool)]
        vol = volumes[(volumes['examId'] == v.examId) & (volumes['seriesNo'] == v.seriesNo)].iloc[0]
        s = central_slice(pos, vol)
        row = pos[pos['slice_idx'] == s].reset_index(drop=True)

        item = _dataset(row, cfg)[0]
        cam = gradcam(model, item['image'][None].to(device), model.backbone.layer4)[0]
        sc = scores[(scores['run'] == v.run) & (scores['examId'] == v.examId)
                    & (scores['seriesNo'] == v.seriesNo) & (scores['slice_idx'] == s)].iloc[0]

        img = item['image'][1].numpy()
        size = img.shape[-1]
        gt, pred = item['box'].numpy() * size, sc[PRED_COLUMNS].to_numpy(dtype=float) * size
        for ax in ax_row:
            ax.imshow(img, cmap='gray')
            ax.axis('off')
        ax_row[1].imshow(cam, cmap='jet', alpha=0.4)
        for b, color in ((gt, 'lime'), (pred, 'red')):
            ax_row[2].add_patch(Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1],
                                          fill=False, edgecolor=color, linewidth=1.5))
        ax_row[0].set_title(f"{v.examId}-{v.seriesNo} z={s} clase {v.aclDiagnosis}", fontsize=8)
        ax_row[1].set_title(f"energía={sc['energy']:.2f} (área {sc['area']:.2f})", fontsize=8)
        ax_row[2].set_title(f"IoU vol={v.iou:.2f} p={sc['p']:.2f}", fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Grad-CAM energy in the true box for the seed-42 CV runs.")
    parser.add_argument('--runs-dir', type=Path, default=RUNS_DIR)
    parser.add_argument('--device', help="cpu or mps; default picks MPS when available")
    args = parser.parse_args()

    refit_cfg = yaml.safe_load(REFIT_CFG.read_text())
    tau = refit_cfg['tau']
    dev = get_device(args.device)
    slices, volumes = load_data(refit_cfg)

    scores, ious, models = [], [], {}
    for fold in FOLDS:
        run = run_name(fold, SEED)
        run_dir = args.runs_dir / run
        ck = torch.load(run_dir / 'best.pt', map_location='cpu', weights_only=False)
        if ck['cfg']['val_fold'] != fold:
            raise ValueError(f"{run}: checkpoint val_fold is {ck['cfg']['val_fold']!r}, expected {fold!r}")
        print(f"=== {run}: Grad-CAM on positive {fold} slices", flush=True)
        model = load_model(ck, dev)
        models[run] = (model, ck['cfg'])
        scores.append(score_slices(model, positive_slices(slices, fold), ck['cfg'], dev).assign(
            model='trained', run=run))
        ious.append(volume_ious(run_dir, volumes, tau))

    # Control: same architecture untrained. If its maps also beat the area, the
    # effect comes from the input/architecture, not from what the model learned
    fold = FOLDS[0]
    cfg = models[run_name(fold, SEED)][1]
    print(f"=== random control on positive {fold} slices", flush=True)
    torch.manual_seed(RANDOM_SEED)
    random_model = SliceNet(**{**cfg['model'], 'pretrained': False}).to(dev).eval()
    scores.append(score_slices(random_model, positive_slices(slices, fold), cfg, dev).assign(
        model='random', run=f'random_{fold}'))

    scores = pd.concat(scores, ignore_index=True)
    # The random rows get the trained run's IoU on the same volumes: its rho should be ~0
    vols = aggregate_volumes(scores, pd.concat(ious, ignore_index=True))
    summary = summarize(vols)
    control = paired_control(vols)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    scores.to_csv(REPORTS_DIR / 'gradcam_slices.csv', index=False)
    vols.to_csv(REPORTS_DIR / 'gradcam_volumes.csv', index=False)
    summary.to_csv(REPORTS_DIR / 'gradcam_summary.csv', index=False)
    control.to_csv(REPORTS_DIR / 'gradcam_control.csv', index=False)

    trained = vols[vols['model'] == 'trained']
    groups = {
        'best': (trained.nlargest(N_FIG, 'iou'), f"Grad-CAM: {N_FIG} volúmenes con mayor IoU"),
        'worst': (trained.nsmallest(N_FIG, 'iou'), f"Grad-CAM: {N_FIG} volúmenes con menor IoU"),
        'class2': (trained[trained['aclDiagnosis'] == 2].nsmallest(N_FIG, 'iou'),
                   f"Grad-CAM: {N_FIG} peores de la clase 2"),
    }
    for name, (group, title) in groups.items():
        if len(group):
            figure(group, slices, volumes, scores, models, dev, FIGS_DIR / f'gradcam_{name}.png', title)

    print(f"\nGrad-CAM energy vs box area (tau={tau:g}):")
    print(summary.round(4).to_string(index=False))
    print(f"\nTrained - random energy, paired over {FOLDS[0]} volumes:")
    print(control.round(4).to_string(index=False))
    print(f"\nSaved {REPORTS_DIR}/gradcam_*.csv and {FIGS_DIR}/gradcam_*.png")


if __name__ == '__main__':
    main()
