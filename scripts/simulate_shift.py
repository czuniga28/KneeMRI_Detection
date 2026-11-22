"""
Simulated domain shift for the monitoring rule (phase 6).

Transforms each raw test volume, runs predict_volume on it and writes a CSV
with the schema of scripts/predict.py, one per shift:

    python -m scripts.simulate_shift [--shifts zoom080 zoom125 gain070] [--device cpu]

-> reports/fase6/shift_<name>.csv, ready for `scripts.monitor check`.

Boxes and the depth/height/width columns are in the coordinates of the
transformed volume: the monitor sees what the service would have received.
"""
import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy import ndimage

from scripts.predict import COLUMNS, DEFAULT_CHECKPOINT, PREP_CFG
from src.data.io import load_volume
from src.deploy.predict import MAX_INTENSITY, MAX_SIDE, load_model, predict_volume
from src.train.loop import PROJECT_ROOT, get_device

OUT_DIR = PROJECT_ROOT / 'reports' / 'fase6'
SPLITS = PROJECT_ROOT / 'datasets' / 'interim' / 'splits.csv'
VOLUMES = PROJECT_ROOT / 'datasets' / 'raw' / 'volumetric_data'


def zoom_hw(raw: np.ndarray, factor: float) -> np.ndarray:
    """Rescale H and W by factor (linear), keeping D and the dtype."""
    return ndimage.zoom(raw, (1, factor, factor), order=1)


def center_crop_hw(raw: np.ndarray, max_side: int = MAX_SIDE) -> np.ndarray:
    """Central crop of H and W to at most max_side (check_input rejects larger sides)."""
    _, h, w = raw.shape
    top, left = max(0, (h - max_side) // 2), max(0, (w - max_side) // 2)
    return raw[:, top:top + max_side, left:left + max_side]


def gain(raw: np.ndarray, factor: float) -> np.ndarray:
    """Multiply intensities, clip to [0, MAX_INTENSITY] and round back to the input dtype."""
    return np.clip(np.rint(raw.astype(np.float64) * factor), 0, MAX_INTENSITY).astype(raw.dtype)


SHIFTS = {
    'zoom080': lambda v: zoom_hw(v, 0.8),
    'zoom125': lambda v: center_crop_hw(zoom_hw(v, 1.25)),
    'gain070': lambda v: gain(v, 0.7),
}


def test_files(splits_csv: Path = SPLITS, volumes_dir: Path = VOLUMES) -> list[Path]:
    splits = pd.read_csv(splits_csv)
    return [volumes_dir / f for f in splits.loc[splits['split'] == 'test', 'volumeFilename']]


def run_shift(files, transform, model, cfg, prep, dev) -> pd.DataFrame:
    rows = []
    for path in files:
        row = {'file': str(path)}
        try:
            raw = transform(np.asarray(load_volume(path)))
            row.update(zip(('depth', 'height', 'width'), raw.shape))
            row.update(predict_volume(raw, model, cfg, prep, dev))
        except (ValueError, OSError, pickle.UnpicklingError, EOFError) as e:
            row['error'] = f"{type(e).__name__}: {e}"
            print(f"ERROR {path}: {row['error']}", file=sys.stderr)
        rows.append(row)
    return pd.DataFrame(rows).reindex(columns=COLUMNS)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Predict on transformed test volumes.")
    parser.add_argument('--shifts', nargs='+', choices=list(SHIFTS), default=list(SHIFTS))
    parser.add_argument('--checkpoint', type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument('--splits', type=Path, default=SPLITS)
    parser.add_argument('--volumes', type=Path, default=VOLUMES)
    parser.add_argument('--out-dir', type=Path, default=OUT_DIR)
    parser.add_argument('--device')
    args = parser.parse_args(argv)

    prep = yaml.safe_load(PREP_CFG.read_text())['preprocess']
    dev = get_device(args.device)
    model, cfg = load_model(args.checkpoint, dev)
    files = test_files(args.splits, args.volumes)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for name in args.shifts:
        table = run_shift(files, SHIFTS[name], model, cfg, prep, dev)
        out = args.out_dir / f'shift_{name}.csv'
        table.to_csv(out, index=False)
        print(f"{name}: {len(table)} files, {int(table['error'].notna().sum())} errors -> {out}")


if __name__ == '__main__':
    main()
