"""
Simulated domain shift for the monitoring rule (phase 6).

Transforms each raw test volume, runs predict_volume on it and writes a CSV
with the schema of scripts/predict.py, one per shift:

    python -m scripts.simulate_shift [--shifts none zoom080 gamma050 ...] [--device cpu]

-> reports/fase6/shift_<name>.csv, ready for `scripts.monitor check`.

Shift names are parametric, the number is the parameter x100:
    none        identity
    zoom<f>     zoom H and W by f, then crop the centre to 320 (zoom125: f = 1.25)
    gain<g>     intensities x g (absorbed by the percentile normalization)
    gamma<g>    MAX * (v / MAX) ** g, non-linear: the normalization keeps it
    noise<s>    Gaussian noise, sigma = s * p99.5 of the volume, seed 0

Besides the predict.py schema the CSV has f (zoom factor, 1 without zoom) and
intensity_max (maximum of the transformed volume).

Boxes and the depth/height/width columns are in the coordinates of the
transformed volume: the monitor sees what the service would have received.
"""
import argparse
import pickle
import re
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


def gamma(raw: np.ndarray, g: float) -> np.ndarray:
    """MAX * (v / MAX) ** g, rounded back to the input dtype."""
    v = np.clip(raw.astype(np.float64), 0, MAX_INTENSITY) / MAX_INTENSITY
    return np.rint(MAX_INTENSITY * v ** g).astype(raw.dtype)


def noise(raw: np.ndarray, s: float, seed: int = 0) -> np.ndarray:
    """Add N(0, (s * p99.5)^2) noise, clip to [0, MAX_INTENSITY], round to the input dtype."""
    sigma = s * np.percentile(raw, 99.5)
    noisy = raw.astype(np.float64) + np.random.default_rng(seed).normal(0, sigma, raw.shape)
    return np.clip(np.rint(noisy), 0, MAX_INTENSITY).astype(raw.dtype)


KINDS = {
    'zoom': lambda f: lambda v: center_crop_hw(zoom_hw(v, f)),
    'gain': lambda g: lambda v: gain(v, g),
    'gamma': lambda g: lambda v: gamma(v, g),
    'noise': lambda s: lambda v: noise(v, s),
}
DEFAULT_SHIFTS = ['none', 'zoom060', 'zoom070', 'zoom080', 'zoom090', 'zoom110', 'zoom125', 'zoom140',
                  'gain070', 'gamma050', 'gamma200', 'noise005', 'noise010']


def parse_shift(name: str):
    """(transform, f) of a shift name; f is the zoom factor, 1 for every other kind."""
    if name == 'none':
        return (lambda v: v), 1.0
    m = re.fullmatch(r'(zoom|gain|gamma|noise)(\d{3})', name)
    if not m:
        raise ValueError(f"unknown shift {name!r}")
    kind, param = m.group(1), int(m.group(2)) / 100
    return KINDS[kind](param), (param if kind == 'zoom' else 1.0)


def test_files(splits_csv: Path = SPLITS, volumes_dir: Path = VOLUMES) -> list[Path]:
    splits = pd.read_csv(splits_csv)
    return [volumes_dir / f for f in splits.loc[splits['split'] == 'test', 'volumeFilename']]


def run_shift(files, transform, model, cfg, prep, dev, f: float = 1.0) -> pd.DataFrame:
    rows = []
    for path in files:
        row = {'file': str(path), 'f': f}
        try:
            raw = transform(np.asarray(load_volume(path)))
            row.update(zip(('depth', 'height', 'width'), raw.shape))
            row['intensity_max'] = int(raw.max())
            row.update(predict_volume(raw, model, cfg, prep, dev))
        except (ValueError, OSError, pickle.UnpicklingError, EOFError) as e:
            row['error'] = f"{type(e).__name__}: {e}"
            print(f"ERROR {path}: {row['error']}", file=sys.stderr)
        rows.append(row)
    return pd.DataFrame(rows).reindex(columns=COLUMNS + ['f', 'intensity_max'])


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Predict on transformed test volumes.")
    parser.add_argument('--shifts', nargs='+', default=DEFAULT_SHIFTS)
    parser.add_argument('--checkpoint', type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument('--splits', type=Path, default=SPLITS)
    parser.add_argument('--volumes', type=Path, default=VOLUMES)
    parser.add_argument('--out-dir', type=Path, default=OUT_DIR)
    parser.add_argument('--device')
    args = parser.parse_args(argv)

    prep = yaml.safe_load(PREP_CFG.read_text())['preprocess']
    dev = get_device(args.device)
    shifts = {name: parse_shift(name) for name in args.shifts}  # fail on a bad name before loading
    model, cfg = load_model(args.checkpoint, dev)
    files = test_files(args.splits, args.volumes)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for name, (transform, f) in shifts.items():
        table = run_shift(files, transform, model, cfg, prep, dev, f)
        out = args.out_dir / f'shift_{name}.csv'
        table.to_csv(out, index=False)
        print(f"{name}: {len(table)} files, {int(table['error'].notna().sum())} errors -> {out}")


if __name__ == '__main__':
    main()
