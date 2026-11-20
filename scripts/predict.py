"""
Locate the ACL in raw .pck volumes with the deployed model.

    python -m scripts.predict datasets/raw/volumetric_data/1001866-5.pck [...] \\
        --checkpoint runs/refit/s42/last.pt --out predicciones.csv [--device cpu]

Zenodo stores the .pck files flat in datasets/raw/volumetric_data/.

Preprocessing comes from configs/prep.yaml; everything else (tau, aggregation,
image size) from the checkpoint's cfg. The default checkpoint is the seed-42
refit; see src/deploy/predict.py for why one model and not an ensemble.
"""
import argparse
import pickle
import sys
import warnings
from pathlib import Path

import pandas as pd
import yaml

from src.data.io import load_volume
from src.deploy.predict import load_model, predict_volume
from src.train.loop import PROJECT_ROOT, get_device

PREP_CFG = PROJECT_ROOT / 'configs' / 'prep.yaml'
DEFAULT_CHECKPOINT = PROJECT_ROOT / 'runs' / 'refit' / 's42' / 'last.pt'
COLUMNS = ['file', 'depth', 'height', 'width', 'x0', 'y0', 'z0', 'x1', 'y1', 'z1',
           'p_max', 'p_run', 'needs_review', 'error']
# Errors of one bad file; anything else is a bug and should stop the batch
FILE_ERRORS = (ValueError, OSError, pickle.UnpicklingError, EOFError)


def main() -> None:
    parser = argparse.ArgumentParser(description="3D ACL box per raw .pck volume.")
    parser.add_argument('files', nargs='+', type=Path)
    parser.add_argument('--checkpoint', type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device', help="cpu or mps; default picks MPS when available")
    args = parser.parse_args()

    prep = yaml.safe_load(PREP_CFG.read_text())['preprocess']
    dev = get_device(args.device)
    model, cfg = load_model(args.checkpoint, dev)

    rows = []
    for path in args.files:
        row = {'file': str(path)}
        try:
            raw = load_volume(path)
            shape = getattr(raw, 'shape', ())
            if len(shape) == 3:
                row.update(zip(('depth', 'height', 'width'), shape))
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                row.update(predict_volume(raw, model, cfg, prep, dev))
            for w in caught:
                print(f"WARNING {path}: {w.message}", file=sys.stderr)
        except FILE_ERRORS as e:
            row['error'] = f"{type(e).__name__}: {e}"
            print(f"ERROR {path}: {row['error']}", file=sys.stderr)
        rows.append(row)

    table = pd.DataFrame(rows).reindex(columns=COLUMNS)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out, index=False)

    n_err = int(table['error'].notna().sum())
    n_review = int((table['needs_review'] == True).sum())  # noqa: E712 (NaN on error rows)
    print(f"{len(table)} files: {len(table) - n_err} located, {n_review} need review, {n_err} errors -> {args.out}")


if __name__ == '__main__':
    main()
