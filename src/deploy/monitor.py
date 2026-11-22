"""
Monitoring rule for the deployed model (phase 6).

The definition document asks for an alert when >10 % of predictions fall
outside the historical interquartile range. Taken literally that alarm always
fires: the IQR holds the central 50 % by construction, so ~50 % of boxes fall
outside it with no shift at all. The rule here keeps the 10 % threshold and the
box-size features but uses Tukey's fences, [Q1 - 1.5 IQR, Q3 + 1.5 IQR]. This
is a declared deviation from the original plan.

Only geometry is watched (box width and height in raw pixels): a change in
intensity that leaves the boxes alone is invisible to this rule.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

FEATURES = ('width', 'height')  # pixels of the raw image
K = 1.5
THRESHOLD = 0.10
MIN_N = 20
MIN_REF = 30


def box_features(preds: pd.DataFrame) -> pd.DataFrame:
    """
    Box sizes of the rows of a predict.py CSV that produced a box.

    Rows with a non-empty error or a NaN coordinate are dropped and counted in
    attrs['n_dropped']. Rows with needs_review=True are kept: they are still
    predictions the service returned. z_extent is descriptive only.
    """
    coords = ['x0', 'y0', 'z0', 'x1', 'y1', 'z1']
    err = preds['error'] if 'error' in preds else pd.Series(np.nan, index=preds.index)
    has_error = err.notna() & (err.astype(str).str.strip() != '')
    ok = ~has_error & preds[coords].notna().all(axis=1)
    kept = preds[ok]
    feats = pd.DataFrame({
        'file': kept['file'],
        'width': kept['x1'] - kept['x0'],
        'height': kept['y1'] - kept['y0'],
        'z_extent': kept['z1'] - kept['z0'] + 1,
    }, index=kept.index)
    feats.attrs['n_dropped'] = int((~ok).sum())
    return feats


def fit_reference(feats: pd.DataFrame, k: float = K) -> dict:
    """
    Tukey fences of each feature in FEATURES.

    Raises:
        ValueError: If there are fewer than MIN_REF rows or a feature has IQR 0
            (its fences would collapse onto one value).

    Returns:
        {'k', 'n', 'features': {f: {q1, q3, lo, hi}}, 'base_rate'}, where
        base_rate is the fraction of the reference itself outside the fences.
    """
    n = len(feats)
    if n < MIN_REF:
        raise ValueError(f"reference has {n} rows < {MIN_REF}")
    features = {}
    for f in FEATURES:
        q1, q3 = np.percentile(feats[f].to_numpy(dtype=float), [25, 75])
        iqr = q3 - q1
        if iqr == 0:
            raise ValueError(f"feature {f!r} has IQR 0")
        features[f] = {'q1': float(q1), 'q3': float(q3), 'lo': float(q1 - k * iqr), 'hi': float(q3 + k * iqr)}
    ref = {'k': float(k), 'n': int(n), 'features': features}
    ref['base_rate'] = float(outside(feats, ref).mean())
    return ref


def outside_by_feature(feats: pd.DataFrame, ref: dict) -> pd.DataFrame:
    """One boolean column per feature: strictly below lo or above hi."""
    return pd.DataFrame({
        f: (feats[f] < b['lo']) | (feats[f] > b['hi']) for f, b in ref['features'].items()
    }, index=feats.index)


def outside(feats: pd.DataFrame, ref: dict) -> pd.Series:
    """True where any feature is outside its fences; a value on a fence is inside."""
    return outside_by_feature(feats, ref).any(axis=1)


def check_batch(feats: pd.DataFrame, ref: dict, threshold: float = THRESHOLD, min_n: int = MIN_N) -> dict:
    """
    Apply the rule to one batch.

    Returns:
        {n, n_out, frac_out, status, alert}. Below min_n rows the batch is too
        small to decide: status='insufficient', alert=None. Otherwise alert is
        frac_out > threshold (strict, the rule says ">10 %").
    """
    n = len(feats)
    n_out = int(outside(feats, ref).sum())
    frac = n_out / n if n else float('nan')
    if n < min_n:
        return {'n': n, 'n_out': n_out, 'frac_out': frac, 'status': 'insufficient', 'alert': None}
    alert = bool(frac > threshold)
    return {'n': n, 'n_out': n_out, 'frac_out': frac, 'status': 'alert' if alert else 'ok', 'alert': alert}


def false_alarm_prob(p0: float, n: int, threshold: float = THRESHOLD) -> float:
    """P(alert | no shift): P(Binomial(n, p0) > floor(threshold * n))."""
    return float(stats.binom.sf(np.floor(threshold * n), n, p0))


def save_reference(ref: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(ref, indent=2))


def load_reference(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


def reference_rows(preds: pd.DataFrame, splits: pd.DataFrame) -> pd.DataFrame:
    """
    The rows of preds whose volume is not in the test split.

    This is the guard against leakage: the reference never sees a test volume.

    Raises:
        ValueError: If a file of preds is missing from splits (its split, and so
            whether it may enter the reference, would be unknown).
    """
    split = _split_of(preds, splits)
    return preds[(split != 'test').to_numpy()]


def _split_of(preds: pd.DataFrame, splits: pd.DataFrame) -> pd.Series:
    names = preds['file'].map(lambda p: Path(str(p)).name)
    lookup = splits.set_index('volumeFilename')['split']
    missing = sorted(set(names) - set(lookup.index))
    if missing:
        raise ValueError(f"{len(missing)} files not in splits, e.g. {missing[:3]}")
    return names.map(lookup)


def split_rows(preds: pd.DataFrame, splits: pd.DataFrame, split: str) -> pd.DataFrame:
    """The rows of preds in one split (fails like reference_rows on unknown files)."""
    return preds[(_split_of(preds, splits) == split).to_numpy()]
