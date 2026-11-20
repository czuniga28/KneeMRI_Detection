import numpy as np

def normalize_volume(volume: np.ndarray, clip_low_pct: float, clip_high_pct: float) -> np.ndarray:
    """
    Clip a volume to its own intensity percentiles, then z-score it.

    Args:
        volume: Raw volume of any numeric dtype.
        clip_low_pct: Lower percentile (0-100) to clip to.
        clip_high_pct: Upper percentile (0-100) to clip to.

    Returns:
        A float32 array with the same shape as `volume`.
    """
    # Cast first: subtracting the mean from uint16 would wrap negatives around
    vol = volume.astype(np.float32)

    low, high = np.percentile(vol, [clip_low_pct, clip_high_pct])
    vol = np.clip(vol, low, high)

    mean = vol.mean()
    std = vol.std()
    # A constant volume has no contrast to scale; centre it and stop
    if std == 0:
        return np.zeros_like(vol)

    return ((vol - mean) / std).astype(np.float32)


def pad_volume(volume: np.ndarray, target_height: int, target_width: int) -> np.ndarray:
    """
    Pad a (D, h, w) volume to (D, target_height, target_width).

    Padding goes at the end of each in-plane axis, so pixel (0, 0) stays put and ROI coordinates remain valid without any shift.

    Args:
        volume: Normalized volume of shape (D, h, w).
        target_height: Height to pad to.
        target_width: Width to pad to.

    Returns:
        The padded volume, same dtype as `volume`.
    """
    _, h, w = volume.shape
    if h > target_height or w > target_width:
        raise ValueError(
            f"Volume in-plane shape ({h}, {w}) exceeds target ({target_height}, {target_width})"
        )

    fill_value = volume.min()
    pad_width = ((0, 0), (0, target_height - h), (0, target_width - w))
    return np.pad(volume, pad_width, mode="constant", constant_values=fill_value)


def preprocess_volume(raw: np.ndarray, prep: dict) -> np.ndarray:
    """
    Raw (D, h, w) volume to the stored training input: normalize_volume, then
    pad_volume, then cast to storage_dtype.

    The single definition shared by scripts/build_dataset.py and deployment
    (src/deploy/predict.py), so the two cannot drift apart. The cast is part of
    the contract: the model was trained on values rounded to storage_dtype.

    Args:
        raw: Raw volume of shape (D, h, w).
        prep: The 'preprocess' section of configs/prep.yaml.

    Returns:
        Array of shape (D, pad_height, pad_width) and dtype storage_dtype.
    """
    vol = normalize_volume(raw, prep['clip_low_pct'], prep['clip_high_pct'])
    vol = pad_volume(vol, prep['pad_height'], prep['pad_width'])
    return vol.astype(prep['storage_dtype'])
