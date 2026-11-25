"""
Geometry of the simulated domain shifts (exploratory analysis of phase 6).

Boxes are end-exclusive in BOX3D_COLUMNS order (x0, y0, z0, x1, y1, z1). The
zoom shifts of scripts/simulate_shift.py rescale H and W with ndimage.zoom
(grid_mode=False) and then crop the centre to 320; these functions carry the
ground truth through the same steps so a shifted prediction can be scored.
"""
import numpy as np


def zoomed_shape(n, f):
    """Output length of ndimage.zoom on an axis of length n: round(n * f)."""
    return np.round(np.asarray(n) * f).astype(int)


def map_coord(c, n_in, n_out):
    """
    A pixel edge through ndimage.zoom with grid_mode=False.

    That mode sends the centre of pixel i to i * s, s = (n_out - 1) / (n_in - 1).
    An edge c sits half a pixel off a centre, so c' = (c - 0.5) * s + 0.5.
    """
    s = (np.asarray(n_out, dtype=float) - 1) / (np.asarray(n_in, dtype=float) - 1)
    return (np.asarray(c, dtype=float) - 0.5) * s + 0.5


def gt_after_shift(gt, h, w, f, max_side=320):
    """
    Ground-truth boxes after zoom by f in H and W and a central crop to max_side.

    Args:
        gt: Array (n, 6) in BOX3D_COLUMNS order, raw-volume pixels.
        h, w: Raw in-plane size of each volume, scalars or arrays (n,).
        f: Zoom factor. f == 1 returns gt unchanged.

    Returns:
        (boxes, clipped): boxes (n, 6) in the coordinates of the transformed
        volume, clipped to its extent; clipped (n,) is True where the crop cut
        away part of the box's in-plane area. z never changes.
    """
    gt = np.asarray(gt, dtype=float)
    n = len(gt)
    h = np.broadcast_to(np.asarray(h), (n,))
    w = np.broadcast_to(np.asarray(w), (n,))
    if f == 1:
        return gt.copy(), np.zeros(n, dtype=bool)
    hz, wz = zoomed_shape(h, f), zoomed_shape(w, f)
    out = gt.copy()
    out[:, [0, 3]] = map_coord(gt[:, [0, 3]], w[:, None], wz[:, None])
    out[:, [1, 4]] = map_coord(gt[:, [1, 4]], h[:, None], hz[:, None])
    # Same offsets as center_crop_hw
    out[:, [0, 3]] -= np.maximum(0, (wz - max_side) // 2)[:, None]
    out[:, [1, 4]] -= np.maximum(0, (hz - max_side) // 2)[:, None]

    before = _area(out)
    out[:, [0, 3]] = np.clip(out[:, [0, 3]], 0, np.minimum(wz, max_side)[:, None])
    out[:, [1, 4]] = np.clip(out[:, [1, 4]], 0, np.minimum(hz, max_side)[:, None])
    return out, _area(out) < before - 1e-9


def scale_response(pred_shift, pred_base, f):
    """
    Per-volume exponent b with sqrt(A_shift / A_base) = f ** b, A the in-plane area.

    b = 1: the box scales with the image, as it should; b = 0: it ignores scale.

    Raises:
        ValueError: If f == 1 (b is undefined).
    """
    if f == 1:
        raise ValueError("scale_response is undefined at f == 1")
    ratio = np.sqrt(_area(np.asarray(pred_shift, float)) / _area(np.asarray(pred_base, float)))
    return np.log(ratio) / np.log(f)


def _area(boxes):
    return np.clip(boxes[:, 3] - boxes[:, 0], 0, None) * np.clip(boxes[:, 4] - boxes[:, 1], 0, None)
