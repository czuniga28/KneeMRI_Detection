"""
Inference on raw volumes for deployment (phase 5).

Every preprocessing step calls the function training used (preprocess_volume,
slice_image, aggregate_volume, denormalize_boxes); nothing is reimplemented
here, so training and serving cannot drift apart.

One model is deployed, the refit with seed 42 (runs/refit/s42/last.pt). The test
score averaged the IoU of two seeds, but an ensemble of models was never
validated, so serving one would be an untested configuration. Seed 42 is the
configuration's seed, and the spread between seeds on test was 0.0097 IoU.
"""
import warnings
from pathlib import Path

import numpy as np
import torch
from torch import nn

from src.data.transforms import preprocess_volume
from src.eval.aggregate import aggregate_volume
from src.models.net import SliceNet
from src.train.dataset import denormalize_boxes, slice_image

MAX_SIDE = 320
MIN_DEPTH = 3
MAX_INTENSITY = 4095


def check_input(raw: np.ndarray, max_height: int = MAX_SIDE, max_width: int = MAX_SIDE) -> None:
    """
    First barrier against out-of-distribution input.

    Raises:
        ValueError: If raw is not (D, h, w), if h or w exceeds the padded size,
            or if D < 3 (fewer slices than one 2.5D stack).

    Warns:
        UserWarning: If the dtype is not uint16 or values exceed 4095, the two
            properties the data QA gate checked on the training volumes.
    """
    raw = np.asarray(raw)
    if raw.ndim != 3:
        raise ValueError(f"expected a (D, h, w) volume, got shape {raw.shape}")
    d, h, w = raw.shape
    if h > max_height or w > max_width:
        raise ValueError(f"in-plane shape ({h}, {w}) exceeds ({max_height}, {max_width})")
    if d < MIN_DEPTH:
        raise ValueError(f"depth {d} < {MIN_DEPTH}")
    if raw.dtype != np.uint16:
        warnings.warn(f"dtype {raw.dtype}, training volumes were uint16", stacklevel=2)
    if raw.size and raw.max() > MAX_INTENSITY:
        warnings.warn(f"max intensity {raw.max()} > {MAX_INTENSITY}, outside the training range", stacklevel=2)


def load_model(checkpoint: str | Path, device: torch.device) -> tuple[nn.Module, dict]:
    """
    The model and its cfg from a checkpoint, in eval mode (same steps as predict_checkpoint).

    Raises:
        ValueError: If the cfg has no fixed tau, e.g. a CV run's best.pt (it has
            only the candidate taus). Checked here so the batch fails before the
            first file, not with a KeyError inside locate.
    """
    ck = torch.load(checkpoint, map_location='cpu', weights_only=False)
    cfg = ck['cfg']
    if 'tau' not in cfg:
        raise ValueError(f"{checkpoint} has no fixed tau: use a refit checkpoint (runs/refit/s*/last.pt)")
    # Weights are overwritten below, so skip the ImageNet download
    model = SliceNet(**{**cfg['model'], 'pretrained': False}).to(device)
    model.load_state_dict(ck['model'])
    return model.eval(), cfg


@torch.no_grad()
def predict_slices(
    model: nn.Module, vol: np.ndarray, cfg: dict, device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Per-slice predictions on a preprocessed volume.

    Args:
        vol: Output of preprocess_volume, shape (D, H, W).

    Returns:
        p of shape (D,) and boxes of shape (D, 4) in padded-slice pixels,
        clamped and denormalized as in loop.predict.
    """
    model.eval()
    images = torch.stack([
        slice_image(vol, s, cfg['image_size'], cfg.get('context', 1)) for s in range(len(vol))
    ])
    probs, boxes = [], []
    for batch in torch.split(images, cfg['batch_size']):
        out = model(batch.to(device))
        probs.append(torch.sigmoid(out['logit']).cpu().numpy())
        boxes.append(out['box'].clamp(0, 1).cpu().numpy())
    boxes = denormalize_boxes(np.concatenate(boxes), cfg['data']['width'], cfg['data']['height'])
    return np.concatenate(probs).astype(float), boxes


def locate(p: np.ndarray, boxes: np.ndarray, cfg: dict, shape: tuple[int, int, int]) -> dict:
    """
    3D box of the ACL in raw-volume coordinates, with its confidence.

    Padding goes at the end of each axis, so the box needs no shift, but it can
    reach into the padded area: x is clipped to the raw w and y to the raw h.

    Returns:
        x0, y0, z0, x1, y1, z1 (end-exclusive), p_max, p_run (mean p over
        [z0, z1)) and needs_review. best_run always returns a range, falling
        back to the single most likely slice when no p exceeds tau: the model
        then makes up a box without having seen the ACL. needs_review =
        p_max < tau marks that box so it is not used without a human check.
    """
    _, h, w = shape
    p = np.asarray(p, dtype=float)
    tau = cfg['tau']
    x0, y0, z0, x1, y1, z1 = aggregate_volume(p, boxes, tau, **cfg.get('aggregation', {}))
    x0, x1 = np.clip([x0, x1], 0, w)
    y0, y1 = np.clip([y0, y1], 0, h)
    z0, z1 = int(z0), int(z1)
    return {
        'x0': float(x0), 'y0': float(y0), 'z0': z0, 'x1': float(x1), 'y1': float(y1), 'z1': z1,
        'p_max': float(p.max()), 'p_run': float(p[z0:z1].mean()), 'needs_review': bool(p.max() < tau),
    }


def predict_volume(raw: np.ndarray, model: nn.Module, cfg: dict, prep: dict, device: torch.device) -> dict:
    """check_input, preprocess_volume, predict_slices and locate on one raw volume."""
    raw = np.asarray(raw)
    check_input(raw, prep['pad_height'], prep['pad_width'])
    vol = preprocess_volume(raw, prep)
    p, boxes = predict_slices(model, vol, cfg, device)
    return locate(p, boxes, cfg, raw.shape)
