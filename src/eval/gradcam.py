import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn


def gradcam(model: nn.Module, images: Tensor, layer: nn.Module) -> np.ndarray:
    """
    Grad-CAM of the classification logit ('does this slice contain the ACL').

    Args:
        model: SliceNet in eval mode.
        images: Batch of shape (B, 3, H, W), on the model's device.
        layer: Module whose output is the activation A, e.g. model.backbone.layer4.

    Returns:
        Array of shape (B, H, W), >= 0, upsampled bilinearly to the image size.

    Raises:
        ValueError: If the model is in train mode: dropout and batch statistics
            would couple the samples, and summing the logits over the batch
            would no longer give each sample its own gradient.
    """
    if model.training:
        raise ValueError("gradcam needs model.eval()")

    acts = []
    handle = layer.register_forward_hook(lambda _m, _i, out: acts.append(out))
    try:
        # The gradient is the point; a caller's no_grad would make A a leaf without grad_fn
        with torch.enable_grad():
            out = model(images)
            A = acts[0]
            # In eval each logit depends only on its own sample, so a grad_output of ones
            # gives every sample its own d(logit_b)/dA_b. It must be a real tensor: the
            # stride-0 one that .sum() backpropagates makes the Linear backward on MPS
            # return zero for 30 of 32 samples (only rows 0 and 16 survive).
            # autograd.grad does not accumulate into param.grad
            logit = out['logit']
            grads = torch.autograd.grad(logit, A, grad_outputs=torch.ones_like(logit))[0]
    finally:
        handle.remove()

    with torch.no_grad():
        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * A).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=images.shape[-2:], mode='bilinear', align_corners=False)
    return cam[:, 0].detach().cpu().numpy()


def _coverage(lo: float, hi: float, n: int) -> np.ndarray:
    """Fraction of each of n pixels on [0, 1] covered by the interval [lo, hi]."""
    edges = np.arange(n + 1) / n
    return np.clip(np.minimum(edges[1:], hi) - np.maximum(edges[:-1], lo), 0, None) * n


def energy_in_box(cam: np.ndarray, box: np.ndarray) -> float:
    """
    Fraction of the map's mass inside box.

    The 320x320 slice is resized to a square image, so normalized coordinates
    map linearly to the map's pixels. Pixels on the border count by the
    fraction the box covers, so a uniform map gives exactly box_area(box).

    Args:
        cam: Map of shape (H, W), >= 0.
        box: (x0, y0, x1, y1) normalized to [0, 1], as returned by SliceDataset.

    Returns:
        The fraction in [0, 1], or NaN if the map is all zero.
    """
    cam = np.asarray(cam, dtype=float)
    total = cam.sum()
    if total == 0:
        return float('nan')
    x0, y0, x1, y1 = np.clip(np.asarray(box, dtype=float), 0, 1)
    h, w = cam.shape
    inside = _coverage(y0, y1, h)[:, None] * _coverage(x0, x1, w)[None, :]
    return float((cam * inside).sum() / total)


def pointing_hit(cam: np.ndarray, box: np.ndarray) -> bool:
    """Whether the centre of the most activated pixel (first on ties) falls inside box."""
    cam = np.asarray(cam)
    h, w = cam.shape
    r, c = np.unravel_index(np.argmax(cam), cam.shape)
    x, y = (c + 0.5) / w, (r + 0.5) / h
    x0, y0, x1, y1 = box
    return bool(x0 <= x <= x1 and y0 <= y <= y1)


def box_area(box: np.ndarray) -> float:
    """Area of a normalized box: the energy_in_box a map spread uniformly would get by chance."""
    x0, y0, x1, y1 = np.clip(np.asarray(box, dtype=float), 0, 1)
    return float(max(x1 - x0, 0) * max(y1 - y0, 0))
