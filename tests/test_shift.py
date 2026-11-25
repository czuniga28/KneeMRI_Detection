import numpy as np
import pandas as pd
import pytest

from scripts.simulate_shift import center_crop_hw, gamma, noise, parse_shift, zoom_hw
from src.deploy.predict import MAX_INTENSITY
from src.eval.shift import gt_after_shift, map_coord, scale_response, zoomed_shape
from src.train.loop import PROJECT_ROOT

H, W = 300, 310


def box(x0, y0, x1, y1, z0=2, z1=5):
    return np.array([[x0, y0, z0, x1, y1, z1]], dtype=float)


def mask_bbox_after_zoom(b, f):
    """Draw b as a mask, push it through the real shift and take the box of mask > 0.5."""
    mask = np.zeros((1, H, W), dtype=np.float64)
    x0, y0, _, x1, y1, _ = b[0].astype(int)
    mask[0, y0:y1, x0:x1] = 1.0
    out = center_crop_hw(zoom_hw(mask, f))[0] > 0.5
    rows, cols = np.where(out.any(axis=1))[0], np.where(out.any(axis=0))[0]
    return np.array([cols[0], rows[0], cols[-1] + 1, rows[-1] + 1], dtype=float)


def test_zoomed_shape_and_map_coord_match_ndimage_conventions():
    assert zoomed_shape(300, 1.3) == 390 and zoomed_shape(310, 0.7) == 217
    # A pixel centre (edge + 0.5) lands on centre * s
    s = (390 - 1) / (300 - 1)
    assert map_coord(10.5, 300, 390) == pytest.approx(10 * s + 0.5)
    assert map_coord(0.5, 300, 390) == pytest.approx(0.5)


def test_gt_after_shift_identity_at_f_1():
    gt = np.vstack([box(10.3, 20, 50, 80), box(0, 0, 310, 300)])
    out, clipped = gt_after_shift(gt, H, W, 1.0)
    np.testing.assert_array_equal(out, gt)
    assert not clipped.any()


@pytest.mark.parametrize('f', [0.7, 1.3])
def test_gt_after_shift_matches_ndimage_mask(f):
    b = box(120, 100, 190, 170)
    got, clipped = gt_after_shift(b, H, W, f)
    expected = mask_bbox_after_zoom(b, f)
    np.testing.assert_allclose(got[0, [0, 1, 3, 4]], expected, atol=1)
    np.testing.assert_array_equal(got[0, [2, 5]], b[0, [2, 5]])
    assert not clipped[0]


def test_gt_after_shift_crop_offset_and_clipped_flag():
    f = 1.3
    hz, wz = 390, 403
    top, left = (hz - 320) // 2, (wz - 320) // 2
    inside = box(120, 100, 190, 170)
    got, clipped = gt_after_shift(inside, H, W, f)
    assert got[0, 0] == pytest.approx(map_coord(120, W, wz) - left)
    assert got[0, 1] == pytest.approx(map_coord(100, H, hz) - top)
    assert not clipped[0]

    # Near the left edge: after zoom x0 ~ 6, the crop removes 41 columns
    edge = box(5, 100, 80, 170)
    got, clipped = gt_after_shift(edge, H, W, f)
    assert got[0, 0] == 0 and clipped[0]
    np.testing.assert_allclose(got[0, [0, 1, 3, 4]], mask_bbox_after_zoom(edge, f), atol=1)


def test_scale_response():
    base = np.vstack([box(100, 100, 180, 190), box(50, 60, 120, 140)])
    f = 0.8

    def scaled(b, k):
        c = (b[:, [0, 1]] + b[:, [3, 4]]) / 2
        half = (b[:, [3, 4]] - b[:, [0, 1]]) / 2 * k
        out = b.copy()
        out[:, [0, 1]], out[:, [3, 4]] = c - half, c + half
        return out

    np.testing.assert_allclose(scale_response(scaled(base, f), base, f), 1.0)
    np.testing.assert_allclose(scale_response(base, base, f), 0.0)
    with pytest.raises(ValueError):
        scale_response(base, base, 1.0)


def test_gamma_and_noise_keep_dtype_range_and_seed():
    raw = np.random.default_rng(3).integers(0, 4096, size=(3, 40, 50)).astype(np.uint16)
    for out in (gamma(raw, 0.5), gamma(raw, 2.0), noise(raw, 0.05), noise(raw, 0.10)):
        assert out.dtype == np.uint16 and out.shape == raw.shape
        assert out.min() >= 0 and out.max() <= MAX_INTENSITY
    assert gamma(np.array([0, 4095], dtype=np.uint16), 0.5).tolist() == [0, 4095]
    np.testing.assert_array_equal(noise(raw, 0.05), noise(raw, 0.05))
    assert not np.array_equal(noise(raw, 0.05), raw)


def test_parse_shift():
    t, f = parse_shift('none')
    raw = np.ones((3, 4, 5), dtype=np.uint16)
    assert f == 1.0 and t(raw) is raw
    assert parse_shift('zoom125')[1] == 1.25 and parse_shift('gamma200')[1] == 1.0
    with pytest.raises(ValueError):
        parse_shift('blur050')


SHIFT_NONE = PROJECT_ROOT / 'reports' / 'fase6' / 'shift_none.csv'
TEST_VOLUMES = PROJECT_ROOT / 'reports' / 'fase4' / 'test_volumes.csv'


@pytest.mark.skipif(not (SHIFT_NONE.exists() and TEST_VOLUMES.exists()), reason="needs the real shift_none run")
def test_shift_none_reproduces_phase4_iou():
    from scripts.shift_analysis import dev_signals, score_shift, test_ground_truth
    from src.deploy.monitor import load_reference

    ref = load_reference(PROJECT_ROOT / 'reports' / 'fase6' / 'monitor_reference.json')
    got = score_shift('none', pd.read_csv(SHIFT_NONE), test_ground_truth(), ref, dev_signals())
    expected = pd.read_csv(TEST_VOLUMES)
    merged = got.merge(expected, on=['examId', 'seriesNo'], validate='one_to_one', suffixes=('', '_f4'))
    assert len(merged) == len(expected) == 153
    np.testing.assert_allclose(merged['iou'], merged['iou_s42'], atol=1e-6)
