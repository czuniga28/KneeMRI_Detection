import pickle
import sys

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from scripts import predict as predict_script
from src.data.transforms import normalize_volume, pad_volume, preprocess_volume
from src.deploy.predict import check_input, load_model, locate, predict_slices
from src.models.net import SliceNet
from src.train.dataset import BOX_COLUMNS, SliceDataset, slice_image

PREP = {'clip_low_pct': 0.5, 'clip_high_pct': 99.5, 'pad_height': 320, 'pad_width': 320,
        'storage_dtype': 'float16'}
CFG = {'image_size': 32, 'batch_size': 4, 'context': 1, 'tau': 0.5,
       'data': {'width': 320, 'height': 320},
       'aggregation': {'z_method': 'kadane', 'xy_method': 'wmedian'},
       'model': {'pretrained': False, 'dropout': 0.2}}


def raw_volume(shape=(6, 300, 310), seed=0) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 4096, size=shape).astype(np.uint16)


def test_preprocess_volume_matches_the_original_three_lines():
    raw = raw_volume()

    # scripts/build_dataset.py before the extraction
    expected = normalize_volume(raw, PREP['clip_low_pct'], PREP['clip_high_pct'])
    expected = pad_volume(expected, PREP['pad_height'], PREP['pad_width'])
    expected = expected.astype(PREP['storage_dtype'])

    got = preprocess_volume(raw, PREP)
    assert got.dtype == np.float16 and got.shape == (6, 320, 320)
    np.testing.assert_array_equal(got, expected)


def test_slice_image_matches_slice_dataset(tmp_path):
    vol = preprocess_volume(raw_volume(), PREP)
    (tmp_path / 'volumes').mkdir()
    np.save(tmp_path / 'volumes' / 'v.npy', vol)
    slices = pd.DataFrame({
        'volume_file': 'volumes/v.npy', 'slice_idx': range(len(vol)), 'contains_acl': False,
        **{c: np.nan for c in BOX_COLUMNS},
    })
    ds = SliceDataset(slices, tmp_path, size=32, augment=False, context=1)

    for s in range(len(vol)):
        assert torch.equal(ds[s]['image'], slice_image(vol, s, 32, 1))


@pytest.mark.parametrize('shape', [(320, 320), (5, 320, 321), (2, 320, 320)])
def test_check_input_rejects(shape):
    with pytest.raises(ValueError):
        check_input(np.zeros(shape, dtype=np.uint16))


def test_check_input_warns_but_passes():
    with pytest.warns(UserWarning, match='dtype'):
        check_input(np.zeros((5, 64, 64), dtype=np.float32))
    with pytest.warns(UserWarning, match='4095'):
        check_input(np.full((5, 64, 64), 5000, dtype=np.uint16))


def test_locate_clips_to_raw_shape_and_flags_review():
    shape = (6, 300, 310)
    # Every slice predicts a box reaching into the padding (x1 = y1 = 320)
    boxes = np.tile([100.0, 120.0, 320.0, 320.0], (6, 1))

    sure = locate(np.array([0.1, 0.9, 0.95, 0.9, 0.1, 0.1]), boxes, CFG, shape)
    assert (sure['x1'], sure['y1']) == (310, 300)
    assert (sure['z0'], sure['z1']) == (1, 4)
    assert not sure['needs_review']
    assert sure['p_max'] == pytest.approx(0.95)
    assert sure['p_run'] == pytest.approx((0.9 + 0.95 + 0.9) / 3)

    unsure = locate(np.full(6, 0.2), boxes, CFG, shape)
    assert unsure['needs_review']
    assert unsure['z1'] - unsure['z0'] == 1


def test_predict_slices_shapes_and_ranges():
    torch.manual_seed(0)
    model = SliceNet(**CFG['model']).eval()
    vol = preprocess_volume(raw_volume((7, 64, 64)), PREP)

    p, boxes = predict_slices(model, vol, CFG, torch.device('cpu'))

    assert p.shape == (7,) and boxes.shape == (7, 4)
    assert ((p >= 0) & (p <= 1)).all()
    assert ((boxes >= 0) & (boxes <= 320)).all()


def test_load_model_rejects_checkpoint_without_tau(tmp_path):
    # A CV best.pt: candidate taus, no fixed tau
    cfg = {k: v for k, v in CFG.items() if k != 'tau'} | {'taus': [0.3, 0.5, 0.7]}
    ck = tmp_path / 'best.pt'
    torch.save({'model': SliceNet(**cfg['model']).state_dict(), 'cfg': cfg}, ck)

    with pytest.raises(ValueError, match='no fixed tau'):
        load_model(ck, torch.device('cpu'))


def test_cli_keeps_going_after_a_bad_file(tmp_path, monkeypatch):
    torch.manual_seed(0)
    ck = tmp_path / 'last.pt'
    torch.save({'model': SliceNet(**CFG['model']).state_dict(), 'cfg': CFG}, ck)
    prep_cfg = tmp_path / 'prep.yaml'
    prep_cfg.write_text(yaml.safe_dump({'preprocess': PREP}))
    good, bad = tmp_path / 'good.pck', tmp_path / 'bad.pck'
    good.write_bytes(pickle.dumps(raw_volume((5, 64, 64))))
    bad.write_bytes(pickle.dumps(raw_volume((64, 64))))
    out = tmp_path / 'preds.csv'

    monkeypatch.setattr(predict_script, 'PREP_CFG', prep_cfg)
    monkeypatch.setattr(sys, 'argv', ['predict.py', str(bad), str(good), '--checkpoint', str(ck),
                                      '--out', str(out), '--device', 'cpu'])
    predict_script.main()

    table = pd.read_csv(out).set_index('file')
    assert list(table.columns) == predict_script.COLUMNS[1:]
    assert table.loc[str(bad), 'error'].startswith('ValueError')
    assert table.loc[str(bad), ['x0', 'z1']].isna().all()
    assert pd.isna(table.loc[str(good), 'error'])
    assert table.loc[str(good), ['depth', 'height', 'width']].tolist() == [5, 64, 64]
    assert 0 <= table.loc[str(good), 'x0'] <= table.loc[str(good), 'x1'] <= 64
