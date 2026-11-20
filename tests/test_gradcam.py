import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from scripts import gradcam_analysis
from src.eval.bootstrap import bootstrap_spearman
from src.eval.gradcam import box_area, energy_in_box, gradcam, pointing_hit
from src.models.net import SliceNet
from src.train.loop import KEY, predict_checkpoint

from tests.test_evaluate_test import move_to_test

TEST_EXAMS = [1002]
FOLDS = ['fold_0', 'fold_1']


@pytest.fixture
def model():
    torch.manual_seed(0)
    return SliceNet(pretrained=False).eval()


def test_gradcam_shape_nonnegative_and_no_hooks_left(model):
    layer = model.backbone.layer4
    cam = gradcam(model, torch.randn(2, 3, 224, 224), layer)

    assert cam.shape == (2, 224, 224)
    assert (cam >= 0).all()
    assert len(layer._forward_hooks) == 0


def test_gradcam_leaves_no_param_grads(model):
    gradcam(model, torch.randn(2, 3, 224, 224), model.backbone.layer4)

    assert all(p.grad is None for p in model.parameters())


DEVICES = ['cpu'] + (['mps'] if torch.backends.mps.is_available() else [])


@pytest.mark.parametrize('device', DEVICES)
def test_gradcam_batch_matches_one_slice_at_a_time(model, device):
    # Regression: on MPS, backpropagating logit.sum() zeroed every sample but rows 0
    # and 16 of a batch; B >= 17 makes the second surviving row part of the batch
    model.to(device)
    torch.manual_seed(1)
    x = torch.randn(20, 3, 224, 224, device=device)
    layer = model.backbone.layer4

    batched = gradcam(model, x, layer)
    single = np.stack([gradcam(model, x[i:i + 1], layer)[0] for i in range(len(x))])

    assert (single.reshape(len(x), -1).sum(1) > 0).all()
    np.testing.assert_allclose(batched, single, rtol=1e-3, atol=1e-4 * single.max())


def test_gradcam_rejects_train_mode(model):
    model.train()
    with pytest.raises(ValueError, match='eval'):
        gradcam(model, torch.randn(2, 3, 64, 64), model.backbone.layer4)
    assert len(model.backbone.layer4._forward_hooks) == 0


def test_energy_of_uniform_map_is_box_area():
    box = np.array([0.13, 0.27, 0.61, 0.9])
    assert energy_in_box(np.ones((224, 224)), box) == pytest.approx(box_area(box), abs=1e-12)


def test_energy_of_map_only_inside_box_is_one():
    cam = np.zeros((224, 224))
    cam[60:120, 50:100] = np.random.default_rng(0).random((60, 50)) + 0.1
    box = np.array([50, 60, 100, 120]) / 224

    assert energy_in_box(cam, box) == pytest.approx(1.0)


def test_energy_of_zero_map_is_nan():
    assert np.isnan(energy_in_box(np.zeros((8, 8)), np.array([0, 0, 0.5, 0.5])))


def test_pointing_hit():
    box = np.array([0.25, 0.25, 0.5, 0.5])
    cam = np.zeros((100, 100))
    cam[40, 30] = 1.0
    assert pointing_hit(cam, box)

    cam = np.zeros((100, 100))
    cam[80, 10] = 1.0
    assert not pointing_hit(cam, box)


def test_positive_slices_reject_test(run_cfg):
    move_to_test(run_cfg, TEST_EXAMS)
    slices = pd.read_csv(Path(run_cfg['data']['processed_dir']) / 'slices.csv')

    pos = gradcam_analysis.positive_slices(slices, 'fold_0')
    assert pos['contains_acl'].all() and set(pos['split']) == {'fold_0'}
    assert not set(pos['examId']) & set(TEST_EXAMS)
    with pytest.raises(ValueError, match='development'):
        gradcam_analysis.positive_slices(slices, 'test')


@pytest.fixture
def runs(run_cfg, tmp_path, monkeypatch):
    """Seed-42 CV runs for fold_0 and fold_1 with random weights, a test split and a refit config with tau."""
    move_to_test(run_cfg, TEST_EXAMS)
    slices = pd.read_csv(Path(run_cfg['data']['processed_dir']) / 'slices.csv')
    for fold in FOLDS:
        cfg = {**run_cfg, 'val_fold': fold, 'seed': 42,
               'aggregation': {'z_method': 'kadane', 'xy_method': 'wmedian'}}
        run_dir = tmp_path / 'cv' / f'{fold}_s42'
        run_dir.mkdir(parents=True)
        torch.manual_seed(len(fold))
        ck = {'model': SliceNet(**cfg['model']).state_dict(), 'epoch': 0, 'tau': 0.5, 'cfg': cfg}
        torch.save(ck, run_dir / 'best.pt')
        (run_dir / 'config.yaml').write_text(yaml.safe_dump(cfg))
        val = slices[slices['split'] == fold].reset_index(drop=True)
        predict_checkpoint(ck, val, torch.device('cpu')).to_csv(run_dir / 'preds_val.csv', index=False)

    refit_cfg = tmp_path / 'm1_refit.yaml'
    refit_cfg.write_text(yaml.safe_dump({**run_cfg, 'val_fold': None, 'tau': 0.5}))
    monkeypatch.setattr(gradcam_analysis, 'FOLDS', FOLDS)
    monkeypatch.setattr(gradcam_analysis, 'REFIT_CFG', refit_cfg)
    monkeypatch.setattr(gradcam_analysis, 'REPORTS_DIR', tmp_path / 'fase3')
    monkeypatch.setattr(gradcam_analysis, 'FIGS_DIR', tmp_path / 'figs')
    monkeypatch.setattr(sys, 'argv', ['gradcam_analysis.py', '--runs-dir', str(tmp_path / 'cv'), '--device', 'cpu'])
    return {'slices': slices, 'reports': tmp_path / 'fase3'}


def test_script_uses_only_positive_dev_slices(runs):
    gradcam_analysis.main()

    scored = pd.read_csv(runs['reports'] / 'gradcam_slices.csv')
    m = scored.drop(columns=['split', 'aclDiagnosis']).merge(
        runs['slices'], on=KEY + ['slice_idx'], how='left', validate='many_to_one',
    )
    assert m['contains_acl'].all()
    assert set(m['split']) <= set(FOLDS)
    assert not set(m['examId']) & set(TEST_EXAMS)
    trained = m[m['model'] == 'trained']
    assert (trained['run'].str.rsplit('_s', n=1).str[0] == trained['split']).all()
    assert set(m.loc[m['model'] == 'random', 'split']) == {FOLDS[0]}

    summary = pd.read_csv(runs['reports'] / 'gradcam_summary.csv')
    assert set(summary['model']) == {'trained', 'random'}
    control = pd.read_csv(runs['reports'] / 'gradcam_control.csv')
    vols = pd.read_csv(runs['reports'] / 'gradcam_volumes.csv').dropna(subset=['energy'])
    paired = vols[vols['model'] == 'trained'].merge(vols.loc[vols['model'] == 'random', KEY], on=KEY)
    assert control.set_index('class').loc['all', 'n_volumes'] == len(paired)


def test_paired_control_pairs_by_volume():
    # The trained model also scored fold_1 volumes; only the shared fold_0 ones are paired
    vols = pd.DataFrame({
        'model': ['trained'] * 3 + ['random'] * 2,
        'examId': [1, 2, 3, 1, 2], 'seriesNo': [5] * 5, 'aclDiagnosis': [0, 1, 0, 0, 1],
        'energy': [0.6, 0.5, 0.9, 0.2, 0.4],
    })
    c = gradcam_analysis.paired_control(vols).set_index('class')

    assert c.loc['all', 'n_volumes'] == 2
    assert c.loc['all', 'delta'] == pytest.approx(((0.6 - 0.2) + (0.5 - 0.4)) / 2)
    assert c.loc['0', 'delta'] == pytest.approx(0.4)
    assert c.loc['2', 'n_volumes'] == 0 and np.isnan(c.loc['2', 'delta'])


def test_bootstrap_spearman():
    x = np.arange(30.0)
    perfect = bootstrap_spearman(x, x ** 3)
    assert perfect['rho'] == pytest.approx(1.0)
    assert perfect['lo'] == pytest.approx(1.0) and perfect['hi'] == pytest.approx(1.0)

    rng = np.random.default_rng(0)
    y = x + rng.normal(0, 10, size=30)
    r = bootstrap_spearman(x, y)
    assert r['lo'] <= r['rho'] <= r['hi']
    with pytest.raises(ValueError):
        bootstrap_spearman(x[:2], y[:2])
