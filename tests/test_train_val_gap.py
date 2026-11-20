import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from scripts import train_val_gap
from src.models.net import SliceNet
from src.train import loop
from src.train.loop import KEY, predict_checkpoint

from tests.test_evaluate_test import move_to_test

TEST_EXAMS = [1002]
RUN = 'fold_1_s0'


@pytest.fixture
def setup(run_cfg, tmp_path, monkeypatch):
    """One CV run (val_fold fold_1) with random weights, a test split and a refit config with tau."""
    move_to_test(run_cfg, TEST_EXAMS)
    cfg = {**run_cfg, 'aggregation': {'z_method': 'kadane', 'xy_method': 'wmedian'}}
    run_dir = tmp_path / 'cv' / RUN
    run_dir.mkdir(parents=True)
    torch.manual_seed(0)
    ck = {'model': SliceNet(**cfg['model']).state_dict(), 'epoch': 0, 'tau': 0.5, 'cfg': cfg}
    torch.save(ck, run_dir / 'best.pt')
    (run_dir / 'config.yaml').write_text(yaml.safe_dump(cfg))
    slices = pd.read_csv(Path(cfg['data']['processed_dir']) / 'slices.csv')
    val = slices[slices['split'] == cfg['val_fold']].reset_index(drop=True)
    predict_checkpoint(ck, val, torch.device('cpu')).to_csv(run_dir / 'preds_val.csv', index=False)

    refit_cfg = tmp_path / 'm1_refit.yaml'
    refit_cfg.write_text(yaml.safe_dump({**cfg, 'val_fold': None, 'tau': 0.4}))
    monkeypatch.setattr(train_val_gap, 'REFIT_CFG', refit_cfg)
    monkeypatch.setattr(train_val_gap, 'REPORTS_DIR', tmp_path / 'fase3')
    monkeypatch.setattr(sys, 'argv', ['train_val_gap.py', '--runs-dir', str(tmp_path / 'cv'), '--device', 'cpu'])
    return {'cfg': cfg, 'run_dir': run_dir, 'slices': slices, 'reports': tmp_path / 'fase3', 'tau': 0.4}


def test_train_preds_exclude_val_fold_and_test(setup):
    preds = train_val_gap.train_predictions(setup['run_dir'], setup['slices'], torch.device('cpu'))

    splits = preds[KEY].drop_duplicates().merge(setup['slices'][KEY + ['split']].drop_duplicates(), on=KEY)
    assert set(splits['split']) == {'fold_0'}
    assert not set(preds['examId']) & set(TEST_EXAMS)
    assert (setup['run_dir'] / 'preds_train.csv').exists()


def test_both_evaluates_get_the_given_tau(setup, monkeypatch):
    taus = []
    real_evaluate = train_val_gap.evaluate

    def recording_evaluate(*args, **kwargs):
        taus.append(kwargs['tau'])
        return real_evaluate(*args, **kwargs)

    monkeypatch.setattr(train_val_gap, 'evaluate', recording_evaluate)
    train_val_gap.main()

    assert taus == [setup['tau'], setup['tau']]


def test_cached_train_preds_skip_the_model(setup, monkeypatch):
    first = train_val_gap.train_predictions(setup['run_dir'], setup['slices'], torch.device('cpu'))
    monkeypatch.setattr(train_val_gap.torch, 'load', lambda *a, **k: pytest.fail("loaded a model"))

    second = train_val_gap.train_predictions(setup['run_dir'], setup['slices'], torch.device('cpu'))

    pd.testing.assert_frame_equal(first, second)


def test_gap_is_train_minus_val(setup):
    train_val_gap.main()

    table = pd.read_csv(setup['reports'] / 'train_val_gap.csv')
    assert table['run'].tolist() == [RUN]
    assert table.loc[0, 'fold'] == 'fold_1' and table.loc[0, 'seed'] == 0
    np.testing.assert_allclose(table['gap'], table['iou_train'] - table['iou_val'])
    for c in (0, 1, 2):
        np.testing.assert_allclose(
            table[f'gap_class_{c}'], table[f'iou_train_class_{c}'] - table[f'iou_val_class_{c}'],
        )
    summary = pd.read_csv(setup['reports'] / 'train_val_gap_summary.csv')
    assert summary['class'].astype(str).tolist() == ['all', '0', '1', '2']


def test_summarize_gap_averages_runs():
    table = pd.DataFrame({
        'iou_train': [0.8, 0.6], 'iou_val': [0.5, 0.55],
        'iou_train_class_0': [0.8, 0.6], 'iou_val_class_0': [0.75, 0.55],
        'iou_train_class_1': [0.7, np.nan], 'iou_val_class_1': [0.4, np.nan],
        'iou_train_class_2': [np.nan, np.nan], 'iou_val_class_2': [np.nan, np.nan],
    })
    table['gap'] = table['iou_train'] - table['iou_val']
    for c in (0, 1, 2):
        table[f'gap_class_{c}'] = table[f'iou_train_class_{c}'] - table[f'iou_val_class_{c}']

    s = train_val_gap.summarize_gap(table).set_index('class')

    assert s.loc['all', ['n_runs', 'iou_train', 'iou_val', 'gap', 'gap_min', 'gap_max']].tolist() == \
        pytest.approx([2, 0.7, 0.525, 0.175, 0.05, 0.3])
    assert not s.loc['all', 'meets']
    assert s.loc['0', 'gap'] == pytest.approx(0.05) and s.loc['0', 'meets']
    assert s.loc['1', 'n_runs'] == 1 and s.loc['1', 'gap'] == pytest.approx(0.3)
    assert s.loc['2', 'n_runs'] == 0


def test_train_dataset_has_no_augmentation(setup, monkeypatch):
    augment = []
    real_dataset = loop.SliceDataset

    def recording_dataset(*args, **kwargs):
        augment.append(kwargs.get('augment', False))
        return real_dataset(*args, **kwargs)

    monkeypatch.setattr(loop, 'SliceDataset', recording_dataset)
    train_val_gap.train_predictions(setup['run_dir'], setup['slices'], torch.device('cpu'))

    assert augment == [False]
