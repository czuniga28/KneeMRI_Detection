import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from scripts import evaluate_test
from src.eval.baseline import fixed_box_test
from src.models.net import SliceNet

TEST_EXAMS = [1001, 1002]
SEEDS = [42, 0]


def move_to_test(cfg: dict, exams: list[int]) -> None:
    """Relabel exams as 'test' in splits.csv and slices.csv of make_run_files."""
    interim = Path(cfg['data']['interim_dir'])
    slices_csv = Path(cfg['data']['processed_dir']) / 'slices.csv'
    for path in (interim / 'splits.csv', slices_csv):
        df = pd.read_csv(path)
        df.loc[df['examId'].isin(exams), 'split'] = 'test'
        df.to_csv(path, index=False)


@pytest.fixture
def setup(run_cfg, tmp_path, monkeypatch):
    """Two refit checkpoints with different random weights, a test split, and a clean preregistration."""
    move_to_test(run_cfg, TEST_EXAMS)
    cfg = {**run_cfg, 'val_fold': None, 'refit_epochs': 2, 'epochs': 4, 'tau': 0.5,
           'aggregation': {'z_method': 'kadane', 'xy_method': 'wmedian'}}
    refit_root = tmp_path / 'refit'
    for seed in SEEDS:
        torch.manual_seed(seed)
        model = SliceNet(**cfg['model'])
        (refit_root / f's{seed}').mkdir(parents=True)
        torch.save({'model': model.state_dict(), 'epoch': 1, 'tau': cfg['tau'], 'cfg': {**cfg, 'seed': seed}},
                   refit_root / f's{seed}' / 'last.pt')

    monkeypatch.setattr(evaluate_test, 'REPORTS_DIR', tmp_path / 'fase4')
    monkeypatch.setattr(evaluate_test, 'CV_SUMMARY', tmp_path / 'missing.csv')
    monkeypatch.setattr(evaluate_test, '_command', fake_git(tracked=True, clean=True))
    monkeypatch.setattr(sys, 'argv', ['evaluate_test.py', '--refit-root', str(refit_root), '--device', 'cpu'])
    return {'cfg': cfg, 'refit_root': refit_root, 'reports': tmp_path / 'fase4'}


def fake_git(tracked: bool, clean: bool):
    def _command(args):
        if 'ls-files' in args:
            return evaluate_test.PREREGISTRATION if tracked else 'unavailable: error: pathspec did not match'
        if 'status' in args:
            return '' if clean else f' M {evaluate_test.PREREGISTRATION}'
        raise AssertionError(f"unexpected command {args}")
    return _command


def test_refuses_when_already_evaluated(setup, monkeypatch):
    setup['reports'].mkdir()
    (setup['reports'] / 'test_volumes.csv').write_text('x\n')
    monkeypatch.setattr(evaluate_test.torch, 'load', lambda *a, **k: pytest.fail("loaded a model"))

    with pytest.raises(SystemExit, match='--force'):
        evaluate_test.main()


def test_force_overrides_existing_outputs(setup, monkeypatch):
    setup['reports'].mkdir()
    (setup['reports'] / 'test_volumes.csv').write_text('x\n')
    monkeypatch.setattr(sys, 'argv', sys.argv + ['--force'])

    evaluate_test.main()

    assert 'iou_m0' in pd.read_csv(setup['reports'] / 'test_volumes.csv')


@pytest.mark.parametrize('tracked, clean, match', [
    (False, True, 'not committed'),
    (True, False, 'uncommitted changes'),
])
def test_refuses_without_committed_preregistration(setup, monkeypatch, tracked, clean, match):
    monkeypatch.setattr(evaluate_test, '_command', fake_git(tracked, clean))
    monkeypatch.setattr(evaluate_test.torch, 'load', lambda *a, **k: pytest.fail("loaded a model"))

    with pytest.raises(SystemExit, match=match):
        evaluate_test.main()


@pytest.mark.parametrize('change', [{'tau': 0.6}, {'aggregation': {'z_method': 'threshold', 'xy_method': 'wmedian'}}])
def test_refuses_seeds_with_different_configs(setup, change):
    path = setup['refit_root'] / 's0' / 'last.pt'
    ck = torch.load(path, weights_only=False)
    ck['cfg'] = {**ck['cfg'], **change}
    torch.save(ck, path)

    with pytest.raises(ValueError, match=list(change)[0]):
        evaluate_test.main()


def test_common_config_ignores_seed():
    assert evaluate_test.common_config({42: {'seed': 42, 'tau': 0.5}, 0: {'seed': 0, 'tau': 0.5}})['tau'] == 0.5


def test_fixed_box_test_ignores_test_rois():
    roi = {'roiX': 100, 'roiY': 100, 'roiZ': 10, 'roiWidth': 80, 'roiHeight': 80, 'roiDepth': 4}
    rows = [{'examId': i, 'seriesNo': 5, 'aclDiagnosis': 0, 'split': f'fold_{i % 2}', **roi} for i in range(4)]
    rows.append({'examId': 100, 'seriesNo': 5, 'aclDiagnosis': 1, 'split': 'test', **roi})
    volumes = pd.DataFrame(rows)
    huge = {'examId': 101, 'seriesNo': 5, 'aclDiagnosis': 2, 'split': 'test',
            'roiX': 0, 'roiY': 0, 'roiZ': 0, 'roiWidth': 320, 'roiHeight': 320, 'roiDepth': 30}

    alone = fixed_box_test(volumes)
    with_huge = fixed_box_test(pd.concat([volumes, pd.DataFrame([huge])], ignore_index=True))

    # The dev boxes are identical, so the box is exactly the test volume's ROI
    assert alone['iou'].tolist() == pytest.approx([1.0])
    assert with_huge.set_index('examId').loc[100, 'iou'] == pytest.approx(1.0)
    assert set(with_huge['examId']) == {100, 101}


def test_evaluate_gets_the_fixed_tau(setup, monkeypatch):
    taus = []
    real_evaluate = evaluate_test.evaluate

    def recording_evaluate(*args, **kwargs):
        taus.append(kwargs.get('tau', args[3] if len(args) > 3 else None))
        return real_evaluate(*args, **kwargs)

    monkeypatch.setattr(evaluate_test, 'evaluate', recording_evaluate)
    evaluate_test.main()

    assert taus == [setup['cfg']['tau']] * len(SEEDS)


def test_main_writes_seed_averaged_volumes(setup):
    evaluate_test.main()

    table = pd.read_csv(setup['reports'] / 'test_volumes.csv')
    assert table.columns.tolist() == ['examId', 'seriesNo', 'aclDiagnosis', 'iou_s42', 'iou_s0', 'iou', 'iou_m0']
    assert set(table['examId']) == set(TEST_EXAMS)
    np.testing.assert_allclose(table['iou'], (table['iou_s42'] + table['iou_s0']) / 2)

    summary = pd.read_csv(setup['reports'] / 'test_summary.csv')
    assert summary.columns.tolist() == ['class', 'n', 'iou', 'ci_lo', 'ci_hi', 'frac_ge_05',
                                        'seed_std', 'delta_m0', 'm0_lo', 'm0_hi']
    assert summary.loc[0, 'class'] == 'all' and summary.loc[0, 'n'] == len(table)
    assert summary.loc[0, 'iou'] == pytest.approx(table['iou'].mean())

    assert pd.read_csv(setup['reports'] / 'test_slices.csv')['seed'].tolist() == SEEDS
    for seed in SEEDS:
        preds = pd.read_csv(setup['refit_root'] / f's{seed}' / 'preds_test.csv')
        assert set(preds['examId']) == set(TEST_EXAMS)
