import json
import sys

import pytest
import yaml

from scripts import refit as refit_script
from src.train.loop import PROJECT_ROOT

METRICS = {'epochs_trained': 2, 'final_lr': 1e-4, 'seconds_per_epoch': 1.0}


@pytest.fixture
def setup(run_cfg, tmp_path, monkeypatch):
    """Refit config on disk; refit and write_provenance replaced by recorders."""
    cfg = {**run_cfg, 'val_fold': None, 'refit_epochs': 2, 'epochs': 4, 'tau': 0.5}
    config = tmp_path / 'm1_refit.yaml'
    config.write_text(yaml.safe_dump(cfg))
    monkeypatch.setattr(refit_script, 'OUT_ROOT', tmp_path / 'refit')
    monkeypatch.setattr(refit_script, 'SMOKE_OUT_ROOT', tmp_path / 'refit_smoke')

    calls = []

    def fake_refit(cfg, out, max_slices=None, device=None):
        calls.append({'seed': cfg['seed'], 'out': out, 'max_slices': max_slices})
        return METRICS

    monkeypatch.setattr(refit_script, 'refit', fake_refit)
    monkeypatch.setattr(refit_script, 'write_provenance', lambda out, cfg: None)
    monkeypatch.setattr(sys, 'argv', ['refit.py', '--config', str(config), '--seeds', '42', '0'])
    return calls


def test_main_skips_finished_seeds(setup, tmp_path):
    done = tmp_path / 'refit' / 's42'
    done.mkdir(parents=True)
    (done / 'metrics.json').write_text(json.dumps(METRICS))

    refit_script.main()

    assert [c['seed'] for c in setup] == [0]
    assert setup[0]['out'] == tmp_path / 'refit' / 's0'


def test_max_slices_defaults_to_the_smoke_root(setup, tmp_path, monkeypatch):
    # A finished real run must not make the smoke skip, nor the smoke write into it
    done = tmp_path / 'refit' / 's42'
    done.mkdir(parents=True)
    (done / 'metrics.json').write_text(json.dumps(METRICS))
    monkeypatch.setattr(sys, 'argv', sys.argv + ['--max-slices', '16'])

    refit_script.main()

    assert [c['out'] for c in setup] == [tmp_path / 'refit_smoke' / 's42', tmp_path / 'refit_smoke' / 's0']
    assert all(c['max_slices'] == 16 for c in setup)


def test_default_roots():
    assert refit_script.OUT_ROOT == PROJECT_ROOT / 'runs' / 'refit'
    assert refit_script.SMOKE_OUT_ROOT == PROJECT_ROOT / 'runs' / 'refit_smoke'


def test_main_checks_config_before_any_seed(setup, tmp_path, monkeypatch):
    cfg = yaml.safe_load((tmp_path / 'm1_refit.yaml').read_text())
    (tmp_path / 'm1_refit.yaml').write_text(yaml.safe_dump({**cfg, 'val_fold': 'fold_0'}))

    with pytest.raises(ValueError, match='use run'):
        refit_script.main()
    assert setup == []
