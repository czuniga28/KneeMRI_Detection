import numpy as np
import pandas as pd
import pytest

from scripts import monitor as monitor_script
from src.deploy.monitor import (
    MIN_REF, box_features, check_batch, false_alarm_prob, fit_reference, load_reference, outside,
    reference_rows, save_reference,
)


def feats_of(width, height=None) -> pd.DataFrame:
    width = np.asarray(width, dtype=float)
    return pd.DataFrame({'width': width, 'height': width if height is None else np.asarray(height, float)})


def ref_1_100():
    return fit_reference(feats_of(np.arange(1, 101)))


def preds_row(file, x0=10., y0=20., z0=3, x1=50., y1=80., z1=6, needs_review=False, error=np.nan):
    return {'file': file, 'x0': x0, 'y0': y0, 'z0': z0, 'x1': x1, 'y1': y1, 'z1': z1,
            'needs_review': needs_review, 'error': error}


def test_fit_reference_matches_percentile_and_fences():
    x = np.arange(1, 101)
    ref = ref_1_100()
    q1, q3 = np.percentile(x, [25, 75])
    b = ref['features']['width']
    assert b['q1'] == q1 and b['q3'] == q3
    assert b['lo'] == pytest.approx(q1 - 1.5 * (q3 - q1))
    assert b['hi'] == pytest.approx(q3 + 1.5 * (q3 - q1))
    assert ref['n'] == 100 and ref['k'] == 1.5 and ref['base_rate'] == 0.0


def test_fit_reference_rejects_zero_iqr_and_small_n():
    with pytest.raises(ValueError, match='IQR'):
        fit_reference(feats_of(np.arange(1, 101), np.full(100, 7.0)))
    with pytest.raises(ValueError, match='rows'):
        fit_reference(feats_of(np.arange(MIN_REF - 1)))


def test_outside_is_strict_and_any_feature_counts():
    ref = ref_1_100()
    lo, hi = ref['features']['width']['lo'], ref['features']['width']['hi']
    mid = 50.0
    f = feats_of([lo, hi, lo - 1e-6, hi + 1e-6, mid], [mid, mid, mid, mid, hi + 1e-6])
    assert outside(f, ref).tolist() == [False, False, True, True, True]


def test_check_batch_threshold_is_strict_and_small_batches_do_not_decide():
    ref = ref_1_100()
    out = ref['features']['width']['hi'] + 10

    def batch(n, n_out):
        return feats_of([out] * n_out + [50.0] * (n - n_out))

    res = check_batch(batch(100, 10), ref)
    assert res['frac_out'] == 0.10 and res['alert'] is False and res['status'] == 'ok'
    res = check_batch(batch(100, 11), ref)
    assert res['alert'] is True and res['status'] == 'alert'
    res = check_batch(batch(19, 19), ref)
    assert res['status'] == 'insufficient' and res['alert'] is None and res['n_out'] == 19


def test_box_features_drops_errors_and_nans_keeps_needs_review():
    preds = pd.DataFrame([
        preds_row('a.pck'),
        preds_row('b.pck', needs_review=True),
        preds_row('c.pck', x0=np.nan, error='ValueError: bad'),
        preds_row('d.pck', y1=np.nan),
        preds_row('e.pck', error='OSError: x'),
    ])
    f = box_features(preds)
    assert f['file'].tolist() == ['a.pck', 'b.pck']
    assert f.attrs['n_dropped'] == 3
    assert f['width'].tolist() == [40.0, 40.0]
    assert f['height'].tolist() == [60.0, 60.0]
    assert f['z_extent'].tolist() == [4, 4]


def test_reference_json_round_trip(tmp_path):
    ref = fit_reference(feats_of(np.random.default_rng(0).normal(100, 10, 200)))
    save_reference(ref, tmp_path / 'sub' / 'ref.json')
    assert load_reference(tmp_path / 'sub' / 'ref.json') == ref


def test_reference_rows_excludes_test_and_rejects_unknown_files():
    splits = pd.DataFrame({'volumeFilename': ['a.pck', 'b.pck', 'c.pck'],
                           'split': ['fold_0', 'test', 'fold_3']})
    preds = pd.DataFrame([preds_row('raw/a.pck'), preds_row('raw/b.pck'), preds_row('raw/c.pck')])
    assert reference_rows(preds, splits)['file'].tolist() == ['raw/a.pck', 'raw/c.pck']
    with pytest.raises(ValueError, match='not in splits'):
        reference_rows(pd.concat([preds, pd.DataFrame([preds_row('raw/z.pck')])]), splits)


def test_false_alarm_prob_zero_at_p0_zero_and_monotone():
    assert false_alarm_prob(0.0, 50, 0.1) == 0
    probs = [false_alarm_prob(p, 50, 0.1) for p in np.linspace(0, 0.5, 11)]
    assert all(a < b for a, b in zip(probs, probs[1:]))


def test_cli_fit_then_check_appends_summary_rows(tmp_path):
    rng = np.random.default_rng(1)
    names = [f'v{i}.pck' for i in range(80)]
    w, h = rng.normal(80, 8, 80), rng.normal(90, 8, 80)
    preds = pd.DataFrame([preds_row(f'raw/{n}', x0=0., y0=0., x1=a, y1=b) for n, a, b in zip(names, w, h)])
    preds['error'] = preds['error'].astype(object)
    preds.loc[0, 'error'] = 'ValueError: bad'
    split = ['test' if i < 25 else f'fold_{i % 5}' for i in range(80)]
    splits = pd.DataFrame({'volumeFilename': names, 'split': split})
    preds.to_csv(tmp_path / 'p.csv', index=False)
    splits.to_csv(tmp_path / 's.csv', index=False)
    ref_path, summary = tmp_path / 'ref.json', tmp_path / 'summary.csv'

    monitor_script.main(['fit', '--predictions', str(tmp_path / 'p.csv'), '--splits', str(tmp_path / 's.csv'),
                         '--out', str(ref_path)])
    assert load_reference(ref_path)['n'] == 55

    common = ['--reference', str(ref_path), '--summary', str(summary)]
    monitor_script.main(['check', '--predictions', str(tmp_path / 'p.csv'), '--name', 'test',
                         '--split', 'test', '--splits', str(tmp_path / 's.csv'), *common])
    shifted = preds.assign(x1=preds['x1'] * 3)
    shifted.to_csv(tmp_path / 'shift.csv', index=False)
    monitor_script.main(['check', '--predictions', str(tmp_path / 'shift.csv'), '--name', 'shift', *common])

    table = pd.read_csv(summary)
    assert table.columns.tolist() == monitor_script.SUMMARY_COLUMNS
    assert table['name'].tolist() == ['test', 'shift']
    assert table['n'].tolist() == [24, 79] and table['n_dropped'].tolist() == [1, 1]
    assert bool(table.loc[1, 'alert']) and table.loc[1, 'out_width'] > table.loc[1, 'out_height']


def test_shift_transforms_keep_dtype_and_respect_limits():
    from scripts.simulate_shift import parse_shift
    from src.deploy.predict import check_input
    raw = np.random.default_rng(0).integers(0, 4096, size=(4, 300, 310)).astype(np.uint16)
    z = parse_shift('zoom080')[0](raw)
    assert z.dtype == np.uint16 and z.shape == (4, 240, 248)
    z = parse_shift('zoom125')[0](raw)
    assert z.dtype == np.uint16 and z.shape == (4, 320, 320)
    check_input(z)
    g = parse_shift('gain070')[0](np.array([[[0, 1000, 4095]]], dtype=np.uint16))
    assert g.dtype == np.uint16 and g.tolist() == [[[0, 700, 2866]]]
