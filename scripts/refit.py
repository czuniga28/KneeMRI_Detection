import argparse
import json
from pathlib import Path

import yaml

from src.train.loop import PROJECT_ROOT, check_refit_config, refit, write_provenance

OUT_ROOT = PROJECT_ROOT / 'runs' / 'refit'
# Separate root: a smoke run in runs/refit/s42 would make the real run skip seed 42
SMOKE_OUT_ROOT = PROJECT_ROOT / 'runs' / 'refit_smoke'


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 4 refit on the whole development set, one run per seed.")
    parser.add_argument('--config', type=Path, default=PROJECT_ROOT / 'configs' / 'train' / 'm1_refit.yaml')
    parser.add_argument('--seeds', nargs='+', type=int, default=[42, 0], help="Default: the CV seeds")
    parser.add_argument('--device', help="cpu or mps; default picks MPS when available")
    parser.add_argument('--max-slices', type=int, help="Smoke test on this many training slices")
    parser.add_argument('--out-root', type=Path, help=f"Default: {OUT_ROOT}, or {SMOKE_OUT_ROOT} with --max-slices")
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    # Once, before any seed trains
    check_refit_config(cfg)
    out_root = args.out_root or (OUT_ROOT if args.max_slices is None else SMOKE_OUT_ROOT)

    results = {}
    for seed in args.seeds:
        cfg_s = {**cfg, 'seed': seed}
        out = out_root / f"s{seed}"
        done = out / 'metrics.json'
        if done.exists():
            print(f"=== s{seed}: already done, skipping ({done})", flush=True)
            results[seed] = json.loads(done.read_text())
            continue
        print(f"=== s{seed}: refit for {cfg['refit_epochs']} epochs -> {out}", flush=True)
        write_provenance(out, cfg_s)
        results[seed] = refit(cfg_s, out, max_slices=args.max_slices, device=args.device)

    for seed, m in results.items():
        print(f"s{seed}: epochs_trained {m['epochs_trained']}  final_lr {m['final_lr']:.3g}  "
              f"seconds_per_epoch {m['seconds_per_epoch']:.1f}")
    print(f"Saved to {out_root}")


if __name__ == '__main__':
    main()
