"""DipDepth on one night from the raw data: python3 run_night.py KEY [--out FILE.npz]
KEY is DATASET:RECORD (c2018:tr03-0092, psga:00000995-100507, ucd:ucddb002, apsaa:20210919A, isleeps:SN1, osasud:1); data from $DIPDEPTH_DATA."""
import argparse, sys
import numpy as np
from dipdepth import model, readers


def fit(key, x1, keep_failed=False):
    """model.model on one night and what it fits on the way: step 1's pulses, the drop shape (fall s, recovery s) and the expected-drop
    state's EM step variance (None when that state has no EM step); model.model returns none of these, so recorders wrap its functions"""
    rec, orig = {"q": {}}, (model.step1, model.response, model.em_refine)
    def step1(x): rec["step1"] = out = orig[0](x); return out
    def response(fall, recovery, *a, **k): rec["shape"] = (float(fall), float(recovery)); return orig[1](fall, recovery, *a, **k)
    def em_refine(*a, **k): out = orig[2](*a, **k); rec["q"][a[4]] = out["q"]; return out   # a[4] is gamma: 1 for the expected-drop state
    model.step1, model.response, model.em_refine = step1, response, em_refine
    try: row, arrays = model.model(key, x1, keep_failed)
    finally: model.step1, model.response, model.em_refine = orig
    return row, arrays, rec["step1"][0], rec["shape"], rec["q"].get(1.)


def main():
    p = argparse.ArgumentParser(description="DipDepth on one night's SpO2")
    p.add_argument("key", help="DATASET:RECORD, e.g. c2018:tr03-0092")
    p.add_argument("--out", help="save the night's arrays to this .npz file")
    a = p.parse_args()
    try: readers.split_key(a.key)
    except ValueError as e: p.error(str(e))
    try: x1 = readers.spo2(a.key).astype(float)
    except (OSError, ValueError) as e: sys.exit(f"{a.key}: cannot read the night: {e}")
    if not np.isfinite(x1).any(): sys.exit(a.key + ": no valid SpO2")
    row, arrays, pulses, shape, _ = fit(a.key, x1)
    at = np.flatnonzero(pulses > 1e-3); drop = arrays["depth_severity"]
    print(f"{a.key}: {row['seconds']} s of SpO2, {np.isfinite(x1).mean():.1%} valid, oximeter step {row['step']:g} point ({'fine' if row['fine'] else 'coarse'})")
    print(f"drop shape: fall {shape[0]:.1f} s, recovery {shape[1]:.1f} s")
    print(f"pulses: {len(at)}, summed size {pulses[at].sum():.1f} points")
    print(f"runs of 1 point or more: {len(arrays['events'])}, mean own drop {arrays['events'][:, 1].mean() if len(arrays['events']) else 0:.2f} points")
    print(f"counted events: {len(arrays['count_events'])} ({len(arrays['count_events']) / (row['seconds'] / 3600):.1f} per hour)")
    print(f"mean expected drop: {drop.mean():.4f} points per 10 s{'' if row['converged'] else ' (state fit not converged)'}")
    print(f"3% rule dips: {len(arrays['rule3_nadirs'])}")
    if a.out: np.savez(a.out, pulses=pulses, shape=np.array(shape), **arrays)


if __name__ == "__main__":
    main()
