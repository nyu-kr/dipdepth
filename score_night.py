"""The published per-night values of DipDepth from the raw data: python3 score_night.py KEY [KEY ...]
Prints CSV, one block per kind of night, each row with the columns of its results file followed by results/parameters.csv's:
results/sealed.csv for a scored Boston, Athens or Dublin night (ahi is results/sealed.csv's for its nights, else scored respiratory
events per hour of sleep to one decimal), results/unscored.csv for a Boston night without public scoring (te*), results/new_hospitals.csv
for Jaén, Bengaluru and Udine; the row-order columns ablation_order and same_recording_as are left out. Data from $DIPDEPTH_DATA; run
with OMP_NUM_THREADS=1 to reproduce the published values."""
import argparse, csv, math, sys
from pathlib import Path
import numpy as np
from dipdepth import measures, readers, readings
from run_night import fit

RESULTS = Path(__file__).resolve().parent / "results"
HOSPITAL = {"c2018": "Boston", "psga": "Athens", "ucd": "Dublin", "apsaa": "Jaén", "isleeps": "Bengaluru", "osasud": "Udine"}
EFFORT = {"expected drop": "effort_expected_drop", "pulse depths": "effort_pulse_depths", "area below level": "effort_area_below_level",
          "summed dips": "effort_summed_dips", "count": "effort_count"}
PARAMETERS = ["fall_s", "recovery_s", "pulses_per_hour", "runs_per_hour", "state_step_sd"]


def kind(key):
    site, rec = readers.split_key(key)
    return "new_hospitals.csv" if site in readers.KEEP else "unscored.csv" if site == "c2018" and rec.startswith("te") else "sealed.csv"


def columns(name):
    with open(RESULTS / name) as f: return [c for c in next(csv.reader(f)) if c not in ("ablation_order", "same_recording_as")] + PARAMETERS


def score(key, ahi):
    """one night's row as a dict; excluded nights carry only their reason"""
    site, rec = readers.split_key(key); k = kind(key); out = {"key": key, "hospital": HOSPITAL[site]}
    x1 = readers.spo2(key).astype(float)
    if not np.isfinite(x1).any(): return {**out, "excluded": "no valid SpO2"}
    try: row, a, pulses, shape, q = fit(key, x1, keep_failed=k == "sealed.csv")   # the scored nights' results keep a failed state fit as unconverged; the other tests exclude the night
    except Exception as e:
        print(f"{key}: {type(e).__name__}: {e}", file=sys.stderr); return {**out, "excluded": "model fit failed" if k == "new_hospitals.csv" else "prediction failed"}
    n = len(x1); hours = n / 3600; conv = row["converged"]; sev = a["depth_severity"]; rule = np.sort(a["rule3_nadirs"])
    out.update(converged=conv, fall_s=shape[0], recovery_s=shape[1], pulses_per_hour=int((pulses > 1e-3).sum()) / hours,
               runs_per_hour=len(a["events"]) / hours, state_step_sd=math.sqrt(q) if conv and q is not None else math.nan)
    try: wins = measures.effort_windows(x1, readers.effort(key))
    except Exception as e: print(f"{key}: belt: {type(e).__name__}: {e}", file=sys.stderr); wins = None
    effort = readings.effort(x1, sev, a["events"], rule, conv, wins) if wins is not None else {}
    out.update({c: effort.get(name, math.nan) for name, c in EFFORT.items()})
    if k == "sealed.csv" and not conv:   # the scored nights' ablation left out the nights whose state fit failed
        out.update(effort_pulse_depths=math.nan, effort_area_below_level=math.nan, effort_summed_dips=math.nan)
    if k != "sealed.csv" and not np.isfinite(out["effort_count"]): out["excluded"] = "belt not eligible"
    if k == "unscored.csv": return out
    events, stages = readers.references(key)
    ends = [e[1] for e in events if site != "c2018" or e[2].startswith("resp_")]   # Boston's annotation also holds arousals
    out.update(measures.night(x1, sev, a["count_events"], rule, conv, ends, stages if k == "sealed.csv" else None))
    if k == "sealed.csv":
        sleep = measures.sleep_hours(stages, n); out["ahi"] = ahi.get(key, round(len(ends) / sleep, 1) if sleep > 0 else math.nan)
    else: out["ahi"] = len(ends) / (np.isfinite(x1).sum() / 3600)   # per hour of valid SpO2: these datasets are read without sleep stages
    return out


def text(v):
    if v is None or (isinstance(v, float) and math.isnan(v)): return ""
    if isinstance(v, (bool, np.bool_, str)): return str(v)
    return str(int(v)) if isinstance(v, (int, np.integer)) else repr(float(v))


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0]); p.add_argument("keys", nargs="+", metavar="KEY"); args = p.parse_args()
    for key in args.keys:
        try: readers.split_key(key)
        except ValueError as e: p.error(str(e))
    with open(RESULTS / "sealed.csv") as f: ahi = {r["key"]: r["ahi"] for r in csv.DictReader(f)}
    blocks, failed = {}, 0
    for key in args.keys:
        try: r = score(key, ahi)
        except Exception as e: print(f"{key}: unreadable: {type(e).__name__}: {e}", file=sys.stderr); failed += 1; continue
        if kind(key) == "sealed.csv" and "excluded" in r:   # every scored night in the results has a row: a failure here is an error, not an exclusion
            print(f"{key}: {r['excluded']}", file=sys.stderr); failed += 1; continue
        blocks.setdefault(kind(key), []).append(r)
    w = csv.writer(sys.stdout, lineterminator="\n")
    for i, (name, rows) in enumerate(blocks.items()):
        if i: print()
        cols = columns(name); w.writerow(cols); w.writerows([text(r.get(c)) for c in cols] for r in rows)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
