#!/usr/bin/env python3
"""Rebuilds the paper's Tables II to IV from results/ alone, with the text's figures drawn from the same rows (nights higher and lower, the gain
by hospital, the ablation, the expected drop's per-window and screening comparisons, the duplicate night), prints them in the paper's rounding,
and exits nonzero if any value differs from results/expected.json (numbers by more than 1e-9, counts and verdicts at all)."""
import json, math, sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from dipdepth.measures import oxygen_loss, paired, screening

RES = Path(__file__).resolve().parent / "results"
READINGS = {"expected drop": "effort_expected_drop", "pulse depths": "effort_pulse_depths", "area below level": "effort_area_below_level",
            "summed dips": "effort_summed_dips", "count": "effort_count"}
PAIRS = (("expected drop", "pulse depths"), ("expected drop", "summed dips"), ("expected drop", "area below level"), ("expected drop", "count"),
         ("pulse depths", "summed dips"), ("pulse depths", "area below level"))
TOL = 1e-9


def load(name): return pd.read_csv(RES / name, encoding="utf-8")


def col(df, c): return df[c].to_numpy(float)


def effort(df):
    """one Table II row and the ablation's pairs, on the nights with all five readings, in file order"""
    d = df[df[list(READINGS.values())].notna().all(axis=1)]; v = {r: col(d, c) for r, c in READINGS.items()}
    return {"n": len(d), "medians": {r: float(np.median(x)) for r, x in v.items()}}, {f"{a} - {b}": paired(v[a], v[b]) for a, b in PAIRS}


def counting(df, oxygen):
    """one Table III block; the per-night measures on nights with scored events"""
    E = df[df["scored_events"] > 0]; ahi = col(df, "ahi")
    out = {"Events F1": paired(col(E, "count_events_f1"), col(E, "rule_events_f1")),
           "Count per 10 min": paired(col(E, "count_per_10min_rho"), col(E, "rule_per_10min_rho")),
           **{f"AUC, AHI {t}": screening(col(df, "count_per_hour"), col(df, "rule_per_hour"), ahi, t) for t in (5, 15)}}
    if oxygen:
        burden, drop = col(df, "hypoxic_burden"), col(df, "expected_drop_mean")
        out["Oxygen loss"] = oxygen_loss(drop, col(df, "rule_per_hour"), burden, ahi)
        out["vs summed dips"] = oxygen_loss(drop, col(df, "rule_summed_dips_per_hour"), burden, ahi)
    return out


def compute():
    sealed, unscored, new = load("sealed.csv"), load("unscored.csv"), load("new_hospitals.csv")
    rows = {"Sealed": (sealed, sealed), **{h: (sealed[sealed.hospital == h],) * 2 for h in ("Boston", "Athens", "Dublin")},
            "Unscored, Boston": (unscored, unscored.sort_values("ablation_order")),   # the ablation resampled these nights in its own order
            "New hospitals": (new, new), **{h: (new[new.hospital == h],) * 2 for h in ("Jaén", "Bengaluru", "Udine")}}
    out = {"Table II": {}, "ablation": {}}
    for name, (df, abl) in rows.items():
        row, out["ablation"][name] = effort(abl)
        out["Table II"][name] = {**row, "expected drop - count": paired(col(df, READINGS["expected drop"]), col(df, READINGS["count"]))}
    E = sealed[sealed["scored_events"] > 0]; ahi = col(sealed, "ahi")
    out["Table III"] = {"Sealed nights": counting(sealed, True), "New hospitals": counting(new, False)}
    out["text"] = {"Sealed, expected drop per 10 min": paired(col(E, "expected_drop_per_10min_rho"), col(E, "rule_per_10min_rho")),
                   **{f"Sealed, expected drop AUC, AHI {t}": screening(col(sealed, "expected_drop_mean"), col(sealed, "rule_per_hour"), ahi, t) for t in (5, 15)},
                   "counts": {"sealed nights readable": len(sealed), "sealed nights with a converged state": int(sealed.converged.sum()),
                              "unscored nights": len(unscored), "unscored predictions failed": int((unscored.excluded == "prediction failed").sum()),
                              "unscored belts not eligible": int((unscored.excluded == "belt not eligible").sum()), "new-hospital recordings": len(new),
                              "new-hospital recordings with readings": int((~new.excluded.isin(["no valid SpO2", "model fit failed"])).sum())}}
    out["Table IV"] = severity(sealed, load("parameters.csv"))
    once, changed, full = {}, {}, new_verdicts(new)   # one copy of the night released twice, the other dropped
    for _, r in new[new.same_recording_as.notna()].iterrows():
        d = new[new.key != r.same_recording_as]; name = f"without {r.same_recording_as}"
        once[name] = paired(col(d, READINGS["expected drop"]), col(d, READINGS["count"]))
        changed[name] = [f"{k}: {full[k]} -> {v}" for k, v in new_verdicts(d).items() if v != full[k]]
    return out, once, changed, len(full)


def new_verdicts(new):
    """every verdict drawn from the new hospitals: Table II's differences, the ablation, Table III's block"""
    out = {f"Table II, {h}": paired(col(d, READINGS["expected drop"]), col(d, READINGS["count"]))["verdict"]
           for h, d in (("New hospitals", new), *((h, new[new.hospital == h]) for h in ("Jaén", "Bengaluru", "Udine")))}
    out.update({f"ablation, {k}": v["verdict"] for k, v in effort(new)[1].items()})
    return {**out, **{f"Table III, {k}": v["verdict"] for k, v in counting(new, False).items()}}


GROUPS = {"under 5": (0, 5), "5 to 15": (5, 15), "15 to 30": (15, 30), "30 or more": (30, math.inf)}
SEVERITY = {"fall T_f (s)": "fall_s", "recovery T_r (s)": "recovery_s", "pulses per hour": "pulses_per_hour", "runs per hour": "runs_per_hour",
            "state step sd": "state_step_sd", "expected drop": "expected_drop_mean", "rule's dips per hour": "rule_per_hour"}


def severity(sealed, params):
    """Table IV: medians by AHI group and the rank correlation with the AHI; the shape rows leave out nights whose shape kept its start (20 s, 20 s)"""
    d = sealed.merge(params, on="key", validate="one_to_one"); ahi = col(d, "ahi")
    moved = ~(np.isclose(col(d, "fall_s"), 20, rtol=0, atol=1e-6) & np.isclose(col(d, "recovery_s"), 20, rtol=0, atol=1e-6))
    out = {"nights": {g: int(((ahi >= a) & (ahi < b)).sum()) for g, (a, b) in GROUPS.items()},
           "shape kept its start": {g: int((~moved & (ahi >= a) & (ahi < b)).sum()) for g, (a, b) in GROUPS.items()}}
    for name, c in SEVERITY.items():
        v = col(d, c); ok = np.isfinite(v) & (moved if c in ("fall_s", "recovery_s") else True)
        out[name] = {**{g: float(np.median(v[ok & (ahi >= a) & (ahi < b)])) for g, (a, b) in GROUPS.items()},
                     "rho": float(spearmanr(v[ok], ahi[ok])[0]), "n": int(ok.sum())}
    return out


def flat(d, path=()):
    for k, v in d.items():
        if isinstance(v, dict): yield from flat(v, path + (k,))
        elif isinstance(v, list): yield from ((path + (k, str(i)), x) for i, x in enumerate(v))
        else: yield path + (k,), v


def differs(want, got):
    if isinstance(want, str) or isinstance(got, str) or isinstance(want, bool): return want != got
    if isinstance(want, int) and isinstance(got, int): return want != got
    w, g = float(want), float(got)
    return not (math.isnan(w) and math.isnan(g)) and not abs(w - g) <= TOL


def check(expected, got):
    """every expected value recomputed and every recomputed value expected; the mismatches as lines"""
    E, G = dict(flat(expected)), dict(flat(got)); bad = []
    for p in sorted(set(E) | set(G)):
        if p not in G: bad.append(f"missing from the rebuild: {' / '.join(p)} (expected {E[p]})")
        elif p not in E: bad.append(f"not in expected.json: {' / '.join(p)} (rebuilt {G[p]})")
        elif differs(E[p], G[p]): bad.append(f"{' / '.join(p)}: expected {E[p]}, rebuilt {G[p]}")
    return bad


def check_rounded(claims, once):
    """the paper's text figures for the duplicate night, which no summary holds: n, the difference to 3 places and the verdict"""
    bad = []
    for name, c in claims.items():
        g = once.get(name, {})
        if (g.get("n"), round(g.get("diff", math.nan), 3), g.get("verdict")) != (c["n"], c["diff"], c["verdict"]): bad.append(f"{name}: paper {c}, rebuilt {g}")
    return bad + [f"not in the paper's claims: {n}" for n in once if n not in claims]


f3, s3 = (lambda v: f"{v:.3f}"), (lambda v: "0.000" if round(v, 3) == 0 else f"{v:+.3f}")   # a difference that rounds to zero prints unsigned, as in the paper
PLACES = {"fall T_f (s)": 1, "recovery T_r (s)": 1, "state step sd": 3, "expected drop": 2, "rule's dips per hour": 1}   # Table IV's decimals as printed
ci = lambda v: f"[{f3(v['ci'][0])}, {f3(v['ci'][1])}]"


def show(out, once, changed, compared):
    T2, A, T3, X = out["Table II"], out["ablation"], out["Table III"], out["text"]
    print("TABLE II  Breathing effort: median per-night rank correlation of each reading with belt variability (* the highest of the five)")
    print(f"{'Nights':<18}{'n':>4}  {'Exp. drop':>9}  {'Pulse d.':>9}  {'Area':>9}  {'Sum dips':>9}  {'Count':>9}   Expected drop minus count [95%]")
    for name, r in T2.items():
        top = max(r["medians"], key=r["medians"].get); d = r["expected drop - count"]
        cells = "  ".join(f"{f3(v) + ('*' if k == top else ' '):>9}" for k, v in r["medians"].items())
        print(f"{('  ' if name in ('Boston', 'Athens', 'Dublin', 'Jaén', 'Bengaluru', 'Udine') else '') + name:<18}{r['n']:>4}  {cells}   {s3(d['diff'])} {ci(d)}")
    print("\nTABLE III  Counting and oxygen measures against the 3% rule (* the higher value where the interval excludes 0)")
    print(f"{'Measure':<20}{'DipDepth':>9}{'Rule':>8}   Difference [95%]   n")
    for block, rows in T3.items():
        print(f"{block} ({X['counts']['sealed nights readable' if block.startswith('Sealed') else 'new-hospital recordings with readings']})")
        for name, v in rows.items():
            a, b = f3(v["a"]) + "*" * (v["verdict"] == "ahead"), f3(v["b"]) + "*" * (v["verdict"] == "behind")
            extra = f"{v['positive']} at or above, {v['negative']} below" if "positive" in v else f"{v['n']} nights"
            print(f"  {name:<18}{a:>9}{b:>8}   {s3(v['diff'])} {ci(v):<18} {extra}")
    print("\nTEXT")
    hl = lambda v: f"{v['higher']} higher, {v['lower']} lower"
    print("Expected drop against the rule's count, nights:", "; ".join(f"{k} {hl(T2[k]['expected drop - count'])}" for k in ("Sealed", "Unscored, Boston", "New hospitals")))
    for name, v in once.items(): print(f"Duplicate Bengaluru night counted once ({name}): {s3(v['diff'])} {ci(v)}, {v['n']} nights, {v['verdict']}; "
                                       f"{compared - len(changed[name])} of {compared} new-hospital verdicts unchanged")
    print("Median gain by hospital:", ", ".join(f"{k} {s3(T2[k]['expected drop - count']['diff'])} {T2[k]['expected drop - count']['verdict']}"
                                               for k in ("Boston", "Athens", "Dublin", "Jaén", "Bengaluru", "Udine")))
    print(f"Sealed events F1, count against rule: {hl(T3['Sealed nights']['Events F1'])}")
    v = X["Sealed, expected drop per 10 min"]; print(f"Sealed expected drop per 10 min: {f3(v['a'])} against {f3(v['b'])} on {v['n']} nights ({v['verdict']})", end="; ")
    print("expected drop screening:", ", ".join(f"AHI {t} {f3(X[f'Sealed, expected drop AUC, AHI {t}']['a'])} against {f3(X[f'Sealed, expected drop AUC, AHI {t}']['b'])} "
                                                f"({X[f'Sealed, expected drop AUC, AHI {t}']['verdict']})" for t in (5, 15)))
    for name in ("Unscored, Boston", "Sealed", "Athens", "New hospitals"):
        print(f"Ablation, {name}:", "; ".join(f"{p} {s3(v['diff'])} {ci(v)} {v['verdict']}" for p, v in A[name].items()))
    print("Counts:", ", ".join(f"{k} {v}" for k, v in X["counts"].items()))
    print("\nTABLE IV  The fitted model by AHI group on the sealed nights (medians; rho: rank correlation with the AHI)")
    print(f"{'AHI':<22}" + "".join(f"{g:>12}" for g in GROUPS))
    for name, r in out["Table IV"].items():
        print(f"{name:<22}" + "".join(f"{r[g]:>12.{PLACES.get(name, 0)}f}" for g in GROUPS) + (f"   rho {r['rho']:+.2f}  n {r['n']}" if "rho" in r else ""))


if __name__ == "__main__":
    expected = json.load(open(RES / "expected.json", encoding="utf-8")); out, once, changed, compared = compute(); show(out, once, changed, compared)
    bad = check(expected["summaries"], out) + check_rounded(expected["paper text, 3 places"], once) + [f"{n}, verdict changed: {c}" for n, v in changed.items() for c in v]
    print("\n" + ("\n".join(["MISMATCH " + b for b in bad]) if bad else f"every value matches results/expected.json ({len(dict(flat(out)))} values to {TOL:g}, "
                                                                       f"{len(once)} text figures to 3 places, {compared} new-hospital verdicts with either copy of the duplicate night)"))
    sys.exit(1 if bad else 0)
