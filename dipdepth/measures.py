"""Per-night measures of DipDepth and the 3% rule against technician scoring and belts, and the paired statistics the paper reports."""
import math
import numpy as np
from scipy.stats import rankdata, spearmanr
from .readers import belt_variability

BIN, PER, WINDOW = 10, 60, 600   # 10 s bins; 10-min windows of 60 bins, or 600 s
LO, HI = -15, 40                 # a detection matches a scored event from 15 s before to 40 s after its end
RESAMPLES, MIN_CLASS = 10000, 5
SLEEP = ("N1", "N2", "N3", "R")  # stage labels that count as sleep


# per-night measures
def matched(times, ends):
    """the largest one-to-one matching of sorted detection times and scored event ends within the window; the number matched"""
    times, ends = np.sort(times), np.sort(ends); used = np.zeros(len(ends), bool); i = j = 0
    while i < len(times) and j < len(ends):
        if times[i] < ends[j] + LO: i += 1
        elif times[i] > ends[j] + HI: j += 1
        else: used[j] = True; i += 1; j += 1
    return int(used.sum())


def f1(p, r): return 0.0 if r == 0 else 2 * p * r / (p + r)


def events_f1(times, ends):
    """events F1 = 2m/(D+S) of detection times against scored event ends; 0 when either is empty"""
    if not len(times) or not len(ends): return 0.0
    k = matched(np.asarray(times, float), ends); return f1(k / len(times), k / len(ends))


def rho(a, b):
    ok = np.isfinite(a) & np.isfinite(b); a, b = a[ok], b[ok]
    return spearmanr(a, b).statistic if len(a) >= 3 and np.std(a) > 0 and np.std(b) > 0 else np.nan


def windows(K): return [(i, i + PER) for i in range(0, K - PER + 1, PER)]


def count_rho(times, ends, K):
    """count per 10 minutes: rank correlation across the night's whole 10-min windows (K 10 s bins) of detections and scored event ends"""
    W = windows(K)
    if len(W) < 3: return np.nan
    e = [a * BIN for a, _ in W] + [W[-1][1] * BIN]
    return float(rho(np.histogram(times, e)[0].astype(float), np.histogram(ends, e)[0].astype(float)))


def drop_rho(sev, ends):
    """the same correlation for the expected drop per 10 s bin, averaged over each window"""
    W = windows(len(sev))
    if len(W) < 3: return np.nan
    e = [a * BIN for a, _ in W] + [W[-1][1] * BIN]
    return float(rho(np.array([sev[a:b].mean() for a, b in W]), np.histogram(ends, e)[0].astype(float)))


def rule_drops(x, t):
    """each 3%-rule dip's fall: highest SpO2 from 60 s before (not before the previous dip) minus the dip"""
    prev = -10 ** 9; out = []
    for a in t.astype(int): lo = max(a - 60, prev, 0); out.append(np.nanmax(x[lo:a]) - x[a]); prev = a
    return np.array(out)


def sleep_hours(stages, n):
    """hours scored as sleep from (onset s, label) stages in time order over n seconds"""
    asleep = np.zeros(n, bool)
    for i, (t, lab) in enumerate(stages): asleep[int(t):int(stages[i + 1][0] if i + 1 < len(stages) else n)] = lab in SLEEP
    return np.sum(asleep) / 3600


def hypoxic_burden(x, ends, sleep_h):
    """hypoxic burden, %min per hour of sleep: one window per night from the ensemble average around scored ends, then the area below each event's highest SpO2 in the 100 s before its end
    (where two events' windows overlap, the later event's baseline applies)"""
    segs = [x[int(e) - 120:int(e) + 120] - np.max(x[int(e) - 100:int(e)]) for e in ends if int(e) >= 120 and int(e) + 120 <= len(x)]
    if len(segs) < 3 or sleep_h < 1: return np.nan
    A = np.mean(segs, axis=0); tt = np.arange(-120, 120); m = np.argmin(A[120:180]) + 120
    ws = tt[np.argmax(A[60:m]) + 60] if m > 60 else -60; we = tt[m + np.argmax(A[m:m + 60])] if m + 1 < len(A) else 60
    mask = np.zeros(len(x), bool); base = np.zeros(len(x))
    for e in ends:
        e = int(e)
        if e < 100 or e + we >= len(x): continue
        a, b = max(0, e + ws), e + we; mask[a:b] = True; base[a:b] = np.max(x[e - 100:e])
    return np.sum(np.clip(base[mask] - x[mask], 0, None)) / 60 / sleep_h


def effort_windows(x1, envelopes):
    """10-min windows from the start as (start s, stop s, SpO2 valid in 90% of it, belt variability or None), from readers.effort's
    envelopes over the seconds both signals cover; a night with one belt uses it for both"""
    two = envelopes if len(envelopes) == 2 else np.vstack([envelopes, envelopes])
    belt = [w["value"] for w in belt_variability(two, min(len(x1), two.shape[1]))]
    return [(i * WINDOW, (i + 1) * WINDOW, bool(np.isfinite(x1[i * WINDOW:(i + 1) * WINDOW]).mean() >= .9), v) for i, v in enumerate(belt)]


def correlation(x, y):
    """rank correlation, 0 for a constant reading; refuses missing values and a constant target"""
    assert len(x) == len(y) > 0 and all(v is not None and math.isfinite(v) for v in (*x, *y)), "invalid rank values"
    a, b = rankdata(x), rankdata(y)
    if np.ptp(a) == 0: return 0.
    assert np.ptp(b) > 0, "constant reference target"
    a, b = a - a.mean(), b - b.mean(); value = float(np.dot(a, b) / math.sqrt(float(np.dot(a, a) * np.dot(b, b))))
    assert abs(value) <= 1 + 32 * np.finfo(float).eps, "rho outside mathematical support"
    return max(-1., min(1., value))


def belt_rho(reading, wins):
    """breathing effort: a reading's rank correlation with belt variability over the windows with valid SpO2 and belts, if 12 or more and the belt varies"""
    keep = [(a, b, t) for a, b, ok, t in wins if ok and t is not None]
    if len(keep) < 12 or len({t for *_, t in keep}) < 2: return np.nan
    return correlation([reading(a, b) for a, b, _ in keep], [t for *_, t in keep])


def night(x1, sev, count_events, rule, converged, ends, stages=None, wins=None):
    """one night's measures. x1: SpO2 per second (NaN missing); sev: expected drop per 10 s bin; count_events: (low point s, fall) rows;
    rule: the 3% rule's dip times; ends: scored respiratory event ends (s); stages: (onset s, label) or None; wins: effort windows or None"""
    x1 = np.asarray(x1, float); n = len(x1); hours = n / 3600; x = np.where(np.isnan(x1), np.nanmedian(x1), x1); K = len(sev)
    C = np.asarray(count_events, float).reshape(-1, 2); rule = np.sort(np.asarray(rule, float)); ends = np.sort(np.asarray(ends, float))
    drops = rule_drops(x, rule) if len(rule) else np.zeros(0)
    out = {"scored_events": len(ends), "count_per_hour": len(C) / hours, "rule_per_hour": len(rule) / hours,
           "expected_drop_mean": float(np.mean(sev)) if converged else np.nan,   # an unconverged state gives no state value
           "count_events_f1": events_f1(np.sort(C[:, 0]), ends), "rule_events_f1": events_f1(rule, ends),
           "count_per_10min_rho": count_rho(C[:, 0], ends, K), "rule_per_10min_rho": count_rho(rule, ends, K),
           "expected_drop_per_10min_rho": drop_rho(sev, ends) if converged else np.nan,
           "hypoxic_burden": hypoxic_burden(x, ends, sleep_hours(stages, n)) if stages is not None else np.nan,
           "rule_summed_dips_per_hour": float(np.sum(drops) / hours)}
    if wins is not None:
        out["effort_expected_drop"] = belt_rho(lambda s, e: float(sev[s // 10:e // 10].mean()), wins) if converged else np.nan
        out["effort_count"] = belt_rho(lambda s, e: float(((rule >= s) & (rule < e)).sum()), wins)
    return out


# statistics over nights
def boot(fn, n, seed=0):
    """95% interval of fn over RESAMPLES resamples of n nights; the draws are row positions, so row order fixes the interval"""
    rng = np.random.default_rng(seed); v = np.array([fn(rng.integers(0, n, n)) for _ in range(RESAMPLES)])
    return [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))]


def auc(s, y):
    r = rankdata(s); p = y.sum(); q = len(y) - p; return float((r[y].sum() - p * (p + 1) / 2) / (p * q)) if p and q else np.nan


def pcor(a, ref, z):
    """rank correlation of a and ref, each adjusted linearly for z's ranks; undefined when a residual has no variance"""
    ra, rb, rz = rankdata(a), rankdata(ref), rankdata(z)
    if np.ptp(rz) == 0: return np.nan
    e1 = ra - np.polyval(np.polyfit(rz, ra, 1), rz); e2 = rb - np.polyval(np.polyfit(rz, rb, 1), rz); tol = 1e-9 * len(a)
    return float(np.corrcoef(e1, e2)[0, 1]) if np.ptp(ra) > 0 and np.ptp(rb) > 0 and np.std(e1) > tol and np.std(e2) > tol else np.nan


def verdict(v):
    """ahead when the interval lies above 0, behind when below, level otherwise"""
    ci = v.get("ci", [np.nan, np.nan])
    v["verdict"] = "not estimable" if not np.all(np.isfinite(ci)) else "ahead" if ci[0] > 0 else "behind" if ci[1] < 0 else "level"; return v


def paired(a, b):
    """per-night measure: medians of a and b and the median paired difference over nights where both are defined"""
    ok = np.isfinite(a) & np.isfinite(b); dd = a[ok] - b[ok]
    if not len(dd): return verdict({"n": 0})
    return verdict({"n": int(ok.sum()), "a": float(np.median(a[ok])), "b": float(np.median(b[ok])), "diff": float(np.median(dd)),
                    "ci": boot(lambda i: np.median(dd[i]), len(dd)), "higher": int((dd > 0).sum()), "lower": int((dd < 0).sum())})


def screening(a, b, ahi, thr):
    """screening AUC at an AHI threshold of a and b (detections per hour) and their difference"""
    ok = np.isfinite(a) & np.isfinite(b) & np.isfinite(ahi); a, b, y = a[ok], b[ok], ahi[ok] >= thr; p = int(y.sum())
    out = {"n": int(ok.sum()), "positive": p, "negative": int(ok.sum()) - p}
    if min(p, len(y) - p) < MIN_CLASS: return verdict(out)   # fewer than 5 people in a class: not estimable
    return verdict({**out, "a": auc(a, y), "b": auc(b, y), "diff": auc(a, y) - auc(b, y), "ci": boot(lambda i: auc(a[i], y[i]) - auc(b[i], y[i]), len(y))})


def oxygen_loss(a, b, burden, ahi):
    """oxygen loss: rank-partial correlation of each night's mean reading with its hypoxic burden given the AHI, for a and b, and their difference"""
    k = np.isfinite(burden) & np.isfinite(a) & np.isfinite(ahi) & np.isfinite(b); n = int(k.sum())
    if n < 10: return verdict({"n": n})
    P = lambda v, i: pcor(v[k][i], burden[k][i], ahi[k][i]); every = np.arange(n)
    return verdict({"n": n, "a": P(a, every), "b": P(b, every), "diff": P(a, every) - P(b, every), "ci": boot(lambda i: P(a, i) - P(b, i), n)})
