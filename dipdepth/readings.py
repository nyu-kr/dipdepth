"""Five per-window readings of one night's SpO2, compared by their breathing-effort correlation in Table II and the ablation."""
import numpy as np
from scipy.ndimage import percentile_filter
from .measures import belt_rho, rule_drops

NAMES = ("expected drop", "pulse depths", "area below level", "summed dips", "count")


def readings(x1, sev, runs, rule):
    """per-window value (start s, stop s) of each reading: the state's expected drop, the summed depths of the runs of pulses the state
    is fitted to, the summed drop below the starting usual level, the 3% rule's dips each by its fall, and the rule's count"""
    x = np.where(np.isnan(x1), np.nanmedian(x1), x1); d = np.clip(percentile_filter(x, 90, size=180) - x, 0, None)
    rule = np.asarray(rule, float); drops = rule_drops(x, rule) if len(rule) else np.zeros(0); t, dep = np.asarray(runs, float).reshape(-1, 2).T
    return {"expected drop": lambda s, e: float(sev[s // 10:e // 10].mean()), "pulse depths": lambda s, e: float(dep[(t >= s) & (t < e)].sum()),
            "area below level": lambda s, e: float(d[s:e].sum()), "summed dips": lambda s, e: float(drops[(rule >= s) & (rule < e)].sum()),
            "count": lambda s, e: float(((rule >= s) & (rule < e)).sum())}


def effort(x1, sev, runs, rule, converged, wins):
    """each reading's breathing-effort correlation on the night's effort windows; the expected drop needs a converged state"""
    R = readings(x1, sev, runs, rule)
    return {name: belt_rho(R[name], wins) if (converged or name != "expected drop") else np.nan for name in NAMES}
