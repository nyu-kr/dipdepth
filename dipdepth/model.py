"""DipDepth on one night's SpO2: step 1 (sparse deconvolution into oxygen-loss pulses), the event count, step 2 (a marked
point process state fitted by variational EM, giving the expected drop per 10 s bin) and the 3% rule.
Operations run in the paper's order, so results match the paper's to floating-point precision on one thread."""
import numpy as np
from scipy.linalg import cholesky_banded, solveh_banded
from scipy.ndimage import median_filter, percentile_filter
from scipy.optimize import minimize_scalar
from scipy.signal import fftconvolve
from scipy.special import log_ndtr, logsumexp

# the paper's settings: level = 90th percentile over 180 s; l1 weight = a tenth of 0.08 x the depth's peak correlation with a 6 s delay, 0.3/s fall, 0.1/s recovery shape
LEVEL, SMOOTH, PENALTY, SHAPE, TENTH, ROUNDS = (90, 180), 60., 0.08, (6, 0.3, 0.1), 0.1, 6


# step 1
def response(fall, recovery, delay=6, length=360):
    t = np.arange(length, dtype=float) - delay; u = np.maximum(t, 0)
    h = np.where(u < fall, .5 * (1 - np.cos(np.pi * u / fall)), np.exp(-np.maximum(u - fall, 0) / recovery))
    return np.where(t >= 0, h, 0)


def forward(q, h): return fftconvolve(q, h)[:len(q)]


def adjoint(r, h): return fftconvolve(r, h[::-1])[len(h) - 1:len(h) - 1 + len(r)]


def weighted_fista(d, h, weights, penalty, iterations=400, initial=None, support=None):
    q = np.zeros(len(d)) if initial is None else initial.copy(); y, momentum = q.copy(), 1.
    lipschitz = weights.max() * np.abs(h).sum() ** 2
    for _ in range(iterations):
        updated = np.maximum(y - (adjoint(weights * (forward(y, h) - d), h) + penalty) / lipschitz, 0)
        if support is not None: updated *= support
        next_momentum = (1 + np.sqrt(1 + 4 * momentum ** 2)) / 2
        y = updated + (momentum - 1) / next_momentum * (updated - q); q, momentum = updated, next_momentum
    return q


def slow_baseline(x, fitted, initial, weights, scale=60., anchor=.02):
    n = len(x); diagonal = np.full(n, 6.); diagonal[[0, -1]] = 1.; diagonal[[1, -2]] = 5.; near = np.full(n - 1, -4.); near[[0, -1]] = -2.
    band = np.zeros((3, n)); band[0] = weights + anchor + scale ** 4 * diagonal; band[1, :-1] = scale ** 4 * near; band[2, :-2] = scale ** 4
    return np.clip(solveh_banded(band, weights * (x + fitted) + anchor * initial, lower=True), 50, 100)


def adam_shape(q, d, weights, parameters, steps=8):
    p = np.log(parameters); first, second = np.zeros(2), np.zeros(2)
    loss = lambda z: .5 * np.mean(weights * (forward(q, response(*np.exp(z))) - d) ** 2) + .001 * np.sum((z - np.log(20)) ** 2)
    before = loss(p)
    for step in range(1, steps + 1):
        gradient = np.empty(2)
        for i in range(2):
            delta = np.zeros(2); delta[i] = .01; gradient[i] = (loss(p + delta) - loss(p - delta)) / .02
        first = .9 * first + .1 * gradient; second = .999 * second + .001 * gradient ** 2
        proposed = np.clip(p - .05 * first / (1 - .9 ** step) / (np.sqrt(second / (1 - .999 ** step)) + 1e-8), np.log([5, 3]), np.log([60, 60]))
        if loss(proposed) <= loss(p): p = proposed
    assert loss(p) <= before + 1e-10
    return np.exp(p)


def kernel(delay, th1, th2, n=240):
    t = np.arange(n) - delay; u = np.clip(t, 0, None); h = (np.exp(-th2 * u) - np.exp(-th1 * u)) * (t >= 0); return h / h.max()


def step1(x):
    """the input pulses u and the response h: noise weights, the level below the slow ceiling, six rounds of input, shape and baseline, the input refit on its support"""
    delta = np.diff(x, prepend=x[0]); noise = np.maximum(.25, 1.4826 * median_filter(np.abs(delta - median_filter(delta, size=301)), size=301))
    w = np.clip((1 / noise ** 2) / np.median(1 / noise ** 2), .05, 20)
    d = np.clip(percentile_filter(x, LEVEL[0], size=LEVEL[1]) - x, 0, None); b0 = x + d
    k = kernel(*SHAPE); lam = PENALTY * np.max(fftconvolve(d, k[::-1])[len(k) - 1:len(k) - 1 + len(d)])
    base, shape, q = b0.copy(), np.array([20., 20.]), np.zeros(len(x)); pen = np.full(len(x), lam * TENTH); h = response(*shape)
    for _ in range(ROUNDS):
        r = base - x; q = weighted_fista(r, h, w, pen, initial=q)
        shape = np.array(adam_shape(q, r, w, shape), float); h = response(*shape)
        base = slow_baseline(x, forward(q, h), b0, w, scale=SMOOTH); pen = lam * TENTH * .5 / (q + .5)
    qf = weighted_fista(base - x, h, w, pen, initial=q)
    return weighted_fista(base - x, h, w, np.zeros(len(x)), iterations=300, initial=qf, support=qf > 1e-3), h


def runs(q, h, gap=5, thr=1.):
    """step 2's input: runs of pulses (gaps up to gap s) whose own drop is thr points or more, as (last pulse, own drop)"""
    nz = np.flatnonzero(q > 1e-3); out = []
    for g in (np.split(nz, np.flatnonzero(np.diff(nz) > gap) + 1) if len(nz) else []):
        u = np.zeros(g[-1] - g[0] + 1); u[g - g[0]] = q[g]; dep = np.convolve(u, h).max()
        if dep >= thr: out.append((g[-1], dep))
    return np.array(out).reshape(-1, 2)


# the event count
def candidates(x, u, h):
    """per run of pulses (gaps up to 5 s): its own drop and the observed low point 8 to 50 s after its last pulse"""
    n = len(x); nz = np.flatnonzero(u > 1e-3); out = []
    for g in np.split(nz, np.flatnonzero(np.diff(nz) > 5) + 1):
        te = g[-1]; own = np.convolve(np.where(u[g[0]:te + 1] > 1e-3, u[g[0]:te + 1], 0.), h).max()
        out.append((own, min(te + 8 + int(np.argmin(x[te + 8:min(te + 51, n)])), n - 1) if te + 9 < n else n - 1))
    return np.array(out, float)


def count(x, x1, u, h):
    """the events (low point, fall over the 60 s before) read per oximeter step, and the step (the smallest change above 0.01 point): under
    0.2, candidates of 1 point or more merged unless the oxygen recovers 1 point between low points, cut 2.5; else all, recovery 2, cut 3"""
    dx = np.abs(np.diff(x1[np.isfinite(x1)])); dx = dx[dx > .01]; step = float(dx.min()) if len(dx) else np.nan; fine = step < 0.2
    if not np.any(u > 1e-3): return np.zeros((0, 2)), step
    C = candidates(x, u, h); C = C[C[:, 0] >= (1. if fine else 0.)]
    if not len(C): return np.zeros((0, 2)), step
    C = C[np.argsort(C[:, 1], kind="stable")]; t = C[:, 1].astype(int); clusters, cur = [], [0]
    for k in range(1, len(C)):
        a, b = t[cur[-1]], t[k]
        if b == a or x[a:b + 1].max() - x[a] < (1. if fine else 2.): cur.append(k)
        else: clusters.append(cur); cur = [k]
    clusters.append(cur); E = []
    for cl in clusters:
        tn = int(C[min(cl, key=lambda j: x[int(C[j, 1])]), 1]); E.append((tn, x[max(0, tn - 60):tn + 1].max() - x[tn]))
    E = np.array(E, float); E = E[E[:, 1] >= (2.5 if fine else 3.)]
    return E[np.argsort(E[:, 0], kind="stable")], step


def odi(x, thr):
    """the 3% rule's nadirs: a local low point (5 s each side), 15 s after the last, thr points below the highest SpO2 of the 60 s before (not before the last nadir)"""
    out = []; prev = -10 ** 9
    for t in range(5, len(x) - 5):
        w = x[t - 5:t + 6]
        if np.isnan(x[t]) or x[t] > np.nanmin(w) or (out and t - out[-1] < 15): continue
        a = max(t - 60, prev, 0); pk = np.nanmax(x[a:t]) if t > a else np.nan
        if pk - x[t] >= thr: out.append(t); prev = t
    return np.array(out)


# step 2
def bins(events, size, width=10):
    counts, marks, squares = np.zeros(size), np.zeros(size), np.zeros(size)
    for time, depth in events:
        k = int(time) // width
        if 0 <= k < size: value = np.log(depth); counts[k] += 1; marks[k] += value; squares[k] += value ** 2
    return counts, marks, squares


def prior_product(x, variance):
    out = np.zeros(len(x)); out[0] += x[0] / (2 * variance); difference = np.diff(x) / variance; out[:-1] -= difference; out[1:] += difference
    return out


def precision(data_info, variance):
    diagonal = data_info + 2 / variance; diagonal[0] -= .5 / variance; diagonal[-1] -= 1 / variance
    band = np.zeros((2, len(diagonal))); band[0] = diagonal; band[1, :-1] = -1 / variance; return band


def posterior_covariance(band):
    n = band.shape[1]; pivots = np.empty(n); multipliers = np.zeros(n); pivots[0] = band[0, 0]
    for k in range(1, n):
        multipliers[k] = band[1, k - 1] / pivots[k - 1]; pivots[k] = band[0, k] - multipliers[k] ** 2 * pivots[k - 1]
    variance, covariance = np.empty(n), np.empty(n - 1); variance[-1] = 1 / pivots[-1]
    for k in range(n - 2, -1, -1):
        variance[k] = 1 / pivots[k] + multipliers[k + 1] ** 2 * variance[k + 1]; covariance[k] = -multipliers[k + 1] * variance[k + 1]
    return variance, covariance


NODES, WEIGHTS = np.polynomial.hermite.hermgauss(64)
WEIGHTS /= np.sqrt(np.pi)


def mark_terms(m, v, n, total, squares, alpha, gamma, sigma, floor):
    gradient, information = np.zeros(len(m)), np.zeros(len(m)); use = n > 0
    if not gamma or not use.any(): return gradient, information, 0.
    samples = m[use, None] + np.sqrt(2 * v[use, None]) * NODES; mu = alpha + gamma * samples; z = (floor - mu) / sigma
    mills = np.exp(-.5 * z ** 2 - .5 * np.log(2 * np.pi) - log_ndtr(-z)); residual = total[use, None] - n[use, None] * mu
    gradient[use] = ((residual / sigma ** 2 - n[use, None] * mills / sigma) @ WEIGHTS)
    information[use] = (n[use, None] / sigma ** 2 * (1 - mills * (mills - z))) @ WEIGHTS
    log_density = -.5 * (squares[use, None] - 2 * mu * total[use, None] + n[use, None] * mu ** 2) / sigma ** 2 - n[use, None] * log_ndtr(-z)
    return gradient, information, float(np.sum(log_density @ WEIGHTS))


def energy(m, v, c): return float((m[0] ** 2 + v[0]) / 2 + np.sum(np.diff(m) ** 2 + v[1:] + v[:-1] - 2 * c))


def gaussian(info, q):
    band = precision(info, q); v, c = posterior_covariance(band); return v, c, float(2 * np.log(cholesky_banded(band, lower=True)[0]).sum())


def bound(m, info, beta, alpha, q, n, total, squares, gamma, sigma, floor, exposure):
    v, c, ld = gaussian(info, q); mark = mark_terms(m, v, n, total, squares, alpha, gamma, sigma, floor)[2]
    return float(n @ (beta + m) - (exposure * np.exp(beta + m + v / 2)).sum() + mark - energy(m, v, c) / (2 * q) - len(m) * np.log(q) / 2 - ld / 2)


def fit_at_q(n, total, squares, gamma, sigma, floor, q, initial, exposure, parameters_fixed=False):
    """the Gaussian posterior of the state at process variance q: Newton steps on the mean (and the intercepts unless fixed), then the precision"""
    if initial is None: m, info, beta, alpha = np.zeros(len(n)), np.full(len(n), n.mean()), float(np.log(n.mean())), float(total.sum() / n.sum())
    else: m, info, beta, alpha = initial; m, info = m.copy(), info.copy()
    history = []
    for iteration in range(150):
        v, c, ld = gaussian(info, q)
        def objective(mm, bb, aa):
            mark = mark_terms(mm, v, n, total, squares, aa, gamma, sigma, floor)[2]
            return float(n @ (bb + mm) - (exposure * np.exp(bb + mm + v / 2)).sum() + mark - energy(mm, v, c) / (2 * q) - len(n) * np.log(q) / 2 - ld / 2)
        for inner in range(30):
            rate = exposure * np.exp(beta + m + v / 2); mg, mi, _ = mark_terms(m, v, n, total, squares, alpha, gamma, sigma, floor)
            g = n - rate + gamma * mg - prior_product(m, q)
            intercept_g = np.zeros(1) if parameters_fixed else np.array([n.sum() - rate.sum(), mg.sum()]) if gamma else np.array([n.sum() - rate.sum()])
            if max(np.max(np.abs(g)), np.max(np.abs(intercept_g))) < 1e-8: break
            band = precision(rate + gamma ** 2 * mi, q)
            if parameters_fixed: step_m, step_intercept = solveh_banded(band, g, lower=True), np.zeros(2)
            else:
                cross = np.c_[rate, gamma * mi] if gamma else rate[:, None]; diagonal = np.diag([rate.sum(), mi.sum()]) if gamma else np.array([[rate.sum()]])
                solved = solveh_banded(band, np.c_[g, cross], lower=True)
                step_intercept = np.linalg.solve(diagonal - cross.T @ solved[:, 1:], intercept_g - cross.T @ solved[:, 0]); step_m = solved[:, 0] - solved[:, 1:] @ step_intercept
            before = objective(m, beta, alpha)
            for backtrack in range(35):
                fraction = 2. ** -backtrack; mm = m + fraction * step_m; bb = beta + fraction * step_intercept[0]
                aa = alpha + fraction * step_intercept[1] if gamma else alpha; after = objective(mm, bb, aa)
                if np.isfinite(after) and after >= before - 1e-9: m, beta, alpha = mm, float(bb), float(aa); break
            else: raise AssertionError("joint mean/intercept search failed")
        rate = exposure * np.exp(beta + m + v / 2); mg, mi, _ = mark_terms(m, v, n, total, squares, alpha, gamma, sigma, floor)
        desired = rate + gamma ** 2 * mi; before = objective(m, beta, alpha)
        for backtrack in range(35):
            proposed = info + (desired - info) * 2. ** -backtrack; after = bound(m, proposed, beta, alpha, q, n, total, squares, gamma, sigma, floor, exposure)
            if np.isfinite(after) and after >= before - 1e-9: info = proposed; break
        else: raise AssertionError("covariance search failed")
        v, c, ld = gaussian(info, q); rate = exposure * np.exp(beta + m + v / 2); mg, mi, _ = mark_terms(m, v, n, total, squares, alpha, gamma, sigma, floor)
        residuals = [float(np.max(np.abs(n - rate + gamma * mg - prior_product(m, q)))), float(abs(n.sum() - rate.sum())) / n.sum(),
                     float(abs(mg.sum())) / n.sum() if gamma else 0., float(np.max(np.abs(info - rate - gamma ** 2 * mi)))]
        residual = max(residuals[0], residuals[-1]) if parameters_fixed else max(residuals); history.append(after)
        if residual < 1e-6: return {"m": m, "info": info, "beta": beta, "alpha": alpha, "q": q, "v": v, "c": c, "bound": after, "residual": residual}
    raise AssertionError(f"Gaussian optimization failed: q={q}, residual={residuals}, last_bounds={history[-5:]}")


def variational_fit(n, total, squares, gamma, sigma, floor, limit, exposure):
    """the best posterior over the process variance profiled on [1e-6, 2], each fit warm-started from the best so far"""
    best = None
    def evaluate(logq):
        nonlocal best
        initial = None if best is None else (best["m"], best["info"], best["beta"], best["alpha"])
        try: result = fit_at_q(n, total, squares, gamma, sigma, floor, float(np.exp(logq)), initial, exposure)
        except AssertionError as exc:
            if initial is None or not str(exc).startswith("Gaussian optimization failed:"): raise
            result = fit_at_q(n, total, squares, gamma, sigma, floor, float(np.exp(logq)), None, exposure)   # a failed warm start is retried from zero
        if best is None or result["bound"] > best["bound"]: best = result
        return -result["bound"]
    evaluate(np.log(.05)); opt = minimize_scalar(evaluate, bounds=(np.log(1e-6), np.log(2)), method="bounded", options={"xatol": 1e-4, "maxiter": limit})
    evaluate(np.log(1e-6)); evaluate(np.log(2)); best["success"] = bool(opt.success); return best


def em_refine(result, n, total, squares, gamma, sigma, floor, exposure, limit, tolerance):
    """EM from the profiled fit: the posterior at fixed parameters, then the intercepts, the mark offset and the process variance in closed form or by line search"""
    for _ in range(limit):
        beta, alpha, q = result["beta"], result["alpha"], result["q"]
        result = fit_at_q(n, total, squares, gamma, sigma, floor, q, (result["m"], result["info"], beta, alpha), exposure, parameters_fixed=True)
        m, v, c = result["m"], result["v"], result["c"]
        def fixed_bound(bb, aa, qq):
            return float(n @ (bb + m) - (exposure * np.exp(bb + m + v / 2)).sum() + mark_terms(m, v, n, total, squares, aa, gamma, sigma, floor)[2] - energy(m, v, c) / (2 * qq) - len(n) * np.log(qq) / 2)
        new_beta, new_alpha = float(np.log(n.sum()) - logsumexp(np.log(exposure) + m + v / 2)), alpha
        if gamma:
            objective = lambda aa: -mark_terms(m, v, n, total, squares, aa, gamma, sigma, floor)[2]
            update = minimize_scalar(objective, bracket=(alpha - .1, alpha + .1), options={"xtol": 1e-10}); assert update.success, "mark M-step failed"
            if update.fun <= objective(alpha): new_alpha = float(update.x)
        new_q = float(np.clip(energy(m, v, c) / len(n), 1e-6, 2))
        assert fixed_bound(new_beta, new_alpha, new_q) - fixed_bound(beta, alpha, q) >= -1e-8, "fixed-posterior M-step reduced bound"
        delta = float(max(abs(new_beta - beta), abs(new_alpha - alpha), abs(np.log(new_q / q))))
        result = fit_at_q(n, total, squares, gamma, sigma, floor, new_q, (m, result["info"], new_beta, new_alpha), exposure, parameters_fixed=True)
        if delta < tolerance: result["em_converged"] = True; return result
    result["em_converged"] = False; return result


def marked_rate_mean(x, variance, alpha, beta, gamma, sigma, floor):
    nodes, weights = np.polynomial.hermite.hermgauss(32); samples = x[:, None] + np.sqrt(2 * variance[:, None]) * nodes; mu = alpha + gamma * samples
    log_mark = mu + .5 * sigma ** 2 + log_ndtr((mu + sigma ** 2 - floor) / sigma) - log_ndtr((mu - floor) / sigma)
    return np.exp(logsumexp(beta + samples + log_mark + np.log(weights / np.sqrt(np.pi)), axis=1))


def state(events, seconds, threshold=1., depth=True, em_iterations=60, tolerance=1e-4):
    """step 2 on 10 s bins: the state's posterior mean x and the expected drop per bin (depth) or the event rate (count only); converged or not"""
    assert seconds >= 10 and threshold > 0 and em_iterations > 0
    assert np.isfinite(events).all() and np.all(events[:, 1] >= threshold)
    n, total, squares = bins(events, (seconds + 9) // 10); k = len(n); exposure = np.minimum(10, seconds - 10 * np.arange(k)) / 10
    if not n.sum(): return {"severity": np.zeros(k), "x": np.zeros(k), "converged": True}
    gamma, sigma, floor = float(depth), np.log(2.), np.log(threshold)
    first = variational_fit(n, total, squares, gamma, sigma, floor, em_iterations, exposure)
    result = em_refine(first, n, total, squares, gamma, sigma, floor, exposure, em_iterations, tolerance)
    x, v, alpha, beta, q = result["m"], result["v"], result["alpha"], result["beta"], result["q"]
    score = float((energy(x, v, result["c"]) / q - k) / 2); boundary = "lower" if q <= 1.0001e-6 else "upper" if q >= 1.9999 else "interior"
    optimal = abs(score) if boundary == "interior" else max(score, 0) if boundary == "lower" else max(-score, 0)
    converged = bool(first["success"] and result["em_converged"] and result["residual"] < 1e-6 and optimal / k < tolerance)
    severity = marked_rate_mean(x, v, alpha, beta, gamma, sigma, floor) if depth else np.exp(beta + x + .5 * v)
    assert np.isfinite(severity).all() and np.all(v >= 0)
    return {"severity": severity, "x": x, "converged": converged}


def model(key, x1, keep_failed=False):
    """DipDepth on one night's SpO2 (row, arrays); keep_failed: a state fit's AssertionError becomes an empty unconverged state"""
    n = len(x1); x = np.where(np.isnan(x1), np.nanmedian(x1), x1); u, h = step1(x); e = runs(u, h)
    def fit(depth):
        try: return state(e, n, 1., depth=depth)
        except AssertionError:
            if not keep_failed: raise
            return {"severity": np.zeros((n + 9) // 10), "x": np.zeros((n + 9) // 10), "converged": False}
    st = {dep: fit(dep) for dep in (True, False)}; E, step = count(x, x1, u, h)
    arrays = {"oxygen": x1, "events": e, "depth_severity": st[True]["severity"], "depth_x": st[True]["x"], "count_only_severity": st[False]["severity"],
              "count_only_x": st[False]["x"], "count_events": E, "rule3_nadirs": odi(x, 3).astype(float)}
    return {"key": key, "seconds": n, "step": step, "fine": bool(step < 0.2), "converged": bool(st[True]["converged"] and st[False]["converged"])}, arrays
