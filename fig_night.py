"""Fig. 1 of the paper from examples/tr03-0092.npz (Boston night tr03-0092 of the PhysioNet/CinC Challenge 2018): writes fig_night.pdf.
Run: python3 fig_night.py"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

HERE = Path(__file__).resolve().parent
Z = np.load(HERE / "examples/tr03-0092.npz")

VIOLET, BLUE, TECH_BAR, TXT = "#57068C", "#2C7BB6", "#F4A582", "#555555"
LIGHT = "#E6E6E6"; BAND = "#CDB5E0"
WIDTH = 7.16
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"], "font.size": 8, "axes.labelsize": 8,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5, "xtick.major.pad": 2, "ytick.major.pad": 2, "axes.labelpad": 2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.unicode_minus": True, "pdf.fonttype": 42,
    "lines.solid_capstyle": "butt", "savefig.dpi": 200,
})
NOTE = 7                                                               # annotation size, pt


def inch_axes(fig, left, bottom, w, h):
    """an axes placed in inches from the figure's lower left"""
    W, H = fig.get_size_inches(); return fig.add_axes([left / W, bottom / H, w / W, h / H])


def letter(fig, s, x, y):
    """bold panel letter at a figure position in inches (x from left, y from top)"""
    W, H = fig.get_size_inches(); fig.text(x / W, 1 - y / H, s, fontsize=9, fontweight="bold", ha="left", va="top")


def tidy(a, bottom=True):
    a.tick_params(labelbottom=bottom)
    if not bottom: a.spines["bottom"].set_visible(False); a.tick_params(bottom=False)


# ---------------------------------------------------------------- the night's arrays
XS = Z["spo2_filled"].astype(float)                                           # SpO2 as fitted (gaps filled; none in the window shown)
U, H, LEVEL = Z["pulses"].astype(float), Z["shape"].astype(float), Z["level"].astype(float)
SHAPE = Z["shape_params"].astype(float)                              # fall T_f and recovery T_r, seconds
LAG = int(np.argmax(H))                                                # the fitted shape's deepest point, 27 s after its pulse
SCORED, EV, K = Z["scored"], Z["runs"], len(Z["state"])


def response(fall, recovery, delay, length=360):
    t = np.arange(length, dtype=float) - delay; u = np.maximum(t, 0)
    h = np.where(u < fall, .5 * (1 - np.cos(np.pi * u / fall)), np.exp(-np.maximum(u - fall, 0) / recovery))
    return np.where(t >= 0, h, 0)


# the delay is not stored: find the one that reproduces the stored shape
DELAY = [d for d in range(0, 30) if np.allclose(response(SHAPE[0], SHAPE[1], d), H)]
assert len(DELAY) == 1, DELAY
DELAY = DELAY[0]


def forward(q, h):
    n = len(q); m = 1 << int(np.ceil(np.log2(n + len(h)))); return np.fft.irfft(np.fft.rfft(q, m) * np.fft.rfft(h, m), m)[:n]


def lagged(q, w0, w1, lag=LAG, thr=1e-3):
    """pulses drawn under the dip each one makes: (seconds from w0, size) for pulses whose deepest effect falls in w0..w1"""
    j = np.flatnonzero(q > thr); j = j[(j + lag >= w0) & (j + lag < w1)]; return j + lag - w0, q[j]



def fig_night():
    W0, W1 = 14280, 14580                                              # five minutes where the level is flat: six dips
    FH = 3.75; fig = plt.figure(figsize=(WIDTH, FH)); tm = np.arange(W1 - W0) / 60
    fit = (LEVEL - forward(U, H))[W0:W1]; x = XS[W0:W1]
    # (a) trace and pulses
    LA, WA = 0.62, 3.70
    TR_B, TR_H, PU_B, PU_H = 3.00, 0.70, 2.50, 0.40                    # trace and pulse axes: bottom and height, inches
    g = inch_axes(fig, LA, TR_B, WA, TR_H); p = inch_axes(fig, LA, PU_B, WA, PU_H)
    tj, qj = lagged(U, W0, W1)
    for a in (g, p):
        for s_ in tj: a.axvline(s_ / 60, color=LIGHT, lw=0.5, zorder=0)
    g.plot(tm, LEVEL[W0:W1], color=VIOLET, lw=0.9, ls=(0, (3, 2)), zorder=2)
    g.plot(tm, x, color=BLUE, lw=1.1, zorder=3)
    g.plot(tm, fit, color=VIOLET, lw=0.8, zorder=4)
    k = (SCORED[:, 1] > W0) & (SCORED[:, 0] < W1)
    for on, end in SCORED[k]:
        on, end = max(on, W0), min(end, W1); g.add_patch(Rectangle(((on - W0) / 60, 90.0), (end - on) / 60, 0.7, fc=TECH_BAR, ec="none", zorder=1))
    g.set_xlim(0, 5); g.set_ylim(89.7, 99.2); g.set_yticks([90, 95]); g.set_ylabel("SpO₂ (%)", labelpad=5); g.set_xticks(range(6)); tidy(g, False)
    for y, txt, col, kw in ((98.7, "usual level", VIOLET, dict(ls=(0, (3, 2)), lw=0.9)), (96.9, "SpO₂", BLUE, dict(lw=1.1)), (95.1, "model fit", VIOLET, dict(lw=0.8))):
        g.plot([5.06, 5.20], [y, y], color=col, clip_on=False, **kw); g.text(5.27, y, txt, color=col, fontsize=NOTE, va="center", ha="left", clip_on=False)
    g.add_patch(Rectangle((5.06, 90.0), 0.14, 0.7, fc=TECH_BAR, ec="none", clip_on=False)); g.text(5.27, 90.35, "technician", color="#C0623A", fontsize=NOTE, va="center", ha="left", clip_on=False)
    p.vlines(tj / 60, 0, qj, color=VIOLET, lw=1.2); p.plot(tj / 60, qj, "o", ms=2.6, color=VIOLET, mec="none")
    p.set_xlim(0, 5); p.set_ylim(0, 5.3); p.set_yticks([0, 5]); p.set_ylabel("Pulse\n(points)", fontsize=NOTE, labelpad=5, linespacing=0.95); p.set_xticks(range(6)); p.set_xlabel("Time (min)")
    # the shift, said where the pulses are drawn: each sits at its peak effect, LAG s after it occurs
    p.vlines(5.13, 0.5, 3.8, color=VIOLET, lw=1.2, clip_on=False); p.plot(5.13, 3.8, "o", ms=2.6, color=VIOLET, mec="none", clip_on=False)
    p.text(5.27, 2.2, f"drawn {LAG} s\nlater, at peak\neffect", color=VIOLET, fontsize=NOTE, va="center", ha="left", clip_on=False, linespacing=0.95)
    assert qj.max() <= 5
    letter(fig, "a", 0.0, 0.05)
    # (b) the fitted drop shape, as tall as (a)'s two axes together so the longer text block has room
    BL = 5.72
    b = inch_axes(fig, BL, PU_B, WIDTH - BL - 0.13, TR_B + TR_H - PU_B)
    tt = np.arange(91); b.axhline(0, color=LIGHT, lw=0.6, zorder=0); b.plot(tt, -H[:91], color=VIOLET, lw=1.2)
    b.set_xlim(0, 90); b.set_ylim(-1.15, 1.65); b.set_xticks([0, 30, 60, 90]); b.set_yticks([0, -1]); b.set_yticklabels(["0", "−1"])
    b.set_ylabel("SpO₂ change ($-h$)", labelpad=1, y=0.40); b.set_xlabel("Time after pulse (s)")
    br = lambda x0, x1, y: b.annotate("", (x0, y), (x1, y), arrowprops=dict(arrowstyle="|-|,widthA=0.18,widthB=0.18", lw=0.5, color=TXT, shrinkA=0, shrinkB=0))
    br(0, DELAY, 0.12); br(DELAY, LAG, 0.12)
    b.plot([DELAY / 2, DELAY / 2], [0.17, 0.58], color=TXT, lw=0.4)
    b.text(3, 0.62, f"{DELAY} s delay", fontsize=NOTE, color=TXT, ha="left", va="bottom")
    b.text((DELAY + LAG) / 2 + 2, 0.20, "$T_f$", fontsize=NOTE, color=TXT, ha="center", va="bottom")
    for j, t_ in enumerate((f"fall $T_f$ = {SHAPE[0]:.1f} s", f"recovery $T_r$ = {SHAPE[1]:.1f} s")):
        b.text(89, 1.50 - 0.27 * j, t_, fontsize=NOTE, color=TXT, ha="right", va="center")
    letter(fig, "b", 5.40, 0.05)
    # (c) the whole night
    hr = np.arange(K) / 360; LC, WC = 0.66, WIDTH - 0.66 - 0.06
    e = inch_axes(fig, LC, 1.60, WC, 0.46); s = inch_axes(fig, LC, 0.94, WC, 0.56); st = inch_axes(fig, LC, 0.44, WC, 0.42)
    e.vlines(EV[:, 0] / 3600, 0, EV[:, 1], color="#3A3A3A", lw=0.6)
    e.set_ylim(0, 21); e.set_yticks([0, 20]); e.set_ylabel("Run\ndepth\n(points)", fontsize=NOTE, labelpad=3, linespacing=0.95)
    sev = Z["expected_drop"]; s.plot(hr, sev, color=VIOLET, lw=0.8); s.set_ylim(0, 3.7); s.set_yticks([0, 1, 2, 3]); s.set_ylabel("Expected\ndrop\n(points/10 s)", fontsize=NOTE, labelpad=3, linespacing=0.95)
    xm, sd = Z["state"], np.sqrt(Z["state_var"]); st.fill_between(hr, xm - 1.96 * sd, xm + 1.96 * sd, color=BAND, lw=0)
    st.plot(hr, xm, color=VIOLET, lw=0.8); st.axhline(0, color=TXT, lw=0.4, zorder=0); st.set_ylim(-4, 6.2); st.set_yticks([-4, 0, 4]); st.set_ylabel("State $x$", fontsize=NOTE, labelpad=3)
    for a in (e, s, st):
        a.yaxis.set_label_coords(-0.05, 0.5); a.set_xlim(0, K / 360); a.axvspan(W0 / 3600, W1 / 3600, color=VIOLET, alpha=0.18, lw=0, zorder=0)
    hours_ = np.arange(0, int(K // 360) + 1)
    for a in (e, s): a.set_xticks(hours_); tidy(a, False)
    st.set_xticks(hours_); st.set_xlabel("Time since the start of the recording (h)")
    e.annotate("a", (W0 / 3600, 21), xytext=(0, 1.5), textcoords="offset points", fontsize=NOTE, fontweight="bold", color=VIOLET, ha="center", va="bottom", annotation_clip=False)
    letter(fig, "c", 0.0, FH - 2.10)
    fig.savefig(HERE / "fig_night.pdf")
    w, h = fig.get_size_inches(); plt.close(fig); print(f"fig_night.pdf  {w:.2f} x {h:.2f} in")


if __name__ == "__main__":
    fig_night()
