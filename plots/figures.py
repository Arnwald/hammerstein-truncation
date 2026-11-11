"""
figures.py
==========

Figures of the paper "Truncation Error of Single-Level Hammerstein
Identification From Exponential Swept Sines", drawn at the size of the journal:
Times (STIX), one column (3.5 in) or the full page (7.16 in), TrueType fonts
embedded. The script only reads the outputs of `hsim` and plots them:

    python3 plots/figures.py output [--out figures]

where `output` contains the directories written by `hsim exact`, `hsim chain`
and `hsim ess`.

Figures:
  fig_ess        envelope of the deconvolved response to a sweep (one column)
  fig_radius     singularities and Taylor polynomials of arctan and artanh (page)
  fig_bounds     exact ||e2||, ||e1|| against the general bounds (one column)
  fig_curves     relative error per harmonic, and its terms e1, e2 (page)
  fig_map        map (lambda, rho) of the worst harmonic, low-level floors (page)
  fig_running    running of the effective kernels with the matching scale, and
                 the beta function of the describing function (one column)
  fig_validate   closed form against the simulated measurement chain (one column)
"""

from __future__ import annotations

import json
import os
import sys
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

# Categorical palette; the harmonics are also labelled directly on the curves.
HARM = ["#2a78d6", "#eb6834", "#1baf7a", "#6250d6"]
FUNC = {"arctan": "#2a78d6", "artanh": "#eb6834"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
RHO_MAIN = 2.0
COL, PAGE = 3.5, 7.16  # IEEE column and page widths (in)

plt.rcParams.update(
    {
        "font.family": "STIXGeneral",
        "mathtext.fontset": "stix",
        "font.size": 9,
        "axes.titlesize": 9,
        "axes.labelsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": INK,
        "ytick.labelcolor": INK,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "legend.frameon": False,
        "lines.linewidth": 1.1,
        "pdf.fonttype": 42,
        "savefig.dpi": 300,
    }
)


def db20(x):
    with np.errstate(divide="ignore", invalid="ignore"):
        return 20 * np.log10(x)


def save(fig, out, name):
    os.makedirs(out, exist_ok=True)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # a missing glyph is an error
        # no cropping: the figure keeps the width of the column or of the page,
        # and its fonts their nominal size once placed in the paper
        fig.savefig(os.path.join(out, f"{name}.pdf"))
        fig.savefig(os.path.join(out, f"{name}.png"))
    plt.close(fig)
    print(f"  {name}")


def tag(ax, s):
    ax.text(0.01, 0.99, s, transform=ax.transAxes, ha="left", va="top", color=INK,
            fontsize=9, fontweight="bold")


def mark_levels(ax, rho_db):
    """Identification level and radius, labelled above the frame."""
    ax.axvline(0.0, color=INK, lw=0.7)
    ax.axvline(rho_db, color=MUTED, lw=0.7, ls=":")
    tr = ax.get_xaxis_transform()
    ax.text(0.0, 1.01, r"$\lambda_0$", color=INK, va="bottom", ha="center", fontsize=9,
            transform=tr)
    ax.text(rho_db, 1.01, r"$\rho$", color=INK, va="bottom", ha="center", fontsize=9,
            transform=tr)


def rho_tag(r):
    return f"{r:g}" if r != int(r) else f"{int(r)}"


class Exact:
    def __init__(self, path):
        self.path = path
        with open(os.path.join(path, "meta.json"), encoding="utf-8") as f:
            self.meta = json.load(f)
        self.lam = np.array(self.meta["lam_db"])
        self.rho = np.array(self.meta["rho_db"])
        self.n = self.meta["n_curves"]
        self.harm = list(range(1, self.n + 1, 2))

    def load(self, name):
        return np.load(os.path.join(self.path, name))

    def curves(self, f, rho):
        return self.load(f"curves_{f}_rho{rho_tag(rho)}.npy")

    def bounds(self, f, rho):
        return self.load(f"bounds_{f}_rho{rho_tag(rho)}.npy")


def fig_ess(ess_dir, out):
    """Deconvolved response to an exponential sweep: the harmonics are ahead by L ln m."""
    with open(os.path.join(ess_dir, "meta.json"), encoding="utf-8") as f:
        meta = json.load(f)
    ir = np.load(os.path.join(ess_dir, "ir_zoom.npy"))
    fs, t0, L = meta["fs"], meta["t0"], meta["L"]
    t = (meta["ir_zoom_start"] + np.arange(ir.size) - t0) / fs
    # envelope: maximum of |ir| over blocks of 2 ms, in dB re the linear peak
    blk = int(0.002 * fs)
    nb = ir.size // blk
    env = np.abs(ir[:nb * blk]).reshape(nb, blk).max(axis=1)
    tb = t[:nb * blk].reshape(nb, blk).mean(axis=1)
    env_db = db20(env / np.abs(ir).max())
    fig, ax = plt.subplots(figsize=(COL, 2.3), layout="constrained")
    ax.plot(tb, env_db, color=HARM[0], lw=0.9)
    for m in range(1, meta["harmonics"] + 1):
        tm = -L * np.log(m)
        k = np.argmin(np.abs(tb - tm))
        ax.text(tm, env_db[max(k - 2, 0):k + 3].max() + 4, f"{m}", ha="center", va="bottom",
                fontsize=8, color=INK)
    ax.set_xlim(-2.05, 0.15)
    ax.set_ylim(-120, 15)
    ax.set_xlabel("time relative to the linear response (s)")
    ax.set_ylabel("envelope (dB)")
    save(fig, out, "fig_ess")


def fig_radius(ex, out):
    """The radius of convergence: singularities in the complex plane, Taylor polynomials."""
    x = np.array(ex.meta["taylor_x"])
    degs = [int(d) for d in ex.meta["taylor_degrees"]]
    fig = plt.figure(figsize=(PAGE, 2.45), layout="constrained")
    gs = fig.add_gridspec(1, 3, width_ratios=[0.8, 1, 1])
    ax = fig.add_subplot(gs[0, 0])
    th = np.linspace(0, 2 * np.pi, 400)
    ax.fill(np.cos(th), np.sin(th), color="#e8f0fb", zorder=0)
    ax.plot(np.cos(th), np.sin(th), color=MUTED, lw=0.8)
    ax.axhline(0, color=MUTED, lw=0.6)
    ax.axvline(0, color=MUTED, lw=0.6)
    ax.plot([0, 0], [1, -1], "x", color=FUNC["arctan"], ms=7, mew=1.6)
    ax.plot([1, -1], [0, 0], "x", color=FUNC["artanh"], ms=7, mew=1.6)
    ax.text(0.08, 1.08, r"arctan: $\pm i\rho$", color=INK, fontsize=8, va="bottom")
    ax.text(1.05, 0.08, "artanh:\n" + r"$\pm\rho$", color=INK, fontsize=8, va="bottom")
    ax.text(-0.95, -0.35, "converges", color=INK, fontsize=8)
    ax.text(0.45, -1.42, "diverges", color=INK, fontsize=8)
    ax.set_xlim(-1.6, 1.8)
    ax.set_ylim(-1.6, 1.6)
    ax.set_aspect("equal")
    ax.grid(False)
    ax.set_xticks([-1, 0, 1], [r"$-\rho$", "0", r"$\rho$"])
    ax.set_yticks([-1, 0, 1], [r"$-i\rho$", "0", r"$i\rho$"])
    ax.set_xlabel(r"$\mathrm{Re}\,x$")
    ax.set_ylabel(r"$\mathrm{Im}\,x$")
    ax.set_title("(a) singularities", loc="left")
    shades = ["#9ec5f0", "#5a9be3", "#2a78d6", "#0d3f87"]
    for k, f in enumerate(("arctan", "artanh")):
        ax = fig.add_subplot(gs[0, 1 + k])
        d = ex.load(f"taylor_{f}.npy")
        for i, deg in enumerate(degs):
            ax.plot(x, d[1 + i], color=shades[i], lw=0.9, label=f"degree {deg}")
        ax.plot(x, d[0], color=INK, lw=1.4, label=f"$\\rho\\,\\mathrm{{{f}}}(x/\\rho)$")
        ax.axvline(1.0, color=MUTED, lw=0.7, ls=":")
        ax.set_ylim(-1.0, 3.5)
        ax.set_xlim(0, 2.5)
        ax.set_xlabel(r"input amplitude $x/\rho$")
        ax.set_title(f"({'bc'[k]}) {f} and its Taylor polynomials", loc="left")
        if k == 0:
            ax.set_ylabel(r"output$/\rho$")
            ax.legend(loc="upper left", ncol=1, handlelength=1.5, labelspacing=0.2)
    save(fig, out, "fig_radius")


def fig_bounds(ex, out):
    rho_db = 20 * np.log10(RHO_MAIN)
    fig, axes = plt.subplots(2, 1, figsize=(COL, 3.6), sharex=True, layout="constrained")
    ylab = [r"$\|\mathbf{e}_2(\lambda)\|_2$ (dB)", r"$\|\mathbf{e}_1(\lambda)\|_2$ (dB)"]
    ylim = [(-160, 20), (-200, 20)]
    for k, ax in enumerate(axes):
        for f in ("arctan", "artanh"):
            b = ex.bounds(f, RHO_MAIN)
            bound = b[2 * k + 1].copy()
            if k == 1:
                bound[ex.lam > 0] = np.nan  # the bound on e1 only holds for lambda <= lambda_0
            ax.plot(ex.lam, db20(b[2 * k]), color=FUNC[f])
            ax.plot(ex.lam, db20(bound), color=FUNC[f], ls="--", lw=0.9)
        mark_levels(ax, rho_db)
        ax.set_ylim(*ylim[k])
        ax.set_xlim(ex.lam[0], ex.lam[-1])
        ax.set_ylabel(ylab[k])
        tag(ax, "(a)" if k == 0 else "(b)")
    axes[1].set_xlabel(r"prediction level $\lambda$ (dB re $\lambda_0$)")
    handles = [Line2D([], [], color=FUNC["arctan"], label="arctan"),
               Line2D([], [], color=FUNC["artanh"], label="artanh"),
               Line2D([], [], color=MUTED, label="exact"),
               Line2D([], [], color=MUTED, ls="--", lw=0.9, label="bound")]
    axes[0].legend(handles=handles, ncol=2, loc="upper left", bbox_to_anchor=(0.07, 1.0),
                   handlelength=2.0, columnspacing=1.0)
    save(fig, out, "fig_bounds")


def fig_curves(ex, out):
    rho_db = 20 * np.log10(RHO_MAIN)
    fig, axes = plt.subplots(2, 2, figsize=(PAGE, 4.2), sharex=True, sharey=True,
                             layout="constrained")
    ymin = -140
    tags = iter("(a) (b) (c) (d)".split())
    for row, f in enumerate(("arctan", "artanh")):
        c = ex.curves(f, RHO_MAIN)  # [q][i][lambda], q: e, e1, e2, series of e1, h
        rel = lambda q, i: db20(c[q, i] / c[4, i])  # noqa: E731
        ax = axes[row, 0]
        for i, m in enumerate(ex.harm):
            y = rel(0, i)
            ax.plot(ex.lam, y, color=HARM[i])
            floor = y[np.isfinite(y)][0]  # value as lambda -> 0
            ax.axhline(floor, color=HARM[i], lw=0.7, ls="--", alpha=0.8)
            ax.text(-39.3, floor - 2.0, f"$m={m}$", color=INK, fontsize=8, va="top")
        ax.hlines(-20, -33, ex.lam[-1], color=MUTED, lw=0.6, ls="-.")  # clear of the labels
        mark_levels(ax, rho_db)
        ax.set_ylim(ymin, 40)
        ax.set_ylabel(f"{f}: " + r"$|e_m|/|H_m|$ (dB)")
        tag(ax, next(tags))
        ax = axes[row, 1]
        i = len(ex.harm) - 1
        ax.plot(ex.lam, rel(0, i), color=INK, label=r"$e_7=e_{1,7}+e_{2,7}$")
        ax.plot(ex.lam, rel(1, i), color=FUNC["arctan"], ls="--",
                label=r"$e_{1,7}$ (missing orders)")
        ax.plot(ex.lam, rel(2, i), color=FUNC["artanh"], ls="-.",
                label=r"$e_{2,7}$ (pollution)")
        ax.plot(ex.lam, rel(3, i), color=MUTED, ls=":", lw=1.0,
                label=r"series of $e_{1,7}$ (127 terms)")
        mark_levels(ax, rho_db)
        ax.set_ylim(ymin, 40)
        tag(ax, next(tags))
        if row == 1:
            ax.legend(loc="lower right", frameon=True, facecolor="white", edgecolor="none",
                      framealpha=1.0)
    for ax in axes[1]:
        ax.set_xlabel(r"prediction level $\lambda$ (dB re $\lambda_0$)")
    save(fig, out, "fig_curves")


def fig_map(ex, out):
    fig = plt.figure(figsize=(PAGE, 2.9), layout="constrained")
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.1])
    cmap = plt.get_cmap("Blues_r").copy()
    cmap.set_bad("#f3f2ee")
    lo, hi = ex.lam[0], ex.lam[-1]
    for k, f in enumerate(("arctan", "artanh")):
        ax = fig.add_subplot(gs[0, k])
        z = np.ma.masked_invalid(ex.load(f"map_{f}.npy"))
        pm = ax.pcolormesh(ex.lam, ex.rho, z, cmap=cmap, vmin=-100, vmax=20, shading="auto",
                           rasterized=True)
        ax.contour(ex.lam, ex.rho, z, levels=[-40], colors="white", linewidths=0.8,
                   linestyles="--")
        ax.contour(ex.lam, ex.rho, z, levels=[-20], colors=INK, linewidths=1.0,
                   linestyles="solid")
        ax.plot([lo, hi], [lo, hi], color=FUNC["artanh"], lw=0.9, ls="--")
        ax.text(-5.5, -4.8, r"$\lambda=\rho$", color=INK, fontsize=8, ha="right")
        ax.axvline(0.0, color=INK, lw=0.6)
        ax.set_ylim(ex.rho[0], ex.rho[-1])
        ax.set_xlabel(r"$\lambda$ (dB re $\lambda_0$)")
        ax.grid(False)
        ax.set_title(f"({'ab'[k]}) {f}: worst harmonic, $N={ex.n}$", loc="left")
        if k == 0:
            ax.set_ylabel(r"$\rho$ (dB re $\lambda_0$)")
        else:
            ax.tick_params(labelleft=False)
            ax.text(-20, -3.2, r"not identifiable ($\rho\leq\lambda_0$)", fontsize=8,
                    color=INK, ha="center")
            ax.text(10.5, 2.5, "undefined\n" + r"($\lambda\geq\rho$)", fontsize=8,
                    color=INK, ha="center")
    cb = fig.colorbar(pm, ax=fig.axes[:2], location="bottom", shrink=0.75, aspect=35, pad=0.02)
    cb.set_label(r"worst harmonic $|e_m|/|H_m|$ (dB); black: $-20$ dB, white dashed: $-40$ dB")
    cb.ax.tick_params(labelsize=9)
    ax = fig.add_subplot(gs[0, 2])
    fl = ex.load("floors_arctan.npy"), ex.load("floors_artanh.npy")
    n_list = [int(n) for n in ex.meta["n_list"]]
    shades = ["#9ec5f0", "#5a9be3", "#2a78d6", "#0d3f87"]
    pos = ex.rho > 0.01
    for ni, n in enumerate(n_list):
        for arr, ls in ((fl[0], "-"), (fl[1], "--")):
            worst = np.nanmax(arr[ni][pos], axis=1)
            ax.plot(ex.rho[pos], db20(worst), color=shades[ni], ls=ls)
    ax.axhline(-20, color=MUTED, lw=0.6, ls="-.")
    ax.set_ylim(-45, 30)
    ax.set_xlim(0, ex.rho[-1])
    ax.set_xlabel(r"$\rho$ (dB re $\lambda_0$)")
    ax.set_ylabel(r"low-level floor, worst harmonic (dB)")
    handles = [Line2D([], [], color=shades[ni], label=f"$N={n}$") for ni, n in enumerate(n_list)]
    handles[2:2] = [Line2D([], [], color=MUTED, label="arctan"),
                    Line2D([], [], color=MUTED, ls="--", label="artanh")]
    ax.legend(handles=handles, ncol=2, loc="lower left", columnspacing=1.0, frameon=True,
              facecolor="white", edgecolor="none", framealpha=1.0)
    ax.set_title("(c) low-level floor", loc="left")
    save(fig, out, "fig_map")


def fig_running(ex, out):
    mu = np.array(ex.meta["mu_db"])
    fig, axes = plt.subplots(2, 1, figsize=(COL, 4.4), layout="constrained",
                             gridspec_kw={"height_ratios": [1.15, 1]})
    ax = axes[0]
    for f, ls in (("arctan", "-"), ("artanh", "--")):
        r = ex.load(f"running_{f}.npy")  # [q][i][mu], q: g/c, (g - c)/c, beta/c
        for i, n in enumerate(ex.harm):
            ax.plot(mu, db20(np.abs(r[1, i])), color=HARM[i], ls=ls)
    # labels below the curves (they rise to the right)
    ra = ex.load("running_arctan.npy")
    for i, n in enumerate(ex.harm):
        y = db20(np.abs(ra[1, i]))
        x0 = max(-28.0, mu[np.argmax(y > -205.0)] + 2.0)
        j = np.searchsorted(mu, x0)
        ax.text(x0, y[j] - 7, rf"$n={n}$: $(\mu/\rho)^{{{ex.n + 2 - n}}}$", color=INK,
                fontsize=8, ha="left", va="top")
    lam0 = -20 * np.log10(RHO_MAIN)
    ax.axvline(0.0, color=MUTED, lw=0.7, ls=":")
    ax.axvline(lam0, color=INK, lw=0.7)
    tr = ax.get_xaxis_transform()
    ax.text(0.3, 1.01, r"$\mu=\rho$", color=INK, ha="left", va="bottom", fontsize=9,
            transform=tr)
    ax.text(lam0 - 0.3, 1.01, r"$\lambda_0$ ($\rho=2$)", color=INK, ha="right",
            va="bottom", fontsize=9, transform=tr)
    ax.set_ylim(-220, 20)
    ax.set_xlim(mu[0], 12)
    ax.set_xlabel(r"matching scale $\mu/\rho$ (dB)")
    ax.set_ylabel(r"$|\hat G_n(\mu)-G_n|/|G_n|$ (dB)")
    ax.legend(handles=[Line2D([], [], color=MUTED, label="arctan"),
                       Line2D([], [], color=MUTED, ls="--", label="artanh")],
              loc="lower right")
    tag(ax, "(a)")
    ax = axes[1]
    for f in ("arctan", "artanh"):
        d = ex.load(f"describing_{f}.npy")  # g(mu), beta(mu), N = 1
        ok = np.isfinite(d[0])
        ax.plot(d[0, ok], d[1, ok], color=FUNC[f], label=f)
        for udb, mk in ((-6.02, "o"), (0.0, "s"), (6.02, "^")):
            k = np.argmin(np.abs(mu - udb))
            if np.isfinite(d[0, k]) and abs(mu[k] - udb) < 0.05:
                ax.plot(d[0, k], d[1, k], mk, color=FUNC[f], ms=4, mfc="white", mew=1.0)
    ax.plot([0, 1], [0, 0], "o", color=INK, ms=4)
    ax.axvline(2.0, color=MUTED, lw=0.7, ls=":")
    ax.axhline(0.0, color=MUTED, lw=0.6)
    ax.annotate("", xy=(0.45, -0.42), xytext=(0.8, -0.28),
                arrowprops=dict(arrowstyle="->", color=FUNC["arctan"], lw=0.9))
    ax.annotate("", xy=(1.62, 2.6), xytext=(1.44, 0.8),
                arrowprops=dict(arrowstyle="->", color=FUNC["artanh"], lw=0.9))
    ax.text(1.0, 0.15, "IR", ha="center", va="bottom", fontsize=8, color=INK)
    ax.text(0.0, 0.15, "UV", ha="left", va="bottom", fontsize=8, color=INK)
    ax.text(1.97, 2.0, r"$\beta\to\infty$:" + "\nthreshold\n" + r"$\mu=\rho$", ha="right",
            va="center", fontsize=8, color=INK)
    ax.set_xlim(0, 2.05)
    ax.set_ylim(-0.6, 3.5)
    ax.set_xlabel(r"effective gain $g=\hat G_1(\mu)$, $N=1$")
    ax.set_ylabel(r"$\beta(g)=\mu\,\mathrm{d}g/\mathrm{d}\mu$")
    ax.legend(handles=[Line2D([], [], color=FUNC["arctan"], label="arctan"),
                       Line2D([], [], color=FUNC["artanh"], label="artanh"),
                       Line2D([], [], color=MUTED, marker="o", ls="none", mfc="white",
                              label=r"$\mu=\rho/2$"),
                       Line2D([], [], color=MUTED, marker="s", ls="none", mfc="white",
                              label=r"$\mu=\rho$"),
                       Line2D([], [], color=MUTED, marker="^", ls="none", mfc="white",
                              label=r"$\mu=2\rho$")],
              loc="upper left", bbox_to_anchor=(0.06, 1.0), ncol=1)
    tag(ax, "(b)")
    save(fig, out, "fig_running")


def fig_validate(chain_dir, out):
    """Closed form against the simulated measurement chain."""
    with open(os.path.join(chain_dir, "meta.json"), encoding="utf-8") as f:
        meta = json.load(f)
    gains = np.array(meta["gains_db"])
    sim = np.load(os.path.join(chain_dir, "chain.npy"))  # [gain][harmonic], dB
    closed = np.load(os.path.join(chain_dir, "closed_form.npy"))
    level = np.load(os.path.join(chain_dir, "closed_form_level.npy"))  # dB re H1
    m = closed.shape[1]
    # deconvolution leak per window (dB re H1); none in the window of the fundamental
    leak = np.concatenate(([-np.inf], meta["leak_db"]))
    # a point is resolved if the true harmonic is 20 dB above the leak, and the
    # error 20 dB above the error of the chain itself at the identification level
    clean = (level - leak[None, :]) > 20.0
    i0 = np.argmin(np.abs(gains))
    floor_abs = level[i0] + sim[i0]
    clean &= (level + closed) > floor_abs[None, :] + 20.0
    fig, ax = plt.subplots(figsize=(COL, 3.0), layout="constrained")
    for i in range(m):
        h = i + 1
        col = HARM[(h - 1) // 2 % 4] if h % 2 else MUTED
        ls = "-" if h % 2 else "--"
        ax.plot(gains, closed[:, i], color=col, ls=ls, lw=1.0)
        ok = clean[:, i]
        ax.plot(gains[ok], sim[ok, i], "o", color=col, ms=2.8, mfc="none", mew=0.7)
        ax.plot(gains[~ok], sim[~ok, i], "x", color=MUTED, ms=2.5, mew=0.5)
    ax.set_ylim(-160, 40)
    ax.set_xlim(gains[0] - 0.5, gains[-1] + 0.5)
    ax.axvline(0, color=INK, lw=0.6)
    ax.set_xlabel(r"gain $\lambda$ (dB re $\lambda_0$)")
    ax.set_ylabel(r"$|e_m|/|H_m|$ (dB)")
    handles = [Line2D([], [], color=HARM[k], label=f"$m={2 * k + 1}$") for k in range(4)]
    handles += [Line2D([], [], color=MUTED, ls="--", label="$m=2,4,6$"),
                Line2D([], [], color=INK, lw=1.0, label="closed form"),
                Line2D([], [], color=INK, marker="o", ls="none", ms=2.8, mfc="none", mew=0.7,
                       label="sweep chain"),
                Line2D([], [], color=MUTED, marker="x", ls="none", ms=2.5, mew=0.5,
                       label="not resolved")]
    ax.legend(handles=handles, ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.0),
              handlelength=1.6, columnspacing=0.8, labelspacing=0.25, borderaxespad=0.2)
    save(fig, out, "fig_validate")
    d = np.where(clean, np.abs(closed - sim), np.nan)
    print(f"  chain vs closed form: max difference {np.nanmax(d):.4f} dB over "
          f"{int(clean.sum())} resolved points (of {clean.size})")


def main(argv):
    if not argv:
        print(__doc__)
        sys.exit(2)
    root, out = argv[0], "figures"
    it = iter(argv[1:])
    for a in it:
        if a == "--out":
            out = next(it)
        else:
            raise SystemExit(f"unknown option: {a}")
    ex = Exact(os.path.join(root, "exact"))
    fig_ess(os.path.join(root, "ess"), out)
    fig_radius(ex, out)
    fig_bounds(ex, out)
    fig_curves(ex, out)
    fig_map(ex, out)
    fig_running(ex, out)
    fig_validate(os.path.join(root, "chain"), out)


if __name__ == "__main__":
    main(sys.argv[1:])
