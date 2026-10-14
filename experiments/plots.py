"""Plotting helpers (matplotlib, static PNG for the LaTeX report).

Colour roles follow one fixed categorical order (blue, orange, aqua) so that
colour always identifies the same entity within a figure; simulation and theory
are distinguished by mark type (markers with error bars vs. lines/open markers).
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]      # validated categorical slots 1-4
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e3e2de"
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
CMAP = LinearSegmentedColormap.from_list("seqblue", SEQ)

plt.rcParams.update({
    "font.size": 8.5, "axes.titlesize": 9.5, "axes.labelsize": 8.5, "legend.fontsize": 7.5,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "legend.frameon": False,
    "figure.dpi": 200, "savefig.dpi": 200, "savefig.bbox": "tight", "text.color": INK,
})


def _save(fig, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def metric_vs_x(path, series, xlabel, ylabel, title, xticks=None, xticklabels=None, ylim_zero=False,
                logy=False, figsize=(5.0, 3.2), legend_loc="best"):
    """Markers with 95% CI error bars for simulation, lines or open markers for theory.

    Each entry of `series` is a dict with keys: label, x, y, yerr (optional), color (slot index,
    optional), and optionally theory_x, theory_y, theory_label, theory_markers (bool).
    """
    fig, ax = plt.subplots(figsize=figsize)
    for i, s in enumerate(series):
        col = SERIES[s.get("color", i) % len(SERIES)]
        if "theory_x" in s:
            tcol = SERIES[s["theory_color"] % len(SERIES)] if "theory_color" in s else col
            if s.get("theory_markers"):
                ax.plot(s["theory_x"], s["theory_y"], ls="none", marker="s", ms=7, mfc="none", mec=tcol,
                        mew=1.4, label=s.get("theory_label", "theory"), zorder=3)
            else:
                ax.plot(s["theory_x"], s["theory_y"], color=tcol, lw=1.6, label=s.get("theory_label", "theory"), zorder=2)
        if "y" in s:
            ax.errorbar(s["x"], s["y"], yerr=s.get("yerr"), fmt="o", color=col, ms=4.5, capsize=3, lw=1.2,
                        label=s["label"], zorder=4)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title, loc="left")
    if xticks is not None:
        ax.set_xticks(xticks)
    if xticklabels is not None:
        ax.set_xticklabels(xticklabels)
    if logy:
        ax.set_yscale("log")
    elif ylim_zero:
        ax.set_ylim(bottom=0)
    ax.legend(loc=legend_loc)
    _save(fig, path)


def distribution_panels(path, bin_labels, panels, suptitle=None, ncols=2, width=None, nbins_show=17):
    """One panel per configuration: simulated p_n bars (95% CI) and theory as open markers."""
    n = len(panels)
    ncols = min(ncols, n)
    nrows = int(np.ceil(n / ncols))
    width = width or 5.2 * ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(width, 2.6 * nrows), squeeze=False, sharey=False)
    x = np.arange(nbins_show)
    for ax, p in zip(axes.flat, panels):
        ax.bar(x, p["p"][:nbins_show], width=0.72, color=SERIES[0], yerr=p.get("half")[:nbins_show] if p.get("half") is not None else None,
               error_kw={"elinewidth": 0.9, "capsize": 2, "ecolor": INK2}, label="simulation", zorder=2)
        if p.get("theory") is not None:
            ax.plot(x, p["theory"][:nbins_show], ls="none", marker="s", ms=5, mfc="none", mec=SERIES[1], mew=1.3,
                    label="theory", zorder=3)
        ax.set_xticks(x); ax.set_xticklabels(bin_labels[:nbins_show], fontsize=6.5)
        ax.set_title(p["title"], loc="left"); ax.set_xlabel("number in system $n$"); ax.set_ylabel("probability")
        ax.legend()
    for ax in list(axes.flat)[n:]:
        ax.axis("off")
    if suptitle:
        fig.suptitle(suptitle, fontsize=9.5, x=0.01, ha="left")
    fig.tight_layout()
    _save(fig, path)


def grouped_distributions(path, bin_labels, dists, title, theory=None, theory_label="stationary $p_n$ (theory)",
                          nbins_show=17, figsize=(5.4, 3.0), legend_loc="best"):
    """Several distributions side by side per bin (e.g. time-average, pre-arrival, post-arrival)."""
    k = len(dists)
    x = np.arange(nbins_show)
    w = 0.8 / k
    fig, ax = plt.subplots(figsize=figsize)
    for i, d in enumerate(dists):
        ax.bar(x + (i - (k - 1) / 2) * w, d["p"][:nbins_show], width=w * 0.95, color=SERIES[i],
               yerr=d["half"][:nbins_show] if d.get("half") is not None else None,
               error_kw={"elinewidth": 0.7, "capsize": 1.5, "ecolor": INK2}, label=d["label"], zorder=2)
    if theory is not None:
        ax.plot(x, theory[:nbins_show], ls="none", marker="s", ms=5, mfc="none", mec=INK, mew=1.2,
                label=theory_label, zorder=3)
    ax.set_xticks(x); ax.set_xticklabels(bin_labels[:nbins_show], fontsize=6.5)
    ax.set_xlabel("number in system $n$"); ax.set_ylabel("probability"); ax.set_title(title, loc="left")
    ax.legend(loc=legend_loc)
    _save(fig, path)


def heatmap_pair(path, sim, th, title, xlabel="$n_2$ (class 2 calls)", ylabel="$n_1$ (class 1 calls)"):
    vmax = np.nanmax([np.nanmax(sim), np.nanmax(th)])
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.9))
    for ax, M, sub in zip(axes, (sim, th), ("simulation (time average)", "product form (theory)")):
        im = ax.imshow(np.ma.masked_invalid(M), cmap=CMAP, vmin=0, vmax=vmax, origin="lower")
        ax.grid(False)
        ax.set_xticks(range(M.shape[1])); ax.set_yticks(range(M.shape[0]))
        ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(sub, loc="left")
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                if np.isfinite(M[i, j]):
                    ax.text(j, i, f"{M[i, j]:.3f}", ha="center", va="center", fontsize=7,
                            color="white" if M[i, j] > 0.55 * vmax else INK)
                else:
                    ax.text(j, i, "infeasible", ha="center", va="center", fontsize=5.5, color=INK2)
    cb = fig.colorbar(im, ax=axes, shrink=0.85, pad=0.02)
    cb.set_label("probability")
    fig.suptitle(title, fontsize=9.5, x=0.01, ha="left")
    _save(fig, path)


def sizing_curve(path, ms, B, target, sim_points, title):
    fig, ax = plt.subplots(figsize=(5.0, 3.2))
    ax.plot(ms, B, marker="o", ms=3.5, color=SERIES[0], lw=1.5, label="Erlang B $B_m(a=3)$")
    ax.axhline(target, color=INK2, ls="--", lw=1.0, label=f"{100 * target:g}% target")
    if sim_points:
        xs = [p[0] for p in sim_points]; ys = [p[1] for p in sim_points]; es = [p[2] for p in sim_points]
        ax.errorbar(xs, ys, yerr=es, fmt="s", color=SERIES[1], ms=5, capsize=4, lw=1.2,
                    label="simulation (mean $\\pm$ 95% CI)", zorder=5)
    ax.set_yscale("log")
    ax.set_xticks(ms)
    ax.set_xlabel("number of channels $m$"); ax.set_ylabel("blocking probability")
    ax.set_title(title, loc="left")
    ax.legend()
    _save(fig, path)
