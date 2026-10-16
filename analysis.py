"""
analysis.py

MarkerAnalysis: high-level wrapper used by main.py.
It loads one run, builds the PDFs and exposes one method per figure.
Every figure is saved in `fig_dir`; nothing is shown unless show=True.

Original implementation: Louis Puel (2024). Cleaned and restructured in 2026.
"""

from __future__ import annotations

import logging
import os

import numpy as np
import matplotlib.pyplot as plt

from class_pdf import (animate_trajectories, critical_field, critical_momentum, get_geometry,
                       losses_cumulative_weight, plot_trajectories, re_current, re_fraction, re_mask,
                       relativistic_parameters, set_style)
from import_simu_data import load_jorek
from pdf_constructor import build_pdfs

logger = logging.getLogger("hottail")

WHICH_SUFFIX = {"all": "", "re": "_RE", "losses": "_losses"}


class MarkerAnalysis:
    """
    Usage:
        A = MarkerAnalysis(track_files, coll_files, losses_files, tokamak="ITER")
        A.load()
        A.build_pdfs(grids, ...)
        A.plot_2d("p", "xi", which="all")
    """

    def __init__(self, track_files, coll_files, losses_files=None, tokamak="ITER",
                 fig_dir="figures", fig_format="png", show=False, geometry_dir=None):
        self.track_files = track_files
        self.coll_files = coll_files
        self.losses_files = losses_files
        self.tokamak = tokamak
        self.geom = get_geometry(tokamak, geometry_dir)
        self.R0, self.Z0 = self.geom["R0"], self.geom["Z0"]
        self.fig_dir = fig_dir
        self.fig_format = fig_format
        self.show = show
        self.data = None
        self.pdfs = None
        self.p_c = None
        os.makedirs(fig_dir, exist_ok=True)
        set_style()

    # ------------------------------------------------------------------ pipeline
    def load(self, t_start=None, t_end=None, t_step=None, origin_time=True):
        """Read the HDF5 files (see import_simu_data.load_jorek)."""
        self.data = load_jorek(self.track_files, self.coll_files, self.losses_files,
                               t_start=t_start, t_end=t_end, t_step=t_step, origin_time=origin_time)
        return self.data

    def build_pdfs(self, grids, p_c=None, **kwargs):
        """Build the PDFs (see pdf_constructor.build_pdfs for the options)."""
        self._need_data()
        self.p_c = p_c
        self.pdfs = build_pdfs(self.data, grids, self.R0, p_c=p_c, **kwargs)
        return self.pdfs

    # ------------------------------------------------------------------ helpers
    def _need_data(self):
        if self.data is None:
            raise RuntimeError("Call load() first")

    def _need_pdfs(self):
        if self.pdfs is None:
            raise RuntimeError("Call build_pdfs() first")

    def _path(self, name, ext=None):
        return os.path.join(self.fig_dir, f"{name}.{ext or self.fig_format}")

    def _save(self, fig, name):
        path = self._path(name)
        fig.savefig(path)
        logger.info("Saved figure %s", path)
        if self.show:
            plt.show()
        else:
            plt.close(fig)
        return path

    def _times_ms(self):
        return self.data.t * 1e3

    def _re_label(self):
        if self.p_c is None:
            return "local $p_c$"
        K = relativistic_parameters(p=self.p_c)[0]
        return rf"$p > {self.p_c:g}\,m_e c$ ($K > {K*1e-3:.0f}$ keV)"

    # ------------------------------------------------------------------ PDF figures
    def plot_1d(self, dim="p", which="all", n_times=3, t_range=None, vmin=None, vmax=None, log=True, mj=True):
        """
        f integrated over every dimension except `dim`, at `n_times` times evenly spaced in `t_range` (index range).
        With mj=True and dim="p", the reference Maxwell-Juttner is overlaid.
        """
        self._need_pdfs()
        pdf = self.pdfs.get(which)
        i0, i1 = (0, pdf.nt - 1) if t_range is None else t_range
        i1 = pdf.nt - 1 if i1 is None else i1
        times = np.unique(np.linspace(i0, i1, n_times).astype(int))
        x = pdf.centers[pdf.idx(dim)]
        prof = pdf.profile(dim)

        if log and vmin is None:   # readable floor: 15 decades below the maximum of the plotted curves
            vmin = 1e-15 * np.nanmax(prof[times])

        fig, ax = plt.subplots(figsize=(7, 4.8))
        colors = plt.get_cmap("plasma")(np.linspace(0, 0.85, len(times)))
        for c, i in zip(colors, times):
            ax.plot(x, prof[i], color=c, label=f"t = {pdf.time[i]*1e3:.3f} ms")
        if mj and dim == "p" and self.pdfs.f_mj is not None:
            f_mj = self.pdfs.f_mj.f[0]
            if log:
                f_mj = np.where(f_mj >= vmin, f_mj, np.nan)   # hide the underflowing tail
            ax.plot(self.pdfs.f_mj.centers[0], f_mj, ls="--", color=plt.rcParams["text.color"], lw=1.2, label="Maxwell-Juttner")
        if log:
            ax.set_yscale("log")
        ax.set_ylim(vmin, vmax)
        ax.set_xlabel(pdf.label(dim))
        ax.set_ylabel(r"$f$" + (f"({pdf.dims[pdf.idx(dim)]})" if len(pdf.dims) > 1 else ""))
        ax.legend(loc="best")
        return self._save(fig, f"PDF_1D_{pdf.dims[pdf.idx(dim)]}{WHICH_SUFFIX[which]}")

    def plot_2d(self, dim1="p", dim2="xi", which="all", i_time=0, log=True, vmin=None, vmax=None,
                interp=1, cmap=None, animate=False, fps=10):
        """Map of f in the (dim1, dim2) plane. animate=True saves a GIF over all times instead of one frame."""
        self._need_pdfs()
        pdf = self.pdfs.get(which)
        name = f"PDF_{pdf.dims[pdf.idx(dim1)]}{pdf.dims[pdf.idx(dim2)]}{WHICH_SUFFIX[which]}"
        opts = dict(log=log, vmin=vmin, vmax=vmax, interp=interp, cmap=cmap)
        if animate:
            return pdf.animate("2d", self._path(name, "gif"), fps=fps, dim1=dim1, dim2=dim2, **opts)
        fig, _, _ = pdf.plot_2d(dim1, dim2, i_time, **opts)
        return self._save(fig, name)

    def plot_poloidal(self, which="all", i_time=0, log=True, vmin=None, vmax=None, interp=1, cmap=None,
                      density=False, geometry=True, limiter=True, animate=False, fps=10):
        """Poloidal (R, Z) map. Requires a space-resolved PDF. density=True gives <n>_phi in m^-3."""
        self._need_pdfs()
        pdf = self.pdfs.get(which)
        name = f"PDF_poloidal{'_density' if density else ''}{WHICH_SUFFIX[which]}"
        opts = dict(log=log, vmin=vmin, vmax=vmax, interp=interp, cmap=cmap, density=density,
                    geom=self.geom if geometry else None, limiter=limiter)
        if animate:
            return pdf.animate("poloidal", self._path(name, "gif"), fps=fps, **opts)
        fig, _, _ = pdf.plot_poloidal(i_time, **opts)
        return self._save(fig, name)

    def plot_topview(self, which="all", i_time=0, log=True, vmin=None, vmax=None, interp=1, cmap=None,
                     geometry=True, limiter=False, animate=False, fps=10):
        """Top view (X, Y) map integrated along Z. Requires a space-resolved PDF."""
        self._need_pdfs()
        pdf = self.pdfs.get(which)
        name = f"PDF_topview{WHICH_SUFFIX[which]}"
        opts = dict(log=log, vmin=vmin, vmax=vmax, interp=interp, cmap=cmap,
                    geom=self.geom if geometry else None, limiter=limiter)
        if animate:
            return pdf.animate("topview", self._path(name, "gif"), fps=fps, **opts)
        fig, _, _ = pdf.plot_topview(i_time, **opts)
        return self._save(fig, name)

    # ------------------------------------------------------------------ time traces
    def plot_re_fraction(self, xi_dependent=False, vmin=None, vmax=None):
        """
        Weighted RE fraction N_RE(t)/N_0 with the isotropic criterion (and the pitch-dependent one if
        xi_dependent=True). With a prescribed p_c, a single curve p > p_c is drawn.
        """
        self._need_data()
        d = self.data
        fig, ax = plt.subplots(figsize=(7, 4.8))
        curves = {}
        if self.p_c is not None:
            curves[self._re_label()] = re_fraction(re_mask(d.p, d.E, d.n, d.T, p_c=self.p_c), d.weight)
        else:
            curves["isotropic $p_c$"] = re_fraction(re_mask(d.p, d.E, d.n, d.T), d.weight)
            if xi_dependent:
                curves[r"$\xi$-dependent $p_c$"] = re_fraction(re_mask(d.p, d.E, d.n, d.T, xi=d.xi), d.weight)
        for label, y in curves.items():
            ax.plot(self._times_ms(), y, marker="+", ms=6, lw=1.2, label=label)
            logger.info("RE fraction (%s) at the end: %.3e", label, y[-1])
        ax.set_yscale("log")
        ax.set_ylim(vmin, vmax)
        ax.set_xlabel("Time [ms]")
        ax.set_ylabel(r"$N_{RE} / N_0$")
        ax.legend(loc="best")
        return self._save(fig, "RE_fraction_time")

    def plot_re_current(self, I_p=None, light_speed=False, seed_threshold=None, vmin=None, vmax=None, log=True):
        """
        RE current |I_RE(t)|. Needs compute_re=True.

        Args:
            I_p: plasma current [A] drawn as a reference line.
            light_speed: recompute I_RE assuming v = c along the field for every RE marker.
            seed_threshold: seed current [A] separating the shaded regions "final I_RE above / below 1 MA".
        """
        self._need_pdfs()
        if self.pdfs.weight_re is None:
            raise ValueError("plot_re_current needs compute_re=True in build_pdfs")
        d = self.data
        I = self.pdfs.I_re if not light_speed else re_current(d.p, self.pdfs.weight_re, self.R0, light_speed=True)
        t = self._times_ms()

        fig, ax = plt.subplots(figsize=(7, 4.8))
        ax.plot(t, np.abs(I), color="tab:blue", label=r"$v = c$" if light_speed else self._re_label())
        if I_p is not None:
            ax.axhline(I_p, ls="--", color="black", label=rf"$I_p = {I_p*1e-6:g}$ MA")
        if seed_threshold is not None:
            top = max(vmax or 0, I_p or 0, np.nanmax(np.abs(I)) * 10)
            ax.axhspan(seed_threshold, top, color="tab:red", alpha=0.15, hatch="\\")
            ax.axhspan(1e-30, seed_threshold, color="tab:green", alpha=0.15, hatch="/")
            ax.axhline(seed_threshold, ls=":", color="white")
            ax.text(0.98, 0.88, r"$I_{RE,final} > 1$ MA", transform=ax.transAxes, ha="right", va="top")
            ax.text(0.98, 0.03, r"$I_{RE,final} < 1$ MA", transform=ax.transAxes, ha="right", va="bottom")
        if log:
            ax.set_yscale("log")
        ax.set_ylim(vmin, vmax)
        ax.set_xlim(t.min(), t.max())
        ax.set_xlabel("Time [ms]")
        ax.set_ylabel(r"$|I_{RE}|$ [A]")
        ax.legend(loc="upper left")
        logger.info("Final RE current: %.3f kA", abs(I[-1]) * 1e-3)
        return self._save(fig, "RE_current_time")

    def plot_fields(self, every=100, first=0, ylims=None):
        """n, E, T and the critical momentum seen by one marker out of `every`, versus time."""
        self._need_data()
        d = self.data
        E_c = critical_field(d.n, d.T)
        p_c = critical_momentum(d.E, E_c, xi=d.xi)
        fields = [(d.n, r"$n_e$ [m$^{-3}$]"), (d.E, r"$E_\parallel$ [V/m]"),
                  (d.T, r"$T_e$ [eV]"), (p_c, r"$p_c(\xi)$ [$m_e c$]")]
        ylims = ylims or [None] * 4
        idx = np.arange(first, d.nmark, every)
        colors = plt.get_cmap("turbo")(np.linspace(0, 1, len(idx)))

        fig, axs = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
        for ax, (F, ylabel), yl in zip(axs.ravel(), fields, ylims):
            for c, i in zip(colors, idx):
                ax.plot(self._times_ms(), F[:, i], color=c, lw=0.5, alpha=0.8)
            ax.set_ylabel(ylabel)
            ax.set_ylim(yl)
        for ax in axs[1]:
            ax.set_xlabel("Time [ms]")
        fig.suptitle(f"Fields seen by {len(idx)} markers")
        return self._save(fig, "fields_time")

    def plot_momentum(self, every=100, first=0, vmin=None, vmax=None, log=False):
        """Momentum of one marker out of `every`, versus time."""
        self._need_data()
        d = self.data
        idx = np.arange(first, d.nmark, every)
        colors = plt.get_cmap("turbo")(np.linspace(0, 1, len(idx)))
        fig, ax = plt.subplots(figsize=(8, 5))
        for c, i in zip(colors, idx):
            ax.plot(self._times_ms(), d.p[:, i], color=c, lw=0.8, alpha=0.8)
        if self.p_c is not None:
            ax.axhline(self.p_c, ls="--", color="white", label=rf"$p_c = {self.p_c:g}$")
            ax.legend(loc="best")
        if log:
            ax.set_yscale("log")
        ax.set_ylim(vmin, vmax)
        ax.set_xlabel("Time [ms]")
        ax.set_ylabel(r"$p$ [$m_e c$]")
        ax.set_title(f"Momentum of {len(idx)} markers")
        return self._save(fig, "momentum_time")

    def plot_trajectories(self, particles=50, t_range=None, distinct_re=False, limiter=True,
                          animate=False, fps=10, s=None):
        """
        Poloidal and top views of marker positions. `particles` is a count (evenly sampled) or an index array.
        distinct_re=True colours RE markers in red (needs compute_re=True). animate=True saves a GIF.
        """
        self._need_data()
        mask = None
        if distinct_re:
            if self.pdfs is None or self.pdfs.mask_re is None:
                logger.warning("distinct_re needs compute_re=True in build_pdfs, ignored")
            else:
                mask = self.pdfs.mask_re
        kw = dict(geom=self.geom, particles=particles, t_range=t_range, mask=mask, limiter=limiter)
        if s is not None:
            kw["s"] = s
        if animate:
            return animate_trajectories(self.data, self._path("trajectories", "gif"), fps=fps, **kw)
        fig = plot_trajectories(self.data, **kw)
        return self._save(fig, "trajectories")

    def plot_weights(self, log=False):
        """
        Weight budget versus time: total marker weight, cumulative lost weight (all and RE),
        and the weight removed by the reweighting scheme. Needs a losses file.
        """
        self._need_data()
        d = self.data
        if not d.has_losses:
            raise ValueError("plot_weights needs a losses file")
        W_tot = np.sum(d.weight, axis=1)
        W_lost = losses_cumulative_weight(d.t_loss, d.w_loss, d.t)
        W_reweight = (W_tot[0] - W_tot) - W_lost
        t = self._times_ms()

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(t, W_tot, color="white", label="Total weight")
        ax.plot(t, W_lost, color="tab:blue", label="Cumulative lost weight")
        if self.p_c is not None:
            W_lost_re = losses_cumulative_weight(d.t_loss, d.w_loss, d.t, p_loss=d.p_loss, p_c=self.p_c)
            ax.plot(t, W_lost_re, color="tab:purple", label=f"Cumulative lost weight, {self._re_label()}")
        ax.plot(t, W_reweight, color="tab:red", label="Weight removed by reweighting")
        if log:
            ax.set_yscale("log")
        ax.set_xlabel("Time [ms]")
        ax.set_ylabel("Weight [electrons]")
        ax.legend(loc="best")
        logger.info("Lost markers: %d", d.n_lost)
        return self._save(fig, "weights_time")
