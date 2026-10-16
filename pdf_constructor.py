"""
pdf_constructor.py

Build the set of distribution functions of one run from the marker data:
    * f_all:    every marker
    * f_re:     runaway markers only (optional)
    * f_losses: markers lost at the wall (optional, cumulative in time)
    * f_mj:     reference Maxwell-Juttner distribution in |p| (1D, for the f(p) plot)

A full-resolution 5D histogram (p, xi, R, Z, phi) is too large in memory, so two modes exist:
    space_resolved=True   fine (R, Z, phi) grid, p and xi reduced to a single bin each
    space_resolved=False  fine (p, xi) grid, R and Z reduced to a single bin each, no phi

Original implementation: Louis Puel (2024). Cleaned and restructured in 2026.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from class_pdf import MarkerPDF, losses_cumulative_weight, re_current, re_mask
from import_simu_data import MarkerData

logger = logging.getLogger("hottail")


@dataclass
class PDFSet:
    f_all: MarkerPDF
    f_mj: MarkerPDF | None = None
    f_re: MarkerPDF | None = None
    f_losses: MarkerPDF | None = None
    mask_re: np.ndarray | None = None     # (time, marker) RE flag
    weight_re: np.ndarray | None = None   # (time, marker) weights of RE markers, zero otherwise
    I_re: np.ndarray | None = None        # (time,) RE current [A]

    def get(self, which):
        """PDF by name: 'all', 're' or 'losses'."""
        pdf = {"all": self.f_all, "re": self.f_re, "losses": self.f_losses}.get(which)
        if which not in ("all", "re", "losses"):
            raise ValueError("which must be 'all', 're' or 'losses'")
        if pdf is None:
            raise ValueError(f"The '{which}' PDF was not built (see build_pdfs options)")
        return pdf


def _coarse(x, n=1, pad=0.05):
    """Edges of `n` bins covering the whole range of x, with a small relative margin."""
    lo, hi = np.nanmin(x), np.nanmax(x)
    span = max(hi - lo, 1e-12)
    return np.linspace(lo - pad * span, hi + pad * span, n + 1)


def _grids_for(grids, space_resolved, samples):
    """
    Dimension names, grids and data lists for one PDF.
    `samples` is a dict name -> array (2D for markers, 1D for losses).
    """
    if space_resolved:
        dims = ["p", "xi", "R", "Z", "phi"]
        g = [_coarse(samples["p"]), _coarse(samples["xi"]), grids["R"], grids["Z"], grids["phi"]]
    else:
        dims = ["p", "xi", "R", "Z"]
        g = [grids["p"], grids["xi"], _coarse(samples["R"]), _coarse(samples["Z"])]
    return dims, g, [samples[d] for d in dims]


def build_pdfs(data: MarkerData, grids, R0, p_c=None, space_resolved=False,
               compute_re=False, xi_dependent=False,
               compute_losses=False, re_losses=False,
               is_pdf=True, spatial_jacobian=True, momentum_jacobian=False,
               gauss_smooth=None, Te_mj=None) -> PDFSet:
    """
    Build every requested PDF of a run.

    Args:
        data:   MarkerData from load_jorek().
        grids:  dict of bin edges for "p", "xi", "R", "Z", "phi".
        R0:     major radius [m], used for the RE current.
        p_c:    prescribed RE momentum threshold [m_e c]; None uses the local critical momentum.
        space_resolved: see module docstring.
        compute_re:     build the RE PDF and the RE current.
        xi_dependent:   pitch-dependent RE criterion (ignored when p_c is given).
        compute_losses: build the cumulative PDF of lost markers (needs a losses file).
        re_losses:      keep only lost markers with p > p_c in the losses PDF.
        is_pdf:         normalise to the marker weight (True) or keep raw histograms (False).
        spatial_jacobian, momentum_jacobian: store f per R dR and per p^2 dp respectively.
        gauss_smooth:   Gaussian smoothing as a fraction of the number of bins.
        Te_mj:          temperature [eV] of the reference Maxwell-Juttner; None skips it.
    """
    out = PDFSet(f_all=None)
    N0 = np.sum(data.weight[0])
    common = dict(is_pdf=is_pdf, momentum_jacobian=momentum_jacobian, spatial_jacobian=spatial_jacobian)

    # ---- all markers
    samples = dict(p=data.p, xi=data.xi, R=data.R, Z=data.Z, phi=data.phi)
    dims, g, arrays = _grids_for(grids, space_resolved, samples)
    out.f_all = MarkerPDF.from_markers(dims, g, data.t, arrays, data.weight, gauss_smooth=gauss_smooth, **common)
    out.f_all.normalize(np.sum(data.weight, axis=1))
    logger.info("Total weight at t0: %.4e | integral of f_all at t0: %.4e", N0, out.f_all.int_f[0])

    # ---- runaway markers
    if compute_re:
        out.mask_re = re_mask(data.p, data.E, data.n, data.T, p_c=p_c, xi=data.xi if xi_dependent else None)
        out.weight_re = np.where(out.mask_re, data.weight, 0.0)
        frac = out.mask_re.mean(axis=1)
        logger.info("RE markers: %.2f %% at t0, %.2f %% at the end", 100 * frac[0], 100 * frac[-1])
        out.f_re = MarkerPDF.from_markers(dims, g, data.t, arrays, out.weight_re, gauss_smooth=gauss_smooth, **common)
        out.f_re.normalize(np.sum(out.weight_re, axis=1))
        out.I_re = re_current(data.p, out.weight_re, R0, xi=data.xi)
        logger.info("RE current at the end: %.3f kA", abs(out.I_re[-1]) * 1e-3)

    # ---- lost markers
    if compute_losses:
        if not data.has_losses:
            raise ValueError("compute_losses=True requires a losses file")
        p_c_losses = p_c if re_losses else None
        samples = dict(p=data.p_loss, xi=data.xi_loss, R=data.R_loss, Z=data.Z_loss, phi=data.phi_loss)
        dims_l, g_l, arrays_l = _grids_for(grids, space_resolved, samples)
        out.f_losses = MarkerPDF.from_losses(dims_l, g_l, data.t, arrays_l, data.t_loss, data.w_loss,
                                             cumulative=True, p_c=p_c_losses, **common)
        W_lost = losses_cumulative_weight(data.t_loss, data.w_loss, data.t, p_loss=data.p_loss, p_c=p_c_losses)
        out.f_losses.normalize(W_lost[1:])
        W_all = losses_cumulative_weight(data.t_loss, data.w_loss, data.t)
        reweight = (N0 - np.sum(data.weight[-1])) - W_all[-1]
        logger.info("Losses: %d markers, weight lost %.4e (%.2f %% of N0), weight removed by reweighting %.4e",
                 data.n_lost, W_all[-1], 100 * W_all[-1] / N0, reweight)

    # ---- Maxwell-Juttner reference
    if Te_mj is not None:
        out.f_mj = MarkerPDF.maxwell_juttner(np.linspace(1e-4, grids["p"][-1], 5000), Te_mj, N=N0)

    return out
