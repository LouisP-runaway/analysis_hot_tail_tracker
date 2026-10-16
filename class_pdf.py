"""
class_pdf.py

Marker distribution functions and physics helpers for hot-tail runaway electron (RE) analysis.

Contents
    1. Physical constants, plot style and tokamak geometry
    2. RE physics: Coulomb logarithm, critical field and momentum, RE selection, RE current
    3. MarkerPDF: histogram-based distribution on any subset of (p, xi, R, Z, phi),
       with integration, normalisation, 2D / poloidal / top-view plots and animations
    4. Marker trajectory plots (static and animated)

Original implementation: Louis Puel (2024). Cleaned and restructured in 2026.
"""

from __future__ import annotations

import logging
import os
import time as _time

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import LogNorm, Normalize
from scipy.ndimage import gaussian_filter, zoom

logger = logging.getLogger("hottail")
logging.getLogger("matplotlib.animation").setLevel(logging.WARNING)

# =========================================================================== #
#  1. Constants, style, geometry
# =========================================================================== #
EPS_0 = 8.8541878128e-12          # vacuum permittivity [F/m]
E_CHARGE = 1.602176634e-19        # elementary charge [C]
M_E = 9.1093837015e-31            # electron mass [kg]
C_LIGHT = 299792458.0             # speed of light [m/s]
E0_E_EV = 510998.95               # electron rest energy [eV]
# E_c = EC_CONSTANT * n_e * ln(Lambda), with E_c the Connor-Hastie critical field [V/m]
EC_CONSTANT = E_CHARGE**3 / (4 * np.pi * EPS_0**2 * M_E * C_LIGHT**2)

DIM_LABELS = {
    "p": r"$p$ [$m_e c$]", "xi": r"$\xi$", "R": r"$R$ [m]", "Z": r"$Z$ [m]", "phi": r"$\varphi$ [rad]",
}
DEFAULT_CMAP = "nipy_spectral"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GEOMETRY_DIR = os.path.join(BASE_DIR, "geometries")
TOKAMAKS = {
    "ITER":   dict(R0=6.3692, Z0=0.5, geometry="ITER_namelist_JOREK_2"),
    "DIII-D": dict(R0=1.7,    Z0=0.0, geometry="DIIID_namelist_JOREK"),
}


def set_style(font_size=12):
    """Consistent matplotlib style for all figures of the package."""
    plt.style.use('dark_background') # dark background
    #plt.style.use('default')         # standard white background

    plt.rcParams.update({
        "font.size": font_size,
        "axes.titlesize": font_size,
        "axes.labelsize": font_size,
        "legend.fontsize": font_size - 2,
        "legend.framealpha": 0.9,
        "lines.linewidth": 1.8,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
        "figure.constrained_layout.use": True,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "image.cmap": DEFAULT_CMAP,
    })


def _read_boundary(path, key):
    """
    Parse a JOREK namelist and return (R, Z) of the lines containing `key`
    ("R_boundary" or "R_limiter"). Values are the tokens in E-notation.
    """
    R, Z = [], []
    with open(path, "r") as fh:
        for line in fh:
            if key not in line:
                continue
            vals = [float(tok.rstrip(",")) for tok in line.split() if "E" in tok]
            if len(vals) >= 2:
                R.append(vals[0])
                Z.append(vals[1])
    return np.array(R), np.array(Z)


def get_geometry(tokamak, geometry_dir=None):
    """
    Return dict(R0, Z0, boundary=(R, Z), limiter=(R, Z) or None) for a tokamak name.
    Missing geometry files give boundary=None with a warning (plots are then drawn without wall).
    """
    if tokamak not in TOKAMAKS:
        raise ValueError(f"Unknown tokamak '{tokamak}'. Available: {list(TOKAMAKS)}")
    cfg = TOKAMAKS[tokamak]
    path = os.path.join(geometry_dir or GEOMETRY_DIR, cfg["geometry"])
    geom = dict(R0=cfg["R0"], Z0=cfg["Z0"], boundary=None, limiter=None)
    if not os.path.isfile(path):
        logger.warning("Geometry file not found: %s (wall not drawn)", path)
        return geom
    Rb, Zb = _read_boundary(path, "R_boundary")
    Rl, Zl = _read_boundary(path, "R_limiter")
    geom["boundary"] = (Rb, Zb) if Rb.size else None
    geom["limiter"] = (Rl, Zl) if Rl.size else None
    return geom


def draw_geometry(ax, geom, limiter=False, view="poloidal"):
    """Draw the wall (white) and optionally the limiter (red) in poloidal (R, Z) or top (X, Y) view."""
    if geom is None:
        return
    curves = [(geom.get("boundary"), "white")]
    if limiter:
        curves.append((geom.get("limiter"), "tab:red"))
    theta = np.linspace(0, 2 * np.pi, 200)
    for curve, color in curves:
        if curve is None:
            continue
        R, Z = curve
        if view == "poloidal":
            ax.plot(R, Z, color=color, lw=1.5)
        else:
            for r in (R.min(), R.max()):
                ax.plot(r * np.cos(theta), r * np.sin(theta), color=color, lw=1.5)


# =========================================================================== #
#  2. Runaway electron physics
# =========================================================================== #
def coulomb_log_relativistic(ne, Te):
    """Relativistic Coulomb logarithm, Hesslow et al., J. Plasma Phys. 84 (2018), Eq. (2.9). ne [m^-3], Te [eV]."""
    return 14.6 + 0.5 * np.log(Te / (ne * 1e-20))


def coulomb_log_thermal(ne, Te):
    """Thermal Coulomb logarithm, Hesslow et al., J. Plasma Phys. 84 (2018), Eq. (2.7). ne [m^-3], Te [eV]."""
    return 14.9 - 0.5 * np.log(ne * 1e-20) + np.log(Te / 1e3)


def critical_field(ne, Te, log_c=None):
    """Connor-Hastie critical electric field E_c [V/m]."""
    if log_c is None:
        log_c = coulomb_log_relativistic(ne, Te)
    return EC_CONSTANT * ne * log_c


def critical_momentum(E, E_c, xi=None):
    """
    Critical momentum p_c [m_e c] above which an electron runs away.

    Isotropic:      p_c = 1 / sqrt(E/E_c - 1)
    Pitch-dependent: p_c = 1 / sqrt((1 - xi) E / (2 E_c) - 1)
    Returns +inf where the field is below threshold (no runaway possible).
    """
    ratio = E / E_c if xi is None else (1.0 - xi) * E / (2.0 * E_c)
    with np.errstate(divide="ignore", invalid="ignore"):
        p_c = np.where(ratio > 1.0, 1.0 / np.sqrt(np.abs(ratio - 1.0)), np.inf)
    return p_c


def re_mask(p, E, n, T, p_c=None, xi=None):
    """
    Boolean (time, marker) array flagging markers counted as runaway electrons.

    Args:
        p, E, n, T: (time, marker) marker momentum [m_e c] and local fields.
        p_c: prescribed momentum threshold [m_e c]. If None, the local critical momentum is used.
        xi:  (time, marker) pitch. If given (and p_c is None), the pitch-dependent criterion is used.
    """
    if p_c is not None:
        return p >= p_c
    E_c = critical_field(n, T)
    return p >= critical_momentum(E, E_c, xi=xi)


def re_fraction(mask, weight):
    """Weighted RE fraction N_RE(t) / N_0, with N_0 the initial total weight."""
    return np.sum(weight * mask, axis=1) / np.sum(weight[0])


def re_current(p, weight_re, R0, xi=None, light_speed=False):
    """
    Total toroidal current carried by the RE markers [A]: I = e / (2 pi R0) * sum_i w_i v_par,i.

    Args:
        p:         (time, marker) momentum [m_e c].
        weight_re: (time, marker) weights, zero for non-RE markers.
        R0:        major radius [m].
        xi:        (time, marker) pitch. If None the full speed is taken as parallel.
        light_speed: if True, every RE marker moves at c along the field.
    """
    if light_speed:
        v = np.full_like(p, C_LIGHT)
    else:
        v = C_LIGHT * np.sqrt(1.0 - 1.0 / (1.0 + p**2))
        if xi is not None:
            v = v * xi
    return np.sum(weight_re * np.nan_to_num(v), axis=1) * E_CHARGE / (2 * np.pi * R0)


def relativistic_parameters(K=None, beta=None, p=None):
    """Convert between kinetic energy K [eV], normalised speed beta and momentum p [m_e c]. Give exactly one."""
    given = [x is not None for x in (K, beta, p)]
    if sum(given) != 1:
        raise ValueError("Provide exactly one of K, beta or p")
    if p is not None:
        gamma = np.sqrt(1 + p**2)
    elif K is not None:
        gamma = 1 + K / E0_E_EV
    else:
        gamma = 1 / np.sqrt(1 - beta**2)
    K = (gamma - 1) * E0_E_EV
    beta = np.sqrt(1 - 1 / gamma**2)
    p = gamma * beta
    return K, beta, p


def losses_cumulative_weight(t_loss, w_loss, t_grid, p_loss=None, p_c=None):
    """
    Cumulative lost weight W(t_k) = sum of w_loss for markers lost at t_loss <= t_grid[k].
    If p_c is given, only markers lost with p_loss > p_c are counted.
    """
    order = np.argsort(t_loss)
    t_sorted, w_sorted = t_loss[order], w_loss[order]
    if p_c is not None and p_loss is not None:
        w_sorted = np.where(p_loss[order] > p_c, w_sorted, 0.0)
    cum = np.concatenate([[0.0], np.cumsum(w_sorted)])
    return cum[np.searchsorted(t_sorted, t_grid, side="right")]


# =========================================================================== #
#  3. MarkerPDF
# =========================================================================== #
class MarkerPDF:
    """
    Time-dependent distribution f(t, x_1, ..., x_n) on a rectangular grid of a subset of (p, xi, R, Z, phi).

    Attributes:
        dims:    list of dimension names, e.g. ["p", "xi", "R", "Z", "phi"].
        grid:    list of bin edges, one array per dimension.
        centers: list of bin centres.
        time:    (nt,) times [s].
        f:       (nt, n_1, ..., n_n) distribution.
        is_pdf:  True if f is a density (integral = number of electrons), False if a raw histogram.
        momentum_jacobian: f stored per p^2 dp (so the p measure is p^2 dp).
        spatial_jacobian:  f stored per R dR (so the R measure is R dR, i.e. a true volume density).
        int_f:   (nt,) integral of f over all dimensions, set by normalize().
    """

    def __init__(self, dims, grids, time, f=None, is_pdf=True):
        if len(dims) != len(grids):
            raise ValueError("dims and grids must have the same length")
        self.dims = list(dims)
        self.grid = [np.asarray(g, dtype=float) for g in grids]
        self.centers = [0.5 * (g[:-1] + g[1:]) for g in self.grid]
        self.time = np.asarray(time, dtype=float)
        self.f = f
        self.is_pdf = is_pdf
        self.momentum_jacobian = False
        self.spatial_jacobian = False
        self.int_f = None

    # ------------------------------------------------------------------ builders
    @classmethod
    def from_markers(cls, dims, grids, time, data, weight, is_pdf=True,
                     momentum_jacobian=False, spatial_jacobian=True, gauss_smooth=None):
        """
        Weighted histogram of the markers at each time.

        Args:
            data:   list of (time, marker) arrays aligned with `dims`.
            weight: (time, marker) marker weights.
            gauss_smooth: optional Gaussian smoothing, as a fraction of the number of bins per dimension.
        """
        pdf = cls(dims, grids, time, is_pdf=is_pdf)
        nt = pdf.time.shape[0]
        pdf.f = np.zeros([nt] + [c.shape[0] for c in pdf.centers])
        t0 = _time.perf_counter()
        for i in range(nt):
            pdf.f[i] = np.histogramdd([d[i] for d in data], bins=pdf.grid,
                                      weights=weight[i], density=is_pdf)[0]
        pdf._finalise(momentum_jacobian, spatial_jacobian, gauss_smooth)
        logger.info("Built %dD marker PDF (%s) at %d times in %.1f s",
                 len(dims), ", ".join(dims), nt, _time.perf_counter() - t0)
        return pdf

    @classmethod
    def from_losses(cls, dims, grids, t_grid, data, t_loss, w_loss, cumulative=True, p_c=None,
                    is_pdf=True, momentum_jacobian=False, spatial_jacobian=True):
        """
        Histogram of the lost markers per time bin of `t_grid` (nt edges give nt-1 frames).

        Args:
            data:   list of 1D arrays (one value per lost marker) aligned with `dims`.
            t_loss, w_loss: loss time and weight of each lost marker.
            cumulative: if True, frame i counts every marker lost before t_grid[i+1].
            p_c: if given, only markers lost with p > p_c are counted (data[0] must be p).
        """
        t_grid = np.asarray(t_grid)
        t_centres = 0.5 * (t_grid[:-1] + t_grid[1:])
        pdf = cls(dims, grids, t_centres, is_pdf=is_pdf)
        nt = t_centres.shape[0]
        pdf.f = np.zeros([nt] + [c.shape[0] for c in pdf.centers])
        sample = np.vstack(data).T
        base = np.ones_like(t_loss, dtype=bool) if p_c is None else (data[0] > p_c)
        t0 = _time.perf_counter()
        for i in range(nt):
            if cumulative:
                mask = base & (t_loss <= t_grid[i + 1])
            else:
                mask = base & (t_loss >= t_grid[i]) & (t_loss < t_grid[i + 1])
            if not np.any(mask):
                continue
            pdf.f[i] = np.histogramdd(sample[mask], bins=pdf.grid, weights=w_loss[mask], density=is_pdf)[0]
        pdf._finalise(momentum_jacobian, spatial_jacobian, None)
        logger.info("Built %dD losses PDF (%s) at %d times in %.1f s",
                 len(dims), ", ".join(dims), nt, _time.perf_counter() - t0)
        return pdf

    @classmethod
    def maxwell_juttner(cls, p_centres, Te_eV, N=1.0):
        """
        1D Maxwell-Juttner distribution in |p|: f(p) = A p^2 exp(-(gamma - 1) / theta), normalised to N electrons.
        """
        p_centres = np.asarray(p_centres, dtype=float)
        dp = np.diff(p_centres)
        edges = np.concatenate([[p_centres[0] - dp[0] / 2], p_centres[:-1] + dp / 2, [p_centres[-1] + dp[-1] / 2]])
        pdf = cls(["p"], [edges], np.array([0.0]), is_pdf=True)
        theta = Te_eV / E0_E_EV
        gamma = np.sqrt(1.0 + p_centres**2)
        pdf.f = (p_centres**2 * np.exp(-(gamma - 1.0) / theta))[None, :]
        pdf.normalize(N)
        return pdf

    def _finalise(self, momentum_jacobian, spatial_jacobian, gauss_smooth):
        """Apply optional smoothing and Jacobians after histogramming."""
        if gauss_smooth is not None:
            sigma = [0] + [gauss_smooth * c.shape[0] for c in self.centers]
            self.f = gaussian_filter(self.f, sigma=sigma, truncate=4)
        if self.is_pdf and momentum_jacobian and "p" in self.dims:
            self.scale_axis("p", 1.0 / self.centers[self.idx("p")]**2)
            self.momentum_jacobian = True
        if self.is_pdf and spatial_jacobian and "R" in self.dims:
            self.scale_axis("R", 1.0 / self.centers[self.idx("R")])
            self.spatial_jacobian = True

    # ------------------------------------------------------------------ basic properties
    @property
    def nt(self) -> int:
        return self.time.shape[0]

    @property
    def ndim(self) -> int:
        return len(self.dims)

    def idx(self, dim) -> int:
        """Index of a dimension given by name or integer."""
        if isinstance(dim, (int, np.integer)):
            return int(dim)
        if dim not in self.dims:
            raise ValueError(f"Dimension '{dim}' not in PDF dims {self.dims}")
        return self.dims.index(dim)

    def label(self, dim) -> str:
        return DIM_LABELS.get(self.dims[self.idx(dim)], self.dims[self.idx(dim)])

    def measure(self, dim):
        """Integration weights of one dimension: bin width times Jacobian (p^2 or R when stored)."""
        d = self.idx(dim)
        if not self.is_pdf:
            return np.ones(self.centers[d].shape[0])
        w = np.diff(self.grid[d])
        if self.dims[d] == "p" and self.momentum_jacobian:
            w = w * self.centers[d]**2
        if self.dims[d] == "R" and self.spatial_jacobian:
            w = w * self.centers[d]
        return w

    # ------------------------------------------------------------------ algebra
    def scale_axis(self, dim, factor):
        """Multiply f along one dimension by a vector (or a scalar)."""
        d = self.idx(dim)
        factor = np.asarray(factor, dtype=float)
        if factor.ndim == 0:
            self.f = self.f * factor
            return
        if factor.shape[0] != self.f.shape[d + 1]:
            raise ValueError(f"Vector length {factor.shape[0]} does not match dimension {self.dims[d]} "
                             f"({self.f.shape[d + 1]} bins)")
        shape = [1] * self.f.ndim
        shape[d + 1] = -1
        self.f = self.f * factor.reshape(shape)

    def integrate(self, dims):
        """Integrate f over the given dimensions (names or indices). Returns an array (nt, remaining dims...)."""
        axes = sorted({self.idx(d) for d in dims}, reverse=True)
        out = self.f
        for d in axes:
            w = self.measure(d)
            shape = [1] * out.ndim
            shape[d + 1] = -1
            out = np.sum(out * w.reshape(shape), axis=d + 1)
        return out

    def integrate_all(self):
        """(nt,) integral over every dimension (number of electrons when normalised)."""
        return self.integrate(range(self.ndim))

    def profile(self, dim):
        """(nt, n_dim) distribution integrated over every dimension except `dim`."""
        keep = self.idx(dim)
        return self.integrate([d for d in range(self.ndim) if d != keep])

    def normalize(self, target):
        """
        Rescale each time slice so that its integral equals `target` (scalar or (nt,) array).
        Time slices with zero integral are left at zero.
        """
        integral = self.integrate_all()
        target = np.broadcast_to(np.asarray(target, dtype=float), integral.shape)
        with np.errstate(divide="ignore", invalid="ignore"):
            factor = np.where(integral > 0, target / integral, 0.0)
        self.f = self.f * factor.reshape([-1] + [1] * self.ndim)
        self.int_f = self.integrate_all()

    # ------------------------------------------------------------------ 2D fields for plotting
    def field_2d(self, dim1, dim2, i_time):
        """Distribution at time index i_time integrated over every dimension except dim1 and dim2. Shape (n2, n1)."""
        d1, d2 = self.idx(dim1), self.idx(dim2)
        if d1 == d2:
            raise ValueError("dim1 and dim2 must differ")
        F = self.integrate([d for d in range(self.ndim) if d not in (d1, d2)])[i_time]
        return F if d1 < d2 else F.T   # (n1, n2), dim1 first

    def field_poloidal(self, i_time, density=False):
        """
        (nZ, nR) map at time index i_time, integrated over p, xi and phi.
        With density=True, return the toroidally averaged electron density [m^-3].
        """
        F = self.field_2d("R", "Z", i_time)     # (nR, nZ)
        if density:
            if not self.is_pdf:
                raise ValueError("density=True requires a normalised PDF (is_pdf=True)")
            R = self.centers[self.idx("R")]
            F = F / (2 * np.pi) if self.spatial_jacobian else F / (2 * np.pi * R[:, None])
        return F.T

    def field_topview(self, i_time):
        """(nphi, nR) map at time index i_time, integrated over p, xi and Z: line-integrated density [m^-2]."""
        F = self.field_2d("R", "phi", i_time)   # (nR, nphi)
        if self.is_pdf and not self.spatial_jacobian:
            F = F / self.centers[self.idx("R")][:, None]
        return F.T

    # ------------------------------------------------------------------ plots
    @staticmethod
    def _norm(C, log, vmin, vmax):
        """Colour normalisation. In log scale, vmin/vmax are in data units (not log10)."""
        if log:
            positive = C[np.isfinite(C) & (C > 0)]
            if vmin is None:
                vmin = positive.min() if positive.size else 1e-30
            if vmax is None:
                vmax = positive.max() if positive.size else 1.0
            return LogNorm(vmin=vmin, vmax=vmax)
        return Normalize(vmin=vmin, vmax=vmax)

    @staticmethod
    def _upsample_map(C, factor, log):
        """Linear upsampling of a (ny, nx) map for display, done in log10 space when log=True."""
        if factor <= 1:
            return C
        if log:
            with np.errstate(divide="ignore"):
                C_in = np.where(C > 0, np.log10(C), np.nan)
            floor = np.nanmin(C_in) - 10 if np.isfinite(C_in).any() else -100
            C_fine = zoom(np.nan_to_num(C_in, nan=floor), factor, order=1)
            return np.where(C_fine > floor + 5, 10.0**C_fine, 0.0)
        return zoom(C, factor, order=1)

    def _upsample(self, C, x_edges, y_edges, factor, log):
        """Upsampled map with the fine cell edges (x, y) tiling the same domain."""
        C_fine = self._upsample_map(C, factor, log)
        xf = np.linspace(x_edges[0], x_edges[-1], C_fine.shape[1] + 1)
        yf = np.linspace(y_edges[0], y_edges[-1], C_fine.shape[0] + 1)
        return xf, yf, C_fine

    def _mesh(self, ax, x_edges, y_edges, C, log, vmin, vmax, cmap, interp):
        """Draw a pcolormesh on the grid edges, after optional display upsampling."""
        cmap = plt.get_cmap(cmap or DEFAULT_CMAP).with_extremes(bad=plt.rcParams["axes.facecolor"])
        norm = self._norm(C, log, vmin, vmax)
        x_edges, y_edges, C = self._upsample(C, x_edges, y_edges, interp, log)
        C = np.ma.masked_less_equal(np.ma.masked_invalid(C), 0.0)   # empty bins are drawn white
        return ax.pcolormesh(x_edges, y_edges, C, norm=norm, cmap=cmap, shading="flat")

    def plot_2d(self, dim1, dim2, i_time=0, ax=None, log=True, vmin=None, vmax=None,
                cmap=None, interp=1, cbar_label=None):
        """
        Map of f integrated over every dimension except (dim1, dim2), at time index i_time.

        Args:
            log:  logarithmic colour scale. vmin/vmax are then in data units (e.g. 1e10, 1e24).
            interp: display upsampling factor (1 = none).
        Returns: fig, ax, pcm
        """
        fig, ax = _fig_ax(ax, figsize=(6.5, 5))
        d1, d2 = self.idx(dim1), self.idx(dim2)
        C = self.field_2d(d1, d2, i_time).T                         # (n2, n1) for pcolormesh
        pcm = self._mesh(ax, self.grid[d1], self.grid[d2], C, log, vmin, vmax, cmap, interp)
        fig.colorbar(pcm, ax=ax, label=cbar_label or ("PDF" if self.is_pdf else "counts"))
        ax.set_xlabel(self.label(d1))
        ax.set_ylabel(self.label(d2))
        ax.set_title(f"t = {self.time[i_time]*1e3:.3f} ms")
        ax.grid(False)
        return fig, ax, pcm

    def plot_poloidal(self, i_time=0, ax=None, log=True, vmin=None, vmax=None, cmap=None,
                      interp=1, density=False, geom=None, limiter=False):
        """
        Poloidal (R, Z) map at time index i_time, integrated over p, xi and phi.

        Args:
            density: plot the toroidally averaged density [m^-3] instead of the integrated PDF.
            geom:    geometry dict from get_geometry() to overlay the wall (and limiter).
        Returns: fig, ax, pcm
        """
        fig, ax = _fig_ax(ax, figsize=(5.5, 7))
        C = self.field_poloidal(i_time, density=density)
        pcm = self._mesh(ax, self.grid[self.idx("R")], self.grid[self.idx("Z")], C, log, vmin, vmax, cmap, interp)
        draw_geometry(ax, geom, limiter=limiter, view="poloidal")
        fig.colorbar(pcm, ax=ax, label=r"$\langle n \rangle_\varphi$ [m$^{-3}$]" if density else "PDF")
        ax.set_xlabel(DIM_LABELS["R"])
        ax.set_ylabel(DIM_LABELS["Z"])
        ax.set_title(f"t = {self.time[i_time]*1e3:.3f} ms")
        ax.set_aspect("equal")
        ax.grid(False)
        return fig, ax, pcm

    def plot_topview(self, i_time=0, ax=None, log=True, vmin=None, vmax=None, cmap=None,
                     interp=1, geom=None, limiter=False):
        """Top view (X, Y) map at time index i_time, integrated over p, xi and Z. Returns: fig, ax, pcm"""
        fig, ax = _fig_ax(ax, figsize=(6.5, 6))
        C = self.field_topview(i_time)                               # (nphi, nR)
        R_edges = self.grid[self.idx("R")]
        phi_edges = self.grid[self.idx("phi")]
        norm = self._norm(C, log, vmin, vmax)
        R_edges, phi_edges, C = self._upsample(C, R_edges, phi_edges, interp, log)
        RR, PP = np.meshgrid(R_edges, phi_edges)
        cmap_obj = plt.get_cmap(cmap or DEFAULT_CMAP).with_extremes(bad=plt.rcParams["axes.facecolor"])
        C = np.ma.masked_less_equal(np.ma.masked_invalid(C), 0.0)   # empty bins are drawn white
        pcm = ax.pcolormesh(RR * np.cos(PP), RR * np.sin(PP), C, norm=norm, cmap=cmap_obj, shading="flat")
        draw_geometry(ax, geom, limiter=limiter, view="top")
        fig.colorbar(pcm, ax=ax, label=r"$\int n\, dZ$ [m$^{-2}$]")
        ax.set_xlabel(r"$X$ [m]")
        ax.set_ylabel(r"$Y$ [m]")
        ax.set_title(f"t = {self.time[i_time]*1e3:.3f} ms")
        ax.set_aspect("equal")
        ax.grid(False)
        return fig, ax, pcm

    def animate(self, kind, path, fps=10, frames=None, **kwargs):
        """
        Save a GIF of plot_2d / plot_poloidal / plot_topview over time.

        Args:
            kind:   "2d" (needs dim1, dim2 in kwargs), "poloidal" or "topview".
            path:   output file (.gif).
            frames: iterable of time indices (default: all).
            kwargs: forwarded to the plot function. Give vmin and vmax so that the colour scale is fixed.
        """
        kwargs = dict(kwargs)
        kwargs.pop("i_time", None)
        if kind == "2d":
            dim1, dim2 = kwargs.pop("dim1"), kwargs.pop("dim2")
            fig, ax, pcm = self.plot_2d(dim1, dim2, 0, **kwargs)
            get = lambda i: self.field_2d(dim1, dim2, i).T
        elif kind == "poloidal":
            fig, ax, pcm = self.plot_poloidal(0, **kwargs)
            get = lambda i: self.field_poloidal(i, density=kwargs.get("density", False))
        elif kind == "topview":
            fig, ax, pcm = self.plot_topview(0, **kwargs)
            get = lambda i: self.field_topview(i)
        else:
            raise ValueError("kind must be '2d', 'poloidal' or 'topview'")
        interp = kwargs.get("interp", 1)
        log_scale = kwargs.get("log", True)
        if kwargs.get("vmin") is None or kwargs.get("vmax") is None:
            logger.warning("animate: vmin/vmax not given, the colour scale of the first frame is kept for all frames")

        def update(i):
            C = self._upsample_map(get(i), interp, log_scale)
            pcm.set_array(np.ma.masked_less_equal(np.ma.masked_invalid(C), 0.0).ravel())
            ax.set_title(f"t = {self.time[i]*1e3:.3f} ms")
            return (pcm,)

        frames = list(range(self.nt)) if frames is None else list(frames)
        anim = FuncAnimation(fig, update, frames=frames, blit=False)
        anim.save(path, writer=PillowWriter(fps=fps))
        plt.close(fig)
        logger.info("Saved animation %s (%d frames)", path, len(frames))


def _fig_ax(ax, figsize):
    if ax is None:
        return plt.subplots(figsize=figsize)
    return ax.get_figure(), ax


# =========================================================================== #
#  4. Marker trajectories
# =========================================================================== #
def _select_markers(nmark, particles):
    """Marker indices: an integer gives that many evenly spaced markers, an array gives explicit indices."""
    if np.ndim(particles) == 0:
        n = min(int(particles), nmark)
        return np.linspace(0, nmark - 1, n).astype(int)
    return np.asarray(particles, dtype=int)


def _trajectory_axes(geom, limiter, figsize):
    fig, (ax_pol, ax_top) = plt.subplots(1, 2, figsize=figsize)
    draw_geometry(ax_pol, geom, limiter=limiter, view="poloidal")
    draw_geometry(ax_top, geom, limiter=limiter, view="top")
    ax_pol.set_xlabel(DIM_LABELS["R"]); ax_pol.set_ylabel(DIM_LABELS["Z"]); ax_pol.set_aspect("equal")
    ax_top.set_xlabel(r"$X$ [m]");      ax_top.set_ylabel(r"$Y$ [m]");      ax_top.set_aspect("equal")
    return fig, ax_pol, ax_top


def plot_trajectories(data, geom=None, particles=50, t_range=None, mask=None, limiter=False,
                      s=4, alpha=0.7, figsize=(11, 5), cmap="turbo"):
    """
    Static poloidal and top views of marker trajectories, one colour per marker.
    With `mask` (time, marker) boolean, RE points are red and the others grey.

    Args:
        particles: number of markers (evenly sampled) or explicit index array.
        t_range:   (i_start, i_end) time-index range (inclusive), default all.
    Returns: fig
    """
    idx = _select_markers(data.nmark, particles)
    i0, i1 = (0, data.nt - 1) if t_range is None else t_range
    ti = np.arange(i0, (data.nt - 1 if i1 is None else i1) + 1)
    R, Z, phi = data.R[np.ix_(ti, idx)], data.Z[np.ix_(ti, idx)], data.phi[np.ix_(ti, idx)]
    X, Y = R * np.cos(phi), R * np.sin(phi)

    fig, ax_pol, ax_top = _trajectory_axes(geom, limiter, figsize)
    if mask is None:
        colors = plt.get_cmap(cmap)(np.linspace(0, 1, len(idx)))
        c = np.repeat(colors[None, :, :], len(ti), axis=0).reshape(-1, 4)
    else:
        m = mask[np.ix_(ti, idx)].ravel()
        c = np.where(m[:, None], [[0.85, 0.1, 0.1, 1.0]], [[0.6, 0.6, 0.6, 0.5]])
    ax_pol.scatter(R.ravel(), Z.ravel(), s=s, c=c, alpha=alpha, edgecolors="none")
    ax_top.scatter(X.ravel(), Y.ravel(), s=s, c=c, alpha=alpha, edgecolors="none")
    fig.suptitle(f"{len(idx)} markers, t = {data.t[ti[0]]*1e3:.3f} to {data.t[ti[-1]]*1e3:.3f} ms")
    return fig


def animate_trajectories(data, path, geom=None, particles=50, t_range=None, mask=None, limiter=False,
                         s=8, alpha=0.8, figsize=(11, 5), fps=10, cmap="turbo"):
    """Animated version of plot_trajectories, saved as a GIF at `path`."""
    idx = _select_markers(data.nmark, particles)
    i0, i1 = (0, data.nt - 1) if t_range is None else t_range
    frames = list(range(i0, (data.nt - 1 if i1 is None else i1) + 1))

    fig, ax_pol, ax_top = _trajectory_axes(geom, limiter, figsize)
    base = plt.get_cmap(cmap)(np.linspace(0, 1, len(idx)))
    sc_pol = ax_pol.scatter(data.R[0, idx], data.Z[0, idx], s=s, alpha=alpha, edgecolors="none")
    sc_top = ax_top.scatter(data.R[0, idx], data.R[0, idx] * 0, s=s, alpha=alpha, edgecolors="none")
    if geom is None or geom.get("boundary") is None:
        ax_pol.set_xlim(np.nanmin(data.R), np.nanmax(data.R)); ax_pol.set_ylim(np.nanmin(data.Z), np.nanmax(data.Z))
        r = np.nanmax(data.R); ax_top.set_xlim(-r, r); ax_top.set_ylim(-r, r)

    def update(i):
        R, Z, phi = data.R[i, idx], data.Z[i, idx], data.phi[i, idx]
        sc_pol.set_offsets(np.column_stack([R, Z]))
        sc_top.set_offsets(np.column_stack([R * np.cos(phi), R * np.sin(phi)]))
        if mask is None:
            colors = base
        else:
            colors = np.where(mask[i, idx][:, None], [[0.85, 0.1, 0.1, 1.0]], [[0.6, 0.6, 0.6, 0.5]])
        sc_pol.set_color(colors); sc_top.set_color(colors)
        fig.suptitle(f"t = {data.t[i]*1e3:.3f} ms")
        return sc_pol, sc_top

    anim = FuncAnimation(fig, update, frames=frames, blit=False)
    anim.save(path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    logger.info("Saved animation %s (%d frames)", path, len(frames))
