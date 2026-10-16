"""
import_simu_data.py

Readers for the HDF5 outputs of the hot-tail marker tracker (JOREK).

Three file families are handled:
    * track files  (diag.h5)       marker trajectories in phase space, under "groups/001/"
    * field files  (Jcoll_*.h5)    plasma fields (n, T, E) seen by each marker
    * losses files (Jlosses_*.h5)  one entry per marker lost at the wall

All (time, marker) arrays share the same shape. Several files of one family
can be given as a list, in which case they are concatenated along time.

Original implementation: Louis Puel (2024). Cleaned and restructured in 2026.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import h5py
import numpy as np

logger = logging.getLogger("hottail")

TRACK_GROUP = "groups/001/"
P_NAN = -0.1    # sentinel replacing NaN momentum of lost markers (falls outside momentum grids)
XI_NAN = -1.1   # sentinel replacing NaN pitch of lost markers (falls outside pitch grids)


# --------------------------------------------------------------------------- #
#  Data container
# --------------------------------------------------------------------------- #
@dataclass
class MarkerData:
    """All marker quantities of one run. 2D arrays are shaped (time, marker)."""

    t: np.ndarray            # (nt,)  time [s]
    weight: np.ndarray       # (nt, nmark) marker weight [electrons]
    p: np.ndarray            # (nt, nmark) momentum [m_e c]
    xi: np.ndarray           # (nt, nmark) pitch cosine
    R: np.ndarray            # (nt, nmark) major radius [m]
    Z: np.ndarray            # (nt, nmark) height [m]
    phi: np.ndarray          # (nt, nmark) toroidal angle in [0, 2 pi)
    n: np.ndarray            # (nt, nmark) electron density at the marker [m^-3]
    T: np.ndarray            # (nt, nmark) electron temperature at the marker [eV]
    E: np.ndarray            # (nt, nmark) parallel electric field at the marker [V/m]
    psi: np.ndarray | None = None   # (nt, nmark) |normalised poloidal flux|, if available

    # Losses: 1D arrays, one entry per lost marker. None when no losses file is given.
    t_loss: np.ndarray | None = None
    w_loss: np.ndarray | None = None
    p_loss: np.ndarray | None = None
    xi_loss: np.ndarray | None = None
    R_loss: np.ndarray | None = None
    Z_loss: np.ndarray | None = None
    phi_loss: np.ndarray | None = None

    @property
    def nt(self) -> int:
        return self.t.shape[0]

    @property
    def nmark(self) -> int:
        return self.weight.shape[1]

    @property
    def has_losses(self) -> bool:
        return self.t_loss is not None

    @property
    def n_lost(self) -> int:
        return 0 if self.t_loss is None else self.t_loss.shape[0]

    def summary(self) -> str:
        lines = [
            f"{self.nt} times from {self.t[0]*1e3:.3f} to {self.t[-1]*1e3:.3f} ms, "
            f"{self.nmark} markers",
            f"initial total weight {self.weight[0].sum():.3e}, "
            f"final total weight {self.weight[-1].sum():.3e}",
            f"p in [{np.nanmin(self.p):.3g}, {np.nanmax(self.p):.3g}] m_e c, "
            f"R in [{np.nanmin(self.R):.2f}, {np.nanmax(self.R):.2f}] m, "
            f"Z in [{np.nanmin(self.Z):.2f}, {np.nanmax(self.Z):.2f}] m",
        ]
        if self.has_losses:
            lines.append(f"{self.n_lost} lost markers, lost weight {self.w_loss.sum():.3e}")
        return "\n".join("  " + s for s in lines)


# --------------------------------------------------------------------------- #
#  Public loader
# --------------------------------------------------------------------------- #
def load_jorek(track_files, coll_files, losses_files=None,
               t_start=None, t_end=None, t_step=None, origin_time=False) -> MarkerData:
    """
    Load one run of the marker tracker.

    Args:
        track_files:  path or list of paths to the track files (diag.h5).
        coll_files:   path or list of paths to the field files (Jcoll_*.h5).
        losses_files: path or list of paths to the losses files (Jlosses_*.h5), optional.
        t_start, t_end, t_step: time-index slice applied to each file before concatenation.
        origin_time:  if True, shift times so that the first time is zero.
    """
    track_files = _as_list(track_files)
    coll_files = _as_list(coll_files)
    losses_files = _as_list(losses_files) if losses_files is not None else None
    if len(track_files) != len(coll_files):
        raise ValueError("track_files and coll_files must have the same number of files")

    sl = slice(t_start, t_end, t_step)

    tracks = [_read_track(f, sl) for f in track_files]
    fields = [_read_fields(f, sl) for f in coll_files]
    track = _concat_dicts(tracks)
    field = _concat_dicts(fields)
    if field["n"].shape != track["p"].shape:
        raise ValueError(f"Field arrays {field['n'].shape} do not match track arrays {track['p'].shape}")

    losses = {}
    if losses_files is not None:
        losses = _concat_dicts([_read_losses(f) for f in losses_files])

    t0 = track["t"][0] if origin_time else 0.0
    data = MarkerData(
        t=track["t"] - t0, weight=track["weight"],
        p=track["p"], xi=track["xi"], R=track["R"], Z=track["Z"], phi=track["phi"],
        psi=track.get("psi"),
        n=field["n"], T=field["T"], E=field["E"],
    )
    if losses:
        data.t_loss = losses["t"] - t0
        data.w_loss = losses["weight"]
        data.p_loss, data.xi_loss = losses["p"], losses["xi"]
        data.R_loss, data.Z_loss, data.phi_loss = losses["R"], losses["Z"], losses["phi"]

    logger.info("Loaded run:\n%s", data.summary())
    return data


# --------------------------------------------------------------------------- #
#  File readers
# --------------------------------------------------------------------------- #
def _read_track(path, sl):
    """Read the marker trajectories (diag.h5). Missing optional datasets give None."""
    logger.info("Reading track file  %s", path)
    with h5py.File(path, "r") as h5:
        g = h5[TRACK_GROUP]
        out = {
            "t": _get(g, "t", sl),
            "weight": _get(g, "weight", sl),
            "p": _get(g, "p", sl),
            "xi": _get(g, "xi", sl),
            "R": _get(g, "R", sl),
            "Z": _get(g, "Z", sl),
            "phi": _get(g, "phi", sl) % (2 * np.pi),
        }
        if "psi_bar" in g:
            out["psi"] = np.abs(_get(g, "psi_bar", sl))
    return out


def _read_fields(path, sl):
    """Read the fields seen by the markers (Jcoll_*.h5)."""
    logger.info("Reading field file  %s", path)
    with h5py.File(path, "r") as h5:
        return {k: _get(h5, k, sl) for k in ("n", "T", "E")}


def _read_losses(path):
    """Read the lost markers (Jlosses_*.h5). NaN momentum and pitch are replaced by sentinels."""
    logger.info("Reading losses file %s", path)
    with h5py.File(path, "r") as h5:
        return {
            "t": h5["t"][:],
            "weight": h5["Weight"][:],
            "p": np.nan_to_num(h5["p"][:], nan=P_NAN),
            "xi": np.nan_to_num(h5["xi"][:], nan=XI_NAN),
            "R": h5["R"][:],
            "Z": h5["Z"][:],
            "phi": h5["Phi"][:] % (2 * np.pi),
        }


# --------------------------------------------------------------------------- #
#  Helpers
# --------------------------------------------------------------------------- #
def _as_list(x):
    return list(x) if isinstance(x, (list, tuple)) else [x]


def _get(group, key, sl):
    """Read dataset `key` with the time slice `sl` on the first axis."""
    if key not in group:
        raise KeyError(f"Dataset '{key}' not found in {group.file.filename}")
    return group[key][sl]


def _concat_dicts(dicts):
    """Concatenate along the first axis the arrays of a list of dicts sharing the same keys."""
    keys = [k for k in dicts[0] if all(k in d and d[k] is not None for d in dicts)]
    if len(dicts) == 1:
        return {k: dicts[0][k] for k in keys}
    return {k: np.concatenate([d[k] for d in dicts], axis=0) for k in keys}
