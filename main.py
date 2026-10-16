"""
main.py

User script for the analysis of hot-tail marker simulations (JOREK marker tracker).
Edit the settings below, then comment or uncomment the figure calls at the bottom.

    python main.py

Figures are written to FIG_DIR. Every method is documented in analysis.py.

Original implementation: Louis Puel (2024). Cleaned and restructured in 2026.
"""

import logging
import os

import numpy as np

from analysis import MarkerAnalysis

logging.basicConfig(level=logging.INFO, format="[hottail] %(message)s")

# =============================================================================
#  INPUT FILES
# =============================================================================
RUN_DIR = os.getcwd()
TRACK_FILES = ["diag.h5"]                                # marker trajectories, list to concatenate in time
COLL_FILES = ["Jcoll_K0._410._np30000_dt50.ns.h5"]         # fields seen by the markers, same length as TRACK_FILES
LOSSES_FILES = None # ["Jlosses_K0._410._np30000_dt10.ns.h5"]     # lost markers, or None

TOKAMAK = "ITER"            # "ITER" or "DIII-D": magnetic axis and wall geometry (geometries/ folder)
FIG_DIR = "figures"         # output folder for figures and animations
SHOW = False                # open each figure in a window in addition to saving it

# =============================================================================
#  TIME SELECTION
# =============================================================================
T_START, T_END, T_STEP = None, None, 1   # time-index slice, e.g. T_STEP = 2 keeps one time out of two
ORIGIN_TIME = True                       # set t = 0 at the first time read

# =============================================================================
#  HISTOGRAM GRIDS (bin edges)
# =============================================================================
# ITER
GRIDS = dict(
    p=np.linspace(1e-4, 3, 151),         # momentum [m_e c]
    xi=np.linspace(-1, 1, 91),           # pitch
    R=np.linspace(1, 10, 121),           # [m]
    Z=np.linspace(-6, 6, 70),            # [m]
    phi=np.linspace(0, 2 * np.pi, 3),    # [rad]
)
# DIII-D
#GRIDS = dict(p=np.linspace(1e-4, 8, 201), xi=np.linspace(-1, 1, 31), R=np.linspace(1, 2.4, 31),
#              Z=np.linspace(-1, 1, 31), phi=np.linspace(0, 2 * np.pi, 3))

# =============================================================================
#  PDF OPTIONS
# =============================================================================
P_C = 1.3                  # momentum [m_e c] above which a marker is a RE. None: local critical momentum
SPACE_RESOLVED = True       # True: fine (R, Z, phi) grid, single p and xi bin. False: fine (p, xi), no space
COMPUTE_RE = True          # build the RE PDF and the RE current
XI_DEPENDENT = False        # pitch-dependent RE criterion (only used when P_C is None)
COMPUTE_LOSSES = False       # build the cumulative PDF of lost markers (needs LOSSES_FILES)
RE_LOSSES = False           # keep only lost markers with p > P_C in the losses PDF
IS_PDF = True               # True: density normalised to the marker weight. False: raw histogram
GAUSS_SMOOTH = None         # Gaussian smoothing, fraction of the number of bins (e.g. 0.02), or None
TE_MJ = 25e3                # temperature [eV] of the Maxwell-Juttner overlaid on the 1D f(p) plot


# =============================================================================
#  ANALYSIS
# =============================================================================
if __name__ == "__main__":
    join = lambda files: None if files is None else [os.path.join(RUN_DIR, f) for f in files]

    A = MarkerAnalysis(join(TRACK_FILES), join(COLL_FILES), join(LOSSES_FILES),
                       tokamak=TOKAMAK, fig_dir=FIG_DIR, show=SHOW)
    A.load(t_start=T_START, t_end=T_END, t_step=T_STEP, origin_time=ORIGIN_TIME)
    A.build_pdfs(GRIDS, p_c=P_C, space_resolved=SPACE_RESOLVED,
                 compute_re=COMPUTE_RE, xi_dependent=XI_DEPENDENT,
                 compute_losses=COMPUTE_LOSSES, re_losses=RE_LOSSES,
                 is_pdf=IS_PDF, gauss_smooth=GAUSS_SMOOTH, Te_mj=TE_MJ)

    # ---- 1D PDF: f(dim) at several times. which: "all", "re" or "losses"
    # A.plot_1d("p", which="all", n_times=3, vmin=1e10, vmax=1e25)

    # ---- 2D PDF in the (dim1, dim2) plane. vmin/vmax in PDF units. animate=True writes a GIF
    #A.plot_2d("p", "xi", which="all", i_time=0, vmin=1e10, vmax=1e20, interp=2, animate=True)

    # ---- Poloidal view (needs SPACE_RESOLVED = True). density=True gives <n>_phi in m^-3
    A.plot_poloidal(which="re", i_time=-1, vmin=1e8, vmax=1e13, interp=3, density=True, animate=True)

    # ---- Top view (needs SPACE_RESOLVED = True)
    #A.plot_topview(which="re", i_time=-1, vmin=1e8, vmax=1e13, interp=2, animate=True)

    # ---- RE fraction N_RE / N_0 versus time
    A.plot_re_fraction(xi_dependent=True, vmin=1e-20, vmax=1e0)

    # ---- RE current versus time (needs COMPUTE_RE = True)
    #A.plot_re_current(I_p=15e6, light_speed=False, seed_threshold=1e-6, vmin=1e-20, vmax=None)

    # ---- Fields (n, E, T, p_c) seen by one marker out of `every`
    #A.plot_fields(every=1000, ylims=[None, None, None, (0, 5)])

    # ---- Momentum of one marker out of `every`
    # A.plot_momentum(every=100, vmin=0, vmax=5)

    # ---- Marker trajectories (count or index array). distinct_re needs COMPUTE_RE = True
    # A.plot_trajectories(particles=50, t_range=(0, 92), distinct_re=False, animate=True)

    # ---- Weight budget: total, lost, reweighted (needs LOSSES_FILES)
    # A.plot_weights(log=False)
