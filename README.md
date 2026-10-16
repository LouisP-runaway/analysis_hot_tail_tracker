# analysis_hot_tail_tracker

Post-processing of JOREK test-marker simulations for the study of **hot-tail runaway electron (RE) generation** during tokamak disruptions.

The marker tracker follows a population of weighted electron markers in the time-dependent MHD fields of a JOREK simulation. These scripts read its HDF5 outputs and rebuild the electron distribution function in the (p, ξ, R, Z, φ) phase space, select the runaway population, follow the markers lost at the wall, and produce the standard figures and animations of a hot-tail study.

<p align="center">
  <img src="docs/overview.png" width="800" alt="Example figures produced on a synthetic run">
</p>

## Features

* Reads the three output families of the tracker (trajectories, fields seen by the markers, wall losses) and concatenates several files in time.
* Builds weighted histograms on any subset of (p, ξ, R, Z, φ), normalised to the number of electrons, with optional p² and R Jacobians.
* Selects runaway markers with a prescribed momentum threshold or the local critical momentum (isotropic or pitch dependent).
* Computes the RE fraction and the RE current versus time.
* Figures: 1D and 2D distribution functions, poloidal and top views of the density, fields and momentum seen by individual markers, marker trajectories, weight budget. Every 2D figure also exists as a GIF animation.
* One user script, `main.py`, with all settings grouped at the top and one line per figure at the bottom.

## Installation

```bash
git clone git@github.com:LouisP-runaway/analysis_hot_tail_tracker.git
cd analysis_hot_tail_tracker
pip install numpy scipy matplotlib h5py pillow
```

Python 3.9 or newer. Pillow is only needed to export GIF animations.

## Input files

| Family | Typical name | Content |
|---|---|---|
| Track file | `diag.h5` | Marker trajectories under `groups/001/`: `t`, `weight`, `p`, `xi`, `R`, `Z`, `phi`, optionally `psi_bar` |
| Field file | `Jcoll_*.h5` | Plasma fields seen by each marker: `n` (m⁻³), `T` (eV), `E` (V/m) |
| Losses file | `Jlosses_*.h5` | One entry per marker lost at the wall: `t`, `Weight`, `p`, `xi`, `R`, `Z`, `Phi` |

Two-dimensional datasets are shaped `(time, marker)`. Momentum is in units of mₑc. A run split into several segments is given as a list of files and concatenated in time.

The wall outline is read from the JOREK namelist of the case (`R_boundary`, `Z_boundary`, and `R_limiter`, `Z_limiter` lines) placed in the `geometries/` folder. A missing geometry file only disables the wall outline.

## Quick start

1. Copy or link the HDF5 files of a run next to the scripts, or set `RUN_DIR` in `main.py`.
2. Edit the settings at the top of `main.py`: file names, tokamak, histogram grids, PDF options.
3. Uncomment the figures you want at the bottom of `main.py`.
4. Run:

```bash
python main.py
```

Figures and animations are written to `FIG_DIR` (default `figures/`).

To try the scripts without real data, generate a small synthetic run with the same HDF5 layout:

```bash
python make_synthetic_data.py test_run
```

The physics of this run is a toy model. It is only meant to check that the pipeline works.

## Usage from Python

```python
import numpy as np
from analysis import MarkerAnalysis

A = MarkerAnalysis(["diag.h5"], ["Jcoll_run.h5"], ["Jlosses_run.h5"], tokamak="ITER")
A.load(t_step=2, origin_time=True)

grids = dict(p=np.linspace(1e-4, 8, 151), xi=np.linspace(-1, 1, 91),
             R=np.linspace(1, 10, 121), Z=np.linspace(-6, 6, 71), phi=np.linspace(0, 2*np.pi, 3))
A.build_pdfs(grids, p_c=1.3, space_resolved=False, compute_re=True, compute_losses=True, Te_mj=25e3)

A.plot_1d("p", which="all", n_times=3, vmin=1e10, vmax=1e25)
A.plot_2d("p", "xi", which="re", vmin=1e10, vmax=1e24, animate=True)
A.plot_re_fraction(xi_dependent=True)
A.plot_re_current(I_p=15e6, seed_threshold=1e-6)
```

### `MarkerAnalysis` methods

| Method | Figure |
|---|---|
| `plot_1d(dim, which, ...)` | f integrated over every dimension except `dim`, at several times, with the Maxwell-Jüttner reference for `dim="p"` |
| `plot_2d(dim1, dim2, which, ...)` | Map of f in the (dim1, dim2) plane |
| `plot_poloidal(which, ...)` | Poloidal (R, Z) map, optionally as a toroidally averaged density in m⁻³ |
| `plot_topview(which, ...)` | Top (X, Y) map of the line-integrated density |
| `plot_re_fraction(...)` | N_RE / N₀ versus time |
| `plot_re_current(...)` | RE current versus time |
| `plot_fields(...)` | n, E, T and p_c seen by a subset of markers |
| `plot_momentum(...)` | Momentum of a subset of markers |
| `plot_trajectories(...)` | Poloidal and top views of marker positions, static or animated |
| `plot_weights(...)` | Weight budget: total, lost at the wall, removed by reweighting |

`which` selects a distribution: `"all"`, `"re"` or `"losses"`. Dimensions are named `"p"`, `"xi"`, `"R"`, `"Z"`, `"phi"`. Every map method accepts `animate=True` to write a GIF over all times instead of a single frame. Arguments are documented in the docstrings of `analysis.py`.

## Conventions

**Grids.** Grids are bin edges. A full-resolution 5D histogram is too large in memory, so two modes exist. `space_resolved=False` keeps the fine (p, ξ) grid with a single R and Z bin and no φ. `space_resolved=True` keeps the fine (R, Z, φ) grid with a single p and ξ bin.

**Normalisation.** With `is_pdf=True` the integral of f over all dimensions equals the total marker weight at each time, i.e. the number of electrons. With `spatial_jacobian=True` (default) f is a density per physical volume R dR dZ dφ, and `plot_poloidal(density=True)` returns the toroidally averaged density in m⁻³. The losses distribution is cumulative in time and normalised to the weight lost before each time.

**Colour scales.** Maps use a logarithmic scale by default. `vmin` and `vmax` are in data units (for example `1e10`, `1e24`), not in log10. Empty bins are drawn in the background colour.

**Runaway criterion.** A marker is counted as runaway when p ≥ p_c. With `p_c` given, the threshold is fixed. Otherwise the local critical momentum is used, from the Connor-Hastie critical field E_c = e³ n_e ln Λ / (4π ε₀² mₑ c²) with the relativistic Coulomb logarithm of Hesslow et al., J. Plasma Phys. 84 (2018):

* isotropic: p_c = 1 / √(E/E_c − 1)
* pitch dependent (`xi_dependent=True`): p_c(ξ) = 1 / √((1 − ξ) E / (2 E_c) − 1)

**RE current.** I_RE = e / (2π R₀) Σᵢ wᵢ v∥,ᵢ over the runaway markers, with v∥ = ξ c √(1 − 1/(1 + p²)), or v∥ = c with `light_speed=True`.

## Code structure

```
main.py                 user script: settings and figure calls
analysis.py             MarkerAnalysis: load, build PDFs, one method per figure
pdf_constructor.py      build_pdfs(): f_all, f_re, f_losses, f_mj into a PDFSet
class_pdf.py            MarkerPDF (histogram, integration, normalisation, maps, animations),
                        RE physics, geometry reader, trajectory plots
import_simu_data.py     HDF5 readers and the MarkerData container
make_synthetic_data.py  synthetic run generator for testing
geometries/             JOREK namelists with the wall and limiter outlines
```

## Credits

Original implementation by Louis Puel (CEA, 2024), author of the hot-tail marker tracker. Cleaned, restructured and extended by Timothé Auclair (CEA, 2026).

## License

To be defined.
