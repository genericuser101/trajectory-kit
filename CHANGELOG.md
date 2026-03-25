# Changelog

All notable changes to trajectory-kit will be documented here.

---

## [0.2.1] — 2026-03-25

### Added


### Changed
- `tutorial.ipynb` and the `dev_tutorial.ipynb` have been expanded and further unit tests added.


### Fixed


### Current Issues
- `.mea` Maestro formats need unit tests, these are coming in the next patch.


---

## [0.2.0] — 2026-03-24

### Added
- Support for Maestro `.mae` files as both typing and topology files, it borrows a lot of the logic from the psf bond network connectivity when passed as the topology file.

### Changed


### Fixed


### Current Issues
- `_iter_mae_atoms` im `mae_parse.py` has a couple of redundancies when converting strings to int or float variables.


---

## [0.1.18] — 2026-03-23

### Added


### Changed


### Fixed
- `positions()` no longer executes domain requests prior to plan
- `select()` no longer executes domain requests prior to plan
- `fetch()` no longer executes domain requests prior to plan


### Current Issues


---

## [0.1.17] — 2026-03-17

### Added
- `positions()` static fallback: when no trajectory file is loaded, positions are read directly from the typing file (PDB or XYZ) and returned as `(1, n_atoms, 3)` — the same shape as a single-frame trajectory
- `"positions"` request for PDB and XYZ typing files, equivalent to a combined `x`, `y`, `z` pass in a single array

### Changed
- Query values now accept shorthand input forms in addition to the canonical `(include, exclude)` pair: bare strings, bare sets, bare `(lo, hi)` range pairs, and single-sided tuples are all normalised automatically

### Fixed
- `positions()` no longer raises when neither `TYPE_Q` nor `TOPO_Q` is provided alongside a loaded trajectory — all atoms are selected by default


### Current Issues
- Query plans are written after there has been a union check for global indicies in a few cases, this needs to be changed otherwise the efficiency of the stochastic planner is void.