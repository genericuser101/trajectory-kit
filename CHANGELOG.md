# Changelog

All notable changes to trajectory-kit will be documented here.


---

## [0.3.0] — 2026-04-20 — MAJOR UPDATE

### Added
- `.coor` file support for trajectories, it is a single frame trajectory primarily used to extract the equilibrated coordinates.
- `.coor` file has unit tests.
- `tutorial_advanced.ipynb` has been added and fully explains the planner system which utilises the new `devFlag: bool = True` behaviour. 

### Changed
- **_get_plan\_*domain*\_plan_shape\_*filetype*** is now a mandatory domain registry function for all domain types, this changes aids in the planner and provided better plan and metadata values. This affect `all_file_parsers` and all the functionality has been unified across domain and formats.
- **Major testing suite overhaul**. The test suite now consists of a central master file from which all other files are defined, physical test files are no longer needed all have been swapped using the master synthetic file. The cross-correlation matrix tests all the combinations of typing, topology, trajectory files.
- `_freeze()` has been made a general function for the topology domain, to freeze neighbour lists between recursive depths, the behaviour is consistent across the `.psf` and `.mae` formats, and will be suggested usage for all further topology files.
- `devFlag = False` is now default behaviour for all queries, in this mode no additional metadata is loaded into the output, planner mode is unavailble and only the payload serves as output. New hotpaths introduced to `positions()`, `fetch()` and `select()` functions.
- Recursive neighbour calls are allowed, maximum neighbour recursive depth `16`.
- Further parser functionality has expanded for ranges, now take distinct value lists as well as ranges for both include and exclude. The (1,10) range and the [1, 2, 3, 4, 5, 6, 7, 8, 9] explicit list produce the same query.
- All files can be querried using the `global_ids` keyword.
- `tutorial.ipynb` has been stramlined to be more accessible to users of various technical backgrounds non-user-facing functions have been removed entirely from the notebook.
- `tutorial_dev.ipynb` has been changed to include an example of developing a toy trajectory file format and has further been expanded for people wanting to contribute to `trajectory_kit`.


### Fixed
- `.mea` Maestro format got unit tests.
- `.mea` Maestro format inconsistencies changed to better suit the topology architecture.



### Current Issues
- `.pdb` does not have the `CONNECT` support yet, this will come next update.
- There is no `writer` support yet, this will also be added in the next update.

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