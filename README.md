# trajectory-kit

A lightweight, fast, engine-agnostic molecular dynamics trajectory extraction library. No analysis, no abstractions, no stress.

---

## Installation

```bash
git clone https://github.com/genericuser101/trajectory-kit.git
cd trajectory-kit
pip install .
```

---

## Getting Trajectories

```python
from trajectory_kit import sim

my_sim = sim(typing="system.pdb", topology="system.psf", trajectory="traj.dcd")

out = my_sim.positions(
    TYPE_Q={"atom_name": "CA"},
    TRAJ_Q={"frame_interval": (0, 99, 1)},
)

positions, meta = out["payload"]  # (n_frames, n_atoms, 3)
```

All CA atoms, frames 0–99, in a single call.

---

## Static Positions (no trajectory)

If you only have a coordinate file, `positions()` reads directly from it and returns the same shape as a single-frame trajectory:

```python
my_sim = sim(typing="system.pdb")

out = my_sim.positions(TYPE_Q={"atom_name": "CA"})

positions = out["payload"]  # (1, n_atoms, 3)
```

---

## Discovering Available Keywords and Requests

Before writing a query, you can inspect exactly which keywords and request strings your loaded files support. There are two ways to do this.

**`print_info()`** prints a formatted summary of all loaded files, system properties, and a side-by-side table of available keywords and requests for every domain:

```python
my_sim = sim(typing="system.pdb", topology="system.psf", trajectory="traj.dcd")
my_sim.print_info()
```

```
=== SIMULATION INFO ===

  files
    typing     system.pdb  (.pdb)
    topology   system.psf  (.psf)
    trajectory traj.dcd    (.dcd)

  system properties
    num_atoms    42000
    num_frames   500

  available keywords and requests
  ──────────  ────────────────  ────────────────  ────────────
              typing            topology          trajectory
  ──────────  ────────────────  ────────────────  ────────────
  keywords    atom_name         atom_name         frame_interval
              ...               ...               global_ids
  ──────────  ────────────────  ────────────────  ────────────
  requests    atom_names        bonds             positions
              positions         charges           global_ids
              ...               ...
```

**Programmatic access** — if you want to work with the sets directly:

```python
# keywords you can filter on
my_sim.get_type_keys()   # typing domain
my_sim.get_topo_keys()   # topology domain
my_sim.get_traj_keys()   # trajectory domain

# request strings you can pass as TYPE_R / TOPO_R / TRAJ_R
my_sim.get_type_reqs()
my_sim.get_topo_reqs()
my_sim.get_traj_reqs()
```

---

## Query Schema

Every query uses an include/exclude pattern. trajectory-kit accepts several levels of shorthand — use whichever is clearest:

```python
# bare value → include only, no exclude
{"atom_name": "CA"}
{"atom_name": {"CA", "CB"}}

# explicit one-sided → include only
{"atom_name": ({"CA", "CB"},)}

# explicit two-sided → include and exclude
{"atom_name": ({"CA", "CB"}, {"CB"})}
```

All four forms are equivalent. Missing or empty exclude (`{}`) means accept everything not explicitly included. **Exclusion always wins over inclusion.**

Numeric fields use a single `(lo, hi)` range instead of sets. A bare pair is treated as an include range:

```python
# shorthand — single range, include only
{"residue_ids": (10, 50)}
{"x": (0.0, 20.0)}

# explicit two-sided — single include range and single exclude range
{"residue_ids": ((10, 50), (8, 12))}
{"x":           ((0.0, 20.0), (8.0, 12.0))}
```

Bounds are inclusive. Either bound can be `None` for open-ended ranges:

```python
{"residue_ids": (10, None)}   # residue 10 and above
{"x": (None, 20.0)}           # everything up to 20 Å
```

### Full example

```python
# CA or CB atoms, in PROT segment, x between 0 and 20 Å, not in residues 8–12
{
    "atom_name":    {"CA", "CB"},
    "segment_name": ({"PROT"}, {}),
    "x":            ((0.0, 20.0), None),
    "residue_ids":  (None, (8, 12)),
}
```

Typing, topology, and trajectory queries all follow identical rules. If you provide both a typing and a topology query, trajectory-kit takes their intersection automatically.

---

## Topology-aware Selection

Topology queries can filter by bond graph — useful for identifying chemical environments without any analysis code:

```python
# atoms with exactly 4 bonds (tetrahedral carbon)
{
    "bonded_with": ([{"total": True, "count": {"eq": 4}}], []),
    "bonded_with_mode": ("all", None),
}

# carbonyl carbons: 1 bond to CD2O2A, 3 bonds to CG2R61
{
    "bonded_with": (
        [
            {"neighbor": {"atom_type": ({"CD2O2A"}, {})}, "count": {"eq": 1}},
            {"neighbor": {"atom_type": ({"CG2R61"}, {})}, "count": {"eq": 3}},
        ],
        [],
    ),
    "bonded_with_mode": ("all", None),
}
```

Supported comparators: `eq`, `ne`, `ge`, `le`, `gt`, `lt`.
`bonded_with_mode` can be `"all"` (atom must satisfy every constraint) or `"any"` (atom must satisfy at least one).

---

## Supported Formats

| Domain     | Formats           |
|------------|-------------------|
| Typing     | `.pdb`, `.xyz`, `.mae` |
| Topology   | `.psf`, `.mae`    |
| Trajectory | `.dcd`            |

### Dual-domain files

Some formats carry enough information to serve as both a typing file and a topology file simultaneously. `.mae` (Maestro) is the current example — it contains per-atom coordinates and names, as well as a full bond table and force-field parameters. You can load it into both slots in one go:

```python
my_sim = sim(typing="system.mae", topology="system.mae")
```

trajectory-kit loads the file once per domain, so each slot gets the appropriate keyword and request set for that domain. Atom count consistency is validated automatically across all loaded files.

---

## Adding a New Format

Write a single `{fmt}_parse.py` file with four functions:

| Function | Purpose |
|----------|---------|
| `_get_{domain}_keys_reqs_{fmt}` | Return available query keywords and request strings |
| `_plan_{domain}_query_{fmt}` | Return a stochastic execution plan without reading the full file |
| `_get_{domain}_query_{fmt}` | Execute a query and return the result |
| `_update_{domain}_globals_{fmt}` | Extract global system properties (atom count, box size, etc.) |

Add the file extension to the domain registry in `main.py`. No other changes required — the architecture validates the contract at import time and raises a clear error if any function is missing.

---

## Support Roadmap

### Typing

### Topology

- **.pdb**: Using the CONNECT keyword to build a topology similar to psf and allow for bond query parsing.

### Trajectories

- **.traj**: Trajectory file.

- **.coor**: Trajectory file.

- **.thermal**: LAMMPS trajectory information file.

---

## Contact & Support

stefan.zhikharev@warwick.ac.uk

This project is maintained by one person. If you want a new feature or you don't like something just email me or write aits not that deep. Love, Stef.