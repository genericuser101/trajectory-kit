"""
tests/conftest.py
=================
Single source of truth for integration-test synthetic files.

Invocation
----------
Run pytest from the project root (the directory containing ``trajectory_kit/``)::

    pytest tests/
    pytest tests/integration/
    pytest tests/integration/test_bonded_with.py

Running from inside ``tests/`` will NOT work because ``trajectory_kit`` must
be on ``sys.path``, which pytest configures via its rootdir discovery when
invoked from the project root.

Design
------
A *master* atom/bond list lives here. Each file-format writer derives its
content from the master; nothing is hand-written per format. If the master
changes, every format auto-regenerates consistently.

Integration tests then cross-correlate outputs: if a canonical field (e.g.
``atom_names``) is available in multiple ``sim`` configurations, all
configurations must return the same output. We never assert absolute values
against hand-authored expectations — we assert *consistency across sims built
from the same master*.

System composition — 54 atoms
-----------------------------
- Ligand  (20 atoms, segment "LIG", residue "DRG" resid 1)
    Aromatic ring, methyl, carboxyl, amine, ether-methyl
- Waters  (30 atoms, segment "SOLV", 10 TIP3P residues resids 2..11)
- Ions    ( 4 atoms, segment "IONS", 2 K+ resids 12..13, 2 Cl- resids 14..15)

Bonds (~40): ring closure, sidechain chains, water O-H; ions unbonded.

Per-frame trajectory: 4 frames where atom i at frame k has x += k (y, z fixed).
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Master atomic data
# ---------------------------------------------------------------------------
# Each atom record:
#   id         : 0-based global index
#   name       : atom_name
#   type       : atom_type / vdw_type
#   resn       : residue_name
#   resi       : residue_id (1-based, per CHARMM convention)
#   seg        : segment_name (4 char max)
#   x, y, z    : Cartesian coordinates (Angstrom)
#   charge     : partial charge (e)
#   mass       : atomic mass (g/mol)
#   occ        : PDB occupancy
#   temp       : PDB temperature factor / MAE pdb_tfactor

def _build_master_atoms():

    atoms = []

    # --- Ligand: 20 atoms ----------------------------------------------------
    # Aromatic ring: 6 carbons in a hexagon centred at (0.7, 1.2)
    ring = [
        (0, "C1", "CA",  0.0,   0.0,  0.0),
        (1, "C2", "CA",  1.4,   0.0,  0.0),
        (2, "C3", "CA",  2.1,   1.2,  0.0),
        (3, "C4", "CA",  1.4,   2.4,  0.0),
        (4, "C5", "CA",  0.0,   2.4,  0.0),
        (5, "C6", "CA", -0.7,   1.2,  0.0),
    ]
    # Methyl off C1 (atom 0)
    methyl = [
        (6, "C7", "CT3", -2.1, 0.0,  0.0),
        (7, "H1", "HA3", -2.6,-0.9,  0.0),
        (8, "H2", "HA3", -2.6, 0.5,  0.8),
        (9, "H3", "HA3", -2.6, 0.5, -0.8),
    ]
    # Carboxyl off C4 (atom 3)
    carboxyl = [
        (10, "C8", "CC",   2.1, 3.6, 0.0),
        (11, "O1", "OC",   3.3, 3.6, 0.0),
        (12, "O2", "OH1",  1.4, 4.8, 0.0),
        (13, "H4", "H",    2.0, 5.6, 0.0),
    ]
    # Amine off C3 (atom 2)
    amine = [
        (14, "N1", "NH2",  3.5, 1.2, 0.0),
        (15, "H5", "H",    4.0, 2.0, 0.3),
        (16, "H6", "H",    4.0, 0.4, 0.3),
    ]
    # Ether + methyl off C5 (atom 4)
    ether = [
        (17, "O3", "OS",  -0.7, 3.6, 0.0),
        (18, "C9", "CT3", -2.1, 3.6, 0.0),
        (19, "H7", "HA3", -2.6, 4.5, 0.0),
    ]

    # Charge / mass map by atom_type
    type_props = {
        "CA":  ( -0.115, 12.011),
        "CT3": ( -0.270, 12.011),
        "HA3": (  0.090,  1.008),
        "CC":  (  0.620, 12.011),
        "OC":  ( -0.760, 15.999),
        "OH1": ( -0.650, 15.999),
        "OS":  ( -0.340, 15.999),
        "NH2": ( -0.800, 14.007),
        "H":   (  0.400,  1.008),
        "OT":  ( -0.834, 15.999),
        "HT":  (  0.417,  1.008),
        "POT": (  1.000, 39.098),
        "CLA": ( -1.000, 35.453),
    }
    ligand_atoms = ring + methyl + carboxyl + amine + ether
    for (idx, name, atype, x, y, z) in ligand_atoms:
        charge, mass = type_props[atype]
        atoms.append({
            "id": idx, "name": name, "type": atype,
            "resn": "DRG", "resi": 1, "seg": "LIG",
            "x": x, "y": y, "z": z,
            "charge": charge, "mass": mass,
            "occ": 1.00, "temp": 0.10,
        })

    # --- Waters: 10 TIP3P molecules, 30 atoms --------------------------------
    # Grid 2 rows × 5 cols, spaced 3.5 Å apart, offset from ligand
    water_x0, water_y0 = 8.0, 0.0
    for w in range(10):
        col = w % 5
        row = w // 5
        ox = water_x0 + col * 3.5
        oy = water_y0 + row * 3.5
        oid = 20 + w * 3
        resid = 2 + w
        # O, H1, H2 — water internal geometry: O at (ox,oy), H1 at +0.96 Å along +x,
        # H2 at +0.96 Å rotated by the H-O-H angle (~104.5°). We just approximate.
        charge_o, mass_o = type_props["OT"]
        charge_h, mass_h = type_props["HT"]
        atoms.append({"id": oid,     "name": "OH2", "type": "OT",
                      "resn": "TIP",  "resi": resid, "seg": "SOLV",
                      "x": ox, "y": oy, "z": 0.0,
                      "charge": charge_o, "mass": mass_o,
                      "occ": 1.00, "temp": 0.50})
        atoms.append({"id": oid + 1, "name": "H1",  "type": "HT",
                      "resn": "TIP",  "resi": resid, "seg": "SOLV",
                      "x": ox + 0.96, "y": oy, "z": 0.0,
                      "charge": charge_h, "mass": mass_h,
                      "occ": 1.00, "temp": 0.50})
        atoms.append({"id": oid + 2, "name": "H2",  "type": "HT",
                      "resn": "TIP",  "resi": resid, "seg": "SOLV",
                      "x": ox - 0.24, "y": oy + 0.93, "z": 0.0,
                      "charge": charge_h, "mass": mass_h,
                      "occ": 1.00, "temp": 0.50})

    # --- Ions: 2 K+, 2 Cl- ---------------------------------------------------
    ion_defs = [
        (50, "POT", "POT", 12, "K",  12.0,  0.0),
        (51, "POT", "POT", 13, "K",  12.0,  4.0),
        (52, "CLA", "CLA", 14, "CL", 15.0,  0.0),
        (53, "CLA", "CLA", 15, "CL", 15.0,  4.0),
    ]
    for (idx, resn, atype, resi, name, x, y) in ion_defs:
        charge, mass = type_props[atype]
        atoms.append({
            "id": idx, "name": name, "type": atype,
            "resn": resn, "resi": resi, "seg": "IONS",
            "x": x, "y": y, "z": 0.0,
            "charge": charge, "mass": mass,
            "occ": 1.00, "temp": 0.60,
        })

    return atoms


def _build_master_bonds():
    """Master bond list — (local_id_i, local_id_j) pairs with 1-based ids,
    matching CHARMM convention. We derive this from the atom structure."""
    bonds = []
    # Ligand ring (atoms 0..5, local_ids 1..6)
    ring_local = [1, 2, 3, 4, 5, 6]
    for i in range(6):
        bonds.append((ring_local[i], ring_local[(i + 1) % 6]))
    # Methyl: C1-C7, C7-H1, C7-H2, C7-H3 (0-6, 6-7, 6-8, 6-9 globals)
    bonds += [(1, 7), (7, 8), (7, 9), (7, 10)]
    # Carboxyl: C4-C8, C8=O1, C8-O2, O2-H (3-10, 10-11, 10-12, 12-13)
    bonds += [(4, 11), (11, 12), (11, 13), (13, 14)]
    # Amine: C3-N, N-H5, N-H6 (2-14, 14-15, 14-16)
    bonds += [(3, 15), (15, 16), (15, 17)]
    # Ether: C5-O3, O3-C9, C9-H7 (4-17, 17-18, 18-19)
    bonds += [(5, 18), (18, 19), (19, 20)]
    # Waters: 10 waters, atoms O at 20..20+27 step 3 (globals 20,23,26,...,47)
    # local_ids = global + 1
    for w in range(10):
        oid = 20 + w * 3
        bonds.append((oid + 1, oid + 2))  # O - H1
        bonds.append((oid + 1, oid + 3))  # O - H2
    # Ions: unbonded — no entries
    return bonds


MASTER_ATOMS = _build_master_atoms()
MASTER_BONDS = _build_master_bonds()
N_ATOMS = len(MASTER_ATOMS)                   # 54
N_FRAMES = 4                                  # for DCD
N_BONDS = len(MASTER_BONDS)

# Convenience id groupings
LIGAND_IDS = [a["id"] for a in MASTER_ATOMS if a["seg"] == "LIG"]
SOLV_IDS   = [a["id"] for a in MASTER_ATOMS if a["seg"] == "SOLV"]
IONS_IDS   = [a["id"] for a in MASTER_ATOMS if a["seg"] == "IONS"]
ALL_IDS    = list(range(N_ATOMS))

# Type-based groupings
OT_TYPE_IDS = [a["id"] for a in MASTER_ATOMS if a["type"] == "OT"]
HT_TYPE_IDS = [a["id"] for a in MASTER_ATOMS if a["type"] == "HT"]
CA_TYPE_IDS = [a["id"] for a in MASTER_ATOMS if a["type"] == "CA"]

# Bond-degree map: atom_global_id -> bond count
def _compute_degree_map():
    deg = {a["id"]: 0 for a in MASTER_ATOMS}
    for (li, lj) in MASTER_BONDS:
        deg[li - 1] += 1
        deg[lj - 1] += 1
    return deg

DEGREE_MAP = _compute_degree_map()


# ===========================================================================
# File writers — master data -> on-disk synthetic files
# ===========================================================================

def _write_pdb(path: Path):
    lines = ["REMARK synthetic file derived from tests/conftest.py master data\n"]
    for a in MASTER_ATOMS:
        serial = a["id"] + 1
        name   = a["name"]
        # PDB field-widths: columns 13-16 for name (right-justified for <4 chars)
        name_col = f" {name:<3s}" if len(name) < 4 else name
        lines.append(
            f"ATOM  {serial:5d} {name_col:4s} {a['resn']:<4s}{a['seg'][0]}{a['resi']:4d}    "
            f"{a['x']:8.3f}{a['y']:8.3f}{a['z']:8.3f}{a['occ']:6.2f}{a['temp']:6.2f}"
            f"      {a['seg']:<4s}\n"
        )
    lines.append("END\n")
    path.write_text("".join(lines))


def _write_xyz(path: Path):
    lines = [f"{N_ATOMS}\n", "synthetic XYZ from master\n"]
    for a in MASTER_ATOMS:
        lines.append(f"{a['name']:<6s} {a['x']:10.4f} {a['y']:10.4f} {a['z']:10.4f}\n")
    path.write_text("".join(lines))


def _write_psf(path: Path):
    out = ["PSF EXT\n\n", "       1 !NTITLE\n",
           " REMARKS synthetic PSF from master\n\n",
           f"{N_ATOMS:10d} !NATOM\n"]
    for a in MASTER_ATOMS:
        serial = a["id"] + 1
        out.append(
            f"{serial:10d} {a['seg']:<8s} {a['resi']:<8d} {a['resn']:<8s} "
            f"{a['name']:<8s} {a['type']:<8s} {a['charge']:13.6f} {a['mass']:13.4f}           0\n"
        )
    out.append(f"\n{N_BONDS:10d} !NBOND: bonds\n")
    # 4 bonds per line in PSF EXT = 8 local_ids
    tokens = []
    for (i, j) in MASTER_BONDS:
        tokens.extend([f"{i:10d}", f"{j:10d}"])
    for k in range(0, len(tokens), 8):
        out.append("".join(tokens[k:k + 8]) + "\n")
    out.append("\n       0 !NTHETA: angles\n\n")
    out.append("       0 !NPHI: dihedrals\n\n")
    out.append("       0 !NIMPHI: impropers\n\n")
    out.append("       0 !NDON: donors\n\n")
    out.append("       0 !NACC: acceptors\n\n")
    out.append("       0 !NNB\n\n")
    out.append("       0       0 !NGRP\n\n")
    path.write_text("".join(out))


def _write_dcd(path: Path, n_frames: int = N_FRAMES):
    """CHARMM DCD — natom × n_frames × (x,y,z) float32 records.
    Per frame k: atom i at (base_x + k, base_y, base_z)."""
    base_x = np.array([a["x"] for a in MASTER_ATOMS], dtype=np.float32)
    base_y = np.array([a["y"] for a in MASTER_ATOMS], dtype=np.float32)
    base_z = np.array([a["z"] for a in MASTER_ATOMS], dtype=np.float32)

    with open(path, "wb") as f:
        # --- Header block 1: 84-byte fixed record ---
        f.write(struct.pack("<i", 84))              # record-length marker
        f.write(b"CORD")                            # signature
        f.write(struct.pack("<i", n_frames))        # NSET
        f.write(struct.pack("<i", 0))               # ISTART
        f.write(struct.pack("<i", 1))               # NSAVC
        f.write(b"\x00" * 20)                       # 5 reserved ints
        f.write(struct.pack("<i", 0))               # NAMNF
        f.write(struct.pack("<f", 0.0))             # DELTA
        f.write(struct.pack("<i", 0))               # unitcell flag (0 = none)
        f.write(b"\x00" * 32)                       # 8 reserved ints
        f.write(struct.pack("<i", 24))              # CHARMM version
        f.write(struct.pack("<i", 84))              # end marker

        # --- Header block 2: title ---
        title = b"REMARK synthetic DCD from master" + b" " * (80 - 32)
        f.write(struct.pack("<i", 4 + len(title)))
        f.write(struct.pack("<i", 1))               # NTITLE
        f.write(title)
        f.write(struct.pack("<i", 4 + len(title)))

        # --- Header block 3: NATOM ---
        f.write(struct.pack("<i", 4))
        f.write(struct.pack("<i", N_ATOMS))
        f.write(struct.pack("<i", 4))

        # --- Frame blocks ---
        rec_len = N_ATOMS * 4
        for k in range(n_frames):
            xs = (base_x + float(k)).astype(np.float32)
            for coord in (xs, base_y, base_z):
                f.write(struct.pack("<i", rec_len))
                f.write(coord.tobytes())
                f.write(struct.pack("<i", rec_len))


def _write_coor(path: Path):
    """NAMD .coor — 4-byte int little-endian natom, then 3*natom float64 xyz."""
    with open(path, "wb") as f:
        f.write(struct.pack("<i", N_ATOMS))
        xs = np.array([a["x"] for a in MASTER_ATOMS], dtype=np.float64)
        ys = np.array([a["y"] for a in MASTER_ATOMS], dtype=np.float64)
        zs = np.array([a["z"] for a in MASTER_ATOMS], dtype=np.float64)
        interleaved = np.empty(3 * N_ATOMS, dtype=np.float64)
        interleaved[0::3] = xs
        interleaved[1::3] = ys
        interleaved[2::3] = zs
        f.write(interleaved.tobytes())


def _write_mae(path: Path):
    """Minimal Maestro .mae with f_m_ct block containing m_atom, m_bond, ffio_ff."""
    lines = ["{\n", '  s_m_title\n', '  :::\n', '  "synthetic-master"\n', "}\n\n"]
    lines.append("f_m_ct {\n")
    lines.append("  s_m_title\n")
    lines.append("  :::\n")
    lines.append('  "synthetic"\n')

    # --- m_atom block ---
    lines.append("  m_atom[%d] {\n" % N_ATOMS)
    lines.append("    i_m_mmod_type\n")
    lines.append("    r_m_x_coord\n")
    lines.append("    r_m_y_coord\n")
    lines.append("    r_m_z_coord\n")
    lines.append("    i_m_residue_number\n")
    lines.append("    s_m_pdb_residue_name\n")
    lines.append("    s_m_pdb_atom_name\n")
    lines.append("    s_m_chain_name\n")
    lines.append("    i_m_atomic_number\n")
    lines.append("    r_m_pdb_occupancy\n")
    lines.append("    r_m_pdb_tfactor\n")
    lines.append("    s_m_grow_name\n")
    lines.append("    :::\n")
    for a in MASTER_ATOMS:
        atomic_num = _atomic_number_from_type(a["type"])
        lines.append(
            f"    {a['id']+1}  1  {a['x']:.4f}  {a['y']:.4f}  {a['z']:.4f}  "
            f"{a['resi']}  \"{a['resn']:<4s}\"  \"{a['name']:<4s}\"  \"{a['seg'][0]}\"  "
            f"{atomic_num}  {a['occ']:.2f}  {a['temp']:.2f}  \"{a['seg']}\"\n"
        )
    lines.append("    :::\n")
    lines.append("  }\n")

    # --- m_bond block ---
    if MASTER_BONDS:
        # Each bond twice (i->j and j->i) in MAE convention
        expanded = []
        for (i, j) in MASTER_BONDS:
            expanded.append((i, j))
            expanded.append((j, i))
        lines.append("  m_bond[%d] {\n" % len(expanded))
        lines.append("    i_m_from\n")
        lines.append("    i_m_to\n")
        lines.append("    i_m_order\n")
        lines.append("    :::\n")
        for k, (i, j) in enumerate(expanded, 1):
            lines.append(f"    {k}  {i}  {j}  1\n")
        lines.append("    :::\n")
        lines.append("  }\n")

    # --- ffio_ff block ---
    lines.append("  ffio_ff {\n")
    lines.append("    s_ffio_name\n")
    lines.append("    :::\n")
    lines.append('    "synthetic"\n')
    lines.append("    ffio_sites[%d] {\n" % N_ATOMS)
    lines.append("      s_ffio_type\n")
    lines.append("      r_ffio_charge\n")
    lines.append("      r_ffio_mass\n")
    lines.append("      :::\n")
    for a in MASTER_ATOMS:
        lines.append(
            f"      {a['id']+1}  \"{a['type']:<4s}\"  {a['charge']:.4f}  {a['mass']:.4f}\n"
        )
    lines.append("      :::\n")
    lines.append("    }\n")
    lines.append("  }\n")

    lines.append("}\n")
    path.write_text("".join(lines))


def _atomic_number_from_type(atype: str) -> int:
    """Rough atomic-number mapping for our small set of types."""
    m = {
        "CA": 6, "CT3": 6, "CC": 6,
        "HA3": 1, "H": 1, "HT": 1,
        "OC": 8, "OH1": 8, "OS": 8, "OT": 8,
        "NH2": 7,
        "POT": 19, "CLA": 17,
    }
    return m.get(atype, 0)


# ===========================================================================
# Session-scoped pytest fixtures — write each file once per test run
# ===========================================================================

@pytest.fixture(scope="session")
def synthetic_dir(tmp_path_factory) -> Path:
    d = tmp_path_factory.mktemp("synthetic")
    return d


@pytest.fixture(scope="session")
def synthetic_pdb(synthetic_dir) -> Path:
    p = synthetic_dir / "synth.pdb"
    _write_pdb(p)
    return p


@pytest.fixture(scope="session")
def synthetic_xyz(synthetic_dir) -> Path:
    p = synthetic_dir / "synth.xyz"
    _write_xyz(p)
    return p


@pytest.fixture(scope="session")
def synthetic_psf(synthetic_dir) -> Path:
    p = synthetic_dir / "synth.psf"
    _write_psf(p)
    return p


@pytest.fixture(scope="session")
def synthetic_dcd(synthetic_dir) -> Path:
    p = synthetic_dir / "synth.dcd"
    _write_dcd(p, n_frames=N_FRAMES)
    return p


@pytest.fixture(scope="session")
def synthetic_coor(synthetic_dir) -> Path:
    p = synthetic_dir / "synth.coor"
    _write_coor(p)
    return p


@pytest.fixture(scope="session")
def synthetic_mae(synthetic_dir) -> Path:
    p = synthetic_dir / "synth.mae"
    _write_mae(p)
    return p


# Convenience aggregator: every synthetic path in one dict (for combinatorial tests).
@pytest.fixture(scope="session")
def synthetic_files(
    synthetic_pdb, synthetic_xyz, synthetic_psf,
    synthetic_mae, synthetic_dcd, synthetic_coor,
):
    return {
        "pdb":  synthetic_pdb,
        "xyz":  synthetic_xyz,
        "psf":  synthetic_psf,
        "mae":  synthetic_mae,
        "dcd":  synthetic_dcd,
        "coor": synthetic_coor,
    }
