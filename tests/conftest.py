"""
conftest.py — trajectory-kit test configuration
================================================
1. Machine-aware large file resolution via test_paths.json (git-ignored).
2. Synthetic minimal file fixtures for logic tests.
3. Shared sim fixtures for integration tests.

SETUP (one-time per machine):
  Copy test_paths.template.json → test_paths.json and fill in your paths.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Machine-aware path loading
# ---------------------------------------------------------------------------

_PATHS_FILE = Path(__file__).parent / "test_paths.json"


def _load_machine_paths() -> dict:
    if not _PATHS_FILE.exists():
        return {}
    with open(_PATHS_FILE) as f:
        return json.load(f)


_MACHINE_PATHS = _load_machine_paths()


def get_real_path(key: str) -> Path | None:
    raw = _MACHINE_PATHS.get(key)
    if raw is None:
        return None
    p = Path(raw)
    return p if p.exists() else None


def requires_real_file(*keys: str):
    missing = [k for k in keys if get_real_path(k) is None]
    return pytest.mark.skipif(
        bool(missing),
        reason=f"Real test file(s) not available on this machine: {missing}. "
               f"Add paths to tests/test_paths.json.",
    )


# ---------------------------------------------------------------------------
# Synthetic PDB  (4 ATOM records, 2 segments, 3 residues)
#
#   global_id 0  CA  ALA  PROT  resid=1  x=1   y=2   z=3   occ=1.00  temp=0.10
#   global_id 1  CB  ALA  PROT  resid=1  x=4   y=5   z=6   occ=1.00  temp=0.20
#   global_id 2  CA  GLY  PROT  resid=2  x=7   y=8   z=9   occ=1.00  temp=0.30
#   global_id 3  OW  TIP  SOLV  resid=3  x=10  y=11  z=12  occ=1.00  temp=0.40
#
# Column layout matches _parse_pdb_atom_row exactly:
#   cols  6:11  serial   12:16  name   17:21  resname   22:26  resseq
#   cols 30:38  x        38:46  y      46:54  z
#   cols 54:60  occupancy        60:66  tempfactor       72:76  segid
# ---------------------------------------------------------------------------

_SYNTHETIC_PDB = """\
ATOM      1  CA  ALA A   1       1.000   2.000   3.000  1.00  0.10      PROT
ATOM      2  CB  ALA A   1       4.000   5.000   6.000  1.00  0.20      PROT
ATOM      3  CA  GLY A   2       7.000   8.000   9.000  1.00  0.30      PROT
ATOM      4  OW  TIP B   3      10.000  11.000  12.000  1.00  0.40      SOLV
END
"""


@pytest.fixture(scope="session")
def synthetic_pdb(tmp_path_factory) -> Path:
    p = tmp_path_factory.mktemp("data") / "synthetic.pdb"
    p.write_text(_SYNTHETIC_PDB, encoding="ascii")
    return p


# ---------------------------------------------------------------------------
# Synthetic PSF  (4 atoms, 3 residues, 3 bonds: 0-1, 1-2, 2-3)
#
#   global_id 0  CA  CT1  ALA  PROT  resid=1  charge=-0.270  mass=12.011
#   global_id 1  CB  CT3  ALA  PROT  resid=1  charge=-0.270  mass=12.011
#   global_id 2  CA  CT2  GLY  PROT  resid=2  charge=-0.180  mass=12.011
#   global_id 3  OW  OT   TIP  SOLV  resid=3  charge=-0.834  mass=15.999
# ---------------------------------------------------------------------------

_SYNTHETIC_PSF = """\
PSF

       1 !NTITLE
 REMARKS synthetic test PSF

       4 !NATOM
       1 PROT     1 ALA  CA   CT1   -0.270   12.011           0
       2 PROT     1 ALA  CB   CT3   -0.270   12.011           0
       3 PROT     2 GLY  CA   CT2   -0.180   12.011           0
       4 SOLV     3 TIP  OW   OT    -0.834   15.999           0

       3 !NBOND: bonds
       1       2       2       3       3       4

       0 !NTHETA: angles

       0 !NPHI: dihedrals

       0 !NIMPHI: impropers

"""


@pytest.fixture(scope="session")
def synthetic_psf(tmp_path_factory) -> Path:
    p = tmp_path_factory.mktemp("data") / "synthetic.psf"
    p.write_text(_SYNTHETIC_PSF, encoding="ascii")
    return p


# ---------------------------------------------------------------------------
# Synthetic XYZ  (4 atoms, single frame)
#
#   global_id 0  C   x=1   y=2   z=3
#   global_id 1  C   x=4   y=5   z=6
#   global_id 2  N   x=7   y=8   z=9
#   global_id 3  O   x=10  y=11  z=12
# ---------------------------------------------------------------------------

_SYNTHETIC_XYZ = """\
4
synthetic test frame
C    1.000   2.000   3.000
C    4.000   5.000   6.000
N    7.000   8.000   9.000
O   10.000  11.000  12.000
"""


@pytest.fixture(scope="session")
def synthetic_xyz(tmp_path_factory) -> Path:
    p = tmp_path_factory.mktemp("data") / "synthetic.xyz"
    p.write_text(_SYNTHETIC_XYZ, encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# Synthetic DCD  (4 atoms, 3 frames, little-endian, no unit cell)
#
#   frame 0: base positions       (atom 0 x=1,  atom 3 x=10)
#   frame 1: base + 1.0           (atom 0 x=2,  atom 3 x=11)
#   frame 2: base + 2.0           (atom 0 x=3,  atom 3 x=12)
# ---------------------------------------------------------------------------

def _write_synthetic_dcd(path: Path, n_atoms: int = 4, n_frames: int = 3):
    def record(payload: bytes) -> bytes:
        length = struct.pack("<i", len(payload))
        return length + payload + length

    icntrl     = [0] * 20
    icntrl[0]  = n_frames
    icntrl[8]  = 0
    icntrl[10] = 0
    hdr_payload = b"CORD" + struct.pack("<20i", *icntrl)
    hdr_payload += b"\x00" * (84 - len(hdr_payload))

    title_payload = struct.pack("<i", 1) + b"synthetic dcd                   "
    natom_payload = struct.pack("<i", n_atoms)

    with open(path, "wb") as f:
        f.write(record(hdr_payload))
        f.write(record(title_payload))
        f.write(record(natom_payload))
        for frame_i in range(n_frames):
            offset = float(frame_i)
            xs = np.array([1.0 + offset, 4.0 + offset,  7.0 + offset, 10.0 + offset], dtype=np.float32)
            ys = np.array([2.0 + offset, 5.0 + offset,  8.0 + offset, 11.0 + offset], dtype=np.float32)
            zs = np.array([3.0 + offset, 6.0 + offset,  9.0 + offset, 12.0 + offset], dtype=np.float32)
            f.write(record(xs.tobytes()))
            f.write(record(ys.tobytes()))
            f.write(record(zs.tobytes()))


@pytest.fixture(scope="session")
def synthetic_dcd(tmp_path_factory) -> Path:
    p = tmp_path_factory.mktemp("data") / "synthetic.dcd"
    _write_synthetic_dcd(p, n_atoms=4, n_frames=3)
    return p


# ---------------------------------------------------------------------------
# Mismatched fixtures — wrong atom count, used by test_metadata.py
# ---------------------------------------------------------------------------

@pytest.fixture
def mismatched_dcd(tmp_path) -> Path:
    """DCD with 99 atoms — intentionally wrong vs the 4-atom typing files."""
    p = tmp_path / "mismatch.dcd"
    _write_synthetic_dcd(p, n_atoms=99, n_frames=2)
    return p


@pytest.fixture
def mismatched_psf(tmp_path) -> Path:
    """PSF with 99 atoms — intentionally wrong vs the 4-atom typing files."""
    lines = ["PSF\n\n       1 !NTITLE\n REMARKS mismatch psf\n\n      99 !NATOM\n"]
    for i in range(1, 100):
        lines.append(f"  {i:6d} SEG      1 ALA  CA   CT1   -0.270   12.011           0\n")
    lines.append("\n       0 !NBOND: bonds\n\n")
    p = tmp_path / "mismatch.psf"
    p.write_text("".join(lines), encoding="ascii")
    return p


# ---------------------------------------------------------------------------
# Convenience: sim fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def sim_synthetic(synthetic_pdb, synthetic_psf, synthetic_dcd):
    from trajectory_kit import sim
    return sim(
        typing=synthetic_pdb,
        topology=synthetic_psf,
        trajectory=synthetic_dcd,
    )


@pytest.fixture(scope="session")
def sim_synthetic_xyz(synthetic_xyz, synthetic_psf, synthetic_dcd):
    from trajectory_kit import sim
    return sim(
        typing=synthetic_xyz,
        topology=synthetic_psf,
        trajectory=synthetic_dcd,
    )


@pytest.fixture(scope="session")
def sim_pdb_only(synthetic_pdb):
    """sim with only a PDB typing file — no topology, no trajectory."""
    from trajectory_kit import sim
    return sim(typing=synthetic_pdb)


@pytest.fixture(scope="session")
def sim_xyz_only(synthetic_xyz):
    """sim with only an XYZ typing file — no topology, no trajectory."""
    from trajectory_kit import sim
    return sim(typing=synthetic_xyz)


@pytest.fixture(scope="session")
def sim_real():
    pdb = get_real_path("pdb")
    psf = get_real_path("psf")
    dcd = get_real_path("dcd")
    if not all([pdb, psf, dcd]):
        pytest.skip("Real simulation files not available on this machine.")
    from trajectory_kit import sim
    return sim(
        typing=pdb,
        topology=psf,
        trajectory=dcd,
    )
