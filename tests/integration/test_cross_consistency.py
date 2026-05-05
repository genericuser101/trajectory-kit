"""
tests/integration/test_cross_consistency.py
============================================

The main integration harness: a test matrix that enumerates valid
(typing, topology, trajectory) sim configurations, collects outputs for
every canonical field each config supports, and cross-checks them for
equality.

The key property being tested: **the same canonical field, queried via
different file-format backends built from the same master data, must
produce identical output.**

No hard-coded expected values. When the master data in conftest.py changes,
these tests auto-follow. When a parser produces wrong output, the test
reports which (canonical_field, config) diverged from the rest.

Canonical fields are grouped by domain:
- typing_fields      : available from typing-domain parsers
- topology_fields    : available from topology-domain parsers
- trajectory_fields  : available from trajectory-domain parsers
- cross_domain_fields: same physical quantity exposed across domains
  (e.g. positions at frame 0 from typing/PDB vs trajectory/COOR)

Some fields have different names in different formats (e.g. PSF's
"atom_types" == MAE topology's "vdw_types"); the canonical map
normalises those.
"""

from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pytest

from trajectory_kit import sim as Sim

from conftest import N_ATOMS, N_FRAMES


# ---------------------------------------------------------------------------
# Sim configurations — every meaningful combination of backends
# ---------------------------------------------------------------------------
# A configuration is (label, {typing?, topology?, trajectory?}). sim requires
# at least one of the three. MAE can serve as BOTH typing and topology.

def _build_configs(files: dict) -> dict:
    return {
        # Single-backend loads
        "pdb":          {"typing": files["pdb"]},
        "xyz":          {"typing": files["xyz"]},
        "psf":          {"topology": files["psf"]},
        "mae_typing":   {"typing": files["mae"]},
        "mae_both":     {"typing": files["mae"], "topology": files["mae"]},
        "dcd":          {"trajectory": files["dcd"]},
        "coor":         {"trajectory": files["coor"]},

        # Typing + topology pairs
        "pdb+psf":      {"typing": files["pdb"], "topology": files["psf"]},
        "xyz+psf":      {"typing": files["xyz"], "topology": files["psf"]},

        # Full stack
        "pdb+psf+dcd":  {"typing": files["pdb"], "topology": files["psf"], "trajectory": files["dcd"]},
        "pdb+psf+coor": {"typing": files["pdb"], "topology": files["psf"], "trajectory": files["coor"]},
        "xyz+psf+dcd":  {"typing": files["xyz"], "topology": files["psf"], "trajectory": files["dcd"]},
        "mae+dcd":      {"typing": files["mae"], "topology": files["mae"], "trajectory": files["dcd"]},
        "mae+coor":     {"typing": files["mae"], "topology": files["mae"], "trajectory": files["coor"]},
    }


# ---------------------------------------------------------------------------
# Canonical field map
# ---------------------------------------------------------------------------
# Each entry: canonical_name -> list of (domain, request_string_in_that_domain).
# When comparing across configs we evaluate each (domain, request) pair and
# all non-None results across configs must be equal.

# Typing-domain fields shared across PDB / XYZ / MAE (typing)
# Note: local_ids is intentionally omitted — it is format-specific (XYZ uses
# 0-based computational convention, PDB/MAE use 1-based file-serial
# convention). That's documented behaviour, not a bug.
TYPING_FIELDS = {
    "typing.global_ids":     ("typing", "global_ids"),
    "typing.atom_names":     ("typing", "atom_names"),
    "typing.x":              ("typing", "x"),
    "typing.y":              ("typing", "y"),
    "typing.z":              ("typing", "z"),
    "typing.positions":      ("typing", "positions"),
    # Residue / segment — PDB and MAE only; XYZ doesn't have these
    "typing.residue_names":  ("typing", "residue_names"),
    "typing.residue_ids":    ("typing", "residue_ids"),
    "typing.segment_names":  ("typing", "segment_names"),
}

# Topology-domain fields — PSF vs MAE (topology)
# Same exclusion: local_ids is file-specific indexing.
TOPOLOGY_FIELDS = {
    "topology.global_ids":     ("topology", "global_ids"),
    "topology.atom_names":     ("topology", "atom_names"),
    "topology.residue_ids":    ("topology", "residue_ids"),
    "topology.residue_names":  ("topology", "residue_names"),
    "topology.segment_names":  ("topology", "segment_names"),
    "topology.charges":        ("topology", "charges"),
    "topology.masses":         ("topology", "masses"),
    # atom types under two names, merged via the alias map below
    "topology.atom_types":     ("topology", "atom_types"),
    "topology.vdw_types":      ("topology", "vdw_types"),
}

# Trajectory-domain (always positions at frame 0)
TRAJECTORY_FIELDS = {
    "trajectory.positions_f0": ("trajectory", "positions"),
}

# Name aliases: canonical -> list of equivalent canonical names to merge.
# e.g. PSF produces topology.atom_types but MAE produces topology.vdw_types
# for the same physical quantity; we merge them before comparing.
FIELD_ALIASES = {
    "topology.atom_types_or_vdw_types": [
        "topology.atom_types",
        "topology.vdw_types",
    ],
}

# Cross-domain equivalences: positions from typing (PDB/XYZ/MAE) at implicit
# frame 0 must equal positions from trajectory (DCD frame 0, COOR) within tol.
CROSS_DOMAIN_POSITION_FIELDS = [
    ("typing", "positions"),
    ("trajectory", "positions"),
]


# ---------------------------------------------------------------------------
# Sim probing utilities
# ---------------------------------------------------------------------------

def _domain_supports_request(sim: Sim, domain: str, request: str) -> bool:
    """Check if a domain's parser advertises the given request string."""
    reqs_attr = {
        "typing":     "type_file_reqs",
        "topology":   "topo_file_reqs",
        "trajectory": "traj_file_reqs",
    }[domain]
    file_attr = {
        "typing":     "type_file",
        "topology":   "top_file",
        "trajectory": "traj_file",
    }[domain]
    if getattr(sim, file_attr) is None:
        return False
    reqs = getattr(sim, reqs_attr)
    return (reqs is not None) and (request in reqs)


def _query_domain(sim: Sim, domain: str, request: str):
    """Issue a single-domain query, return the raw output (or None on error).
    Queries are empty (match all atoms) unless the domain is trajectory, in
    which case we pass global_ids covering all atoms. ``frame_interval`` is
    added only if the trajectory backend advertises support for it (DCD yes,
    COOR no)."""
    if domain == "typing":
        return sim.get_types(QUERY={}, REQUEST=request)
    if domain == "topology":
        return sim.get_topology(QUERY={}, REQUEST=request)
    if domain == "trajectory":
        traj_query: dict = {"global_ids": (list(range(N_ATOMS)), set())}
        if sim.traj_file_keys and "frame_interval" in sim.traj_file_keys:
            traj_query["frame_interval"] = (0, 0, 1)
        out = sim.get_trajectory(QUERY=traj_query, REQUEST=request)
        # get_trajectory returns (ndarray, meta); normalise to ndarray for compare
        if isinstance(out, tuple):
            return out[0]
        return out
    raise ValueError(f"Unknown domain {domain!r}")


def _canonical_equal(a, b, *, atol=1e-4) -> bool:
    """Equality used across configs — array-aware, tolerant for floats."""
    if a is None or b is None:
        return a is b
    # ndarray path — shape must match, values close
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        a_arr = np.asarray(a)
        b_arr = np.asarray(b)
        if a_arr.shape != b_arr.shape:
            return False
        if a_arr.dtype.kind in "fc" or b_arr.dtype.kind in "fc":
            return np.allclose(a_arr, b_arr, atol=atol, equal_nan=True)
        return np.array_equal(a_arr, b_arr)
    # list of floats
    if isinstance(a, list) and a and isinstance(a[0], float):
        if len(a) != len(b):
            return False
        return all(abs(x - y) <= atol for x, y in zip(a, b))
    return a == b


# ---------------------------------------------------------------------------
# Result collection — one pass per session
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def matrix(synthetic_files):
    """
    Build the result matrix.

    Returns a dict: canonical_field -> {config_label: value}

    Configs that don't expose the field are simply absent from that inner
    dict. Tests then verify that where a field is defined in multiple
    configs, all values agree.
    """
    configs = _build_configs(synthetic_files)

    # All canonical fields from all groups
    all_fields = {}
    all_fields.update(TYPING_FIELDS)
    all_fields.update(TOPOLOGY_FIELDS)
    all_fields.update(TRAJECTORY_FIELDS)

    results: dict[str, dict[str, object]] = {f: {} for f in all_fields}

    for label, kwargs in configs.items():
        try:
            s = Sim(**kwargs)
        except Exception as e:
            pytest.fail(f"Sim construction failed for {label!r}: {e}")
        for canon, (domain, req) in all_fields.items():
            if _domain_supports_request(s, domain, req):
                try:
                    results[canon][label] = _query_domain(s, domain, req)
                except Exception as e:
                    results[canon][label] = f"ERROR: {type(e).__name__}: {e}"

    return results


# ===========================================================================
# Tests — one per canonical field group
# ===========================================================================

class TestTypingFieldsConsistent:
    """Every typing field that appears in >=2 configs must agree across them."""

    @pytest.mark.parametrize("canonical", sorted(TYPING_FIELDS))
    def test_field_agrees_across_configs(self, matrix, canonical):
        by_config = matrix[canonical]
        if len(by_config) < 2:
            pytest.skip(f"{canonical} only supported by {len(by_config)} config(s)")
        # No config should have errored
        for label, val in by_config.items():
            assert not (isinstance(val, str) and val.startswith("ERROR:")), \
                f"{canonical} on {label}: {val}"
        # All values equal
        items = list(by_config.items())
        ref_label, ref_val = items[0]
        for other_label, other_val in items[1:]:
            assert _canonical_equal(ref_val, other_val), (
                f"{canonical} disagreement between {ref_label!r} and {other_label!r}\n"
                f"  {ref_label}={ref_val!r}\n"
                f"  {other_label}={other_val!r}"
            )


class TestTopologyFieldsConsistent:

    @pytest.mark.parametrize("canonical", sorted(TOPOLOGY_FIELDS))
    def test_field_agrees_across_configs(self, matrix, canonical):
        by_config = matrix[canonical]
        if len(by_config) < 2:
            pytest.skip(f"{canonical} only supported by {len(by_config)} config(s)")
        for label, val in by_config.items():
            assert not (isinstance(val, str) and val.startswith("ERROR:")), \
                f"{canonical} on {label}: {val}"
        items = list(by_config.items())
        ref_label, ref_val = items[0]
        for other_label, other_val in items[1:]:
            assert _canonical_equal(ref_val, other_val), (
                f"{canonical} disagreement between {ref_label!r} and {other_label!r}\n"
                f"  {ref_label}={ref_val!r}\n"
                f"  {other_label}={other_val!r}"
            )


class TestFieldAliasesConsistent:
    """Aliased fields — e.g. PSF.atom_types vs MAE.vdw_types — must agree."""

    @pytest.mark.parametrize("alias_name,canonical_list", FIELD_ALIASES.items())
    def test_aliased_fields_agree(self, matrix, alias_name, canonical_list):
        merged: dict[str, object] = {}
        for canon in canonical_list:
            for label, val in matrix[canon].items():
                # Tag with canon name so we know the original label
                merged[f"{canon}[{label}]"] = val
        if len(merged) < 2:
            pytest.skip(f"{alias_name} only supported in {len(merged)} config(s)")
        items = list(merged.items())
        ref_label, ref_val = items[0]
        for other_label, other_val in items[1:]:
            assert _canonical_equal(ref_val, other_val), (
                f"Alias {alias_name} disagreement:\n"
                f"  {ref_label}={ref_val!r}\n"
                f"  {other_label}={other_val!r}"
            )


class TestTrajectoryFieldsConsistent:
    """Positions returned from DCD frame-0 must equal positions from COOR."""

    def test_positions_frame_0_consistent(self, matrix):
        by_config = matrix["trajectory.positions_f0"]
        if len(by_config) < 2:
            pytest.skip("Only one trajectory backend")
        items = list(by_config.items())
        ref_label, ref_val = items[0]
        for other_label, other_val in items[1:]:
            assert _canonical_equal(ref_val, other_val), (
                f"Frame-0 positions disagree: {ref_label} vs {other_label}"
            )


class TestCrossDomainPositions:
    """
    The ultimate consistency check: positions from typing (PDB/XYZ/MAE) at
    their implicit frame 0 must equal trajectory positions at frame 0 from
    DCD / COOR. Because all files are derived from the same master.
    """

    def test_all_position_sources_agree(self, matrix):
        sources = {}
        # From typing.positions (shape (1, n, 3))
        for label, val in matrix["typing.positions"].items():
            if isinstance(val, np.ndarray):
                sources[f"typing[{label}]"] = val
        # From trajectory.positions_f0 (shape (1, n, 3))
        for label, val in matrix["trajectory.positions_f0"].items():
            if isinstance(val, np.ndarray):
                sources[f"trajectory[{label}]"] = val
        if len(sources) < 2:
            pytest.skip("Need at least 2 position sources")
        items = list(sources.items())
        ref_label, ref_val = items[0]
        for other_label, other_val in items[1:]:
            assert _canonical_equal(ref_val, other_val, atol=1e-3), (
                f"Position source mismatch: {ref_label} vs {other_label}\n"
                f"  ref shape = {ref_val.shape}\n"
                f"  other shape = {other_val.shape}"
            )


# ---------------------------------------------------------------------------
# Structural smoke tests — every config must at least build and report
# the correct atom count. No cross-field comparison, just basic liveness.
# ---------------------------------------------------------------------------

class TestEveryConfigBuilds:

    @pytest.mark.parametrize("label", [
        "pdb", "xyz", "psf", "mae_typing", "mae_both", "dcd", "coor",
        "pdb+psf", "xyz+psf",
        "pdb+psf+dcd", "pdb+psf+coor", "xyz+psf+dcd",
        "mae+dcd", "mae+coor",
    ])
    def test_config_builds_and_reports_atom_count(self, synthetic_files, label):
        cfg = _build_configs(synthetic_files)[label]
        s = Sim(**cfg)
        assert s.global_system_properties.get("num_atoms") == N_ATOMS
