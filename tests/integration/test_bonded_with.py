"""
tests/integration/test_bonded_with.py
======================================
Integration tests for the bonded_with query feature.

Covers:
- All three shorthand forms (dict, list, tuple) produce identical results
- Bond filtering applies uniformly across every topology request type
  (global_ids, atom_names, charges, masses, ...) — not just global_ids
- Recursive bonded_with in neighbour sub-queries
- Depth ceiling (MAX_BONDED_WITH_DEPTH = 16) raises RecursionError cleanly
- Call-spanning neighbour cache: duplicate sub-queries resolve exactly once

These tests use the synthetic PSF. Every assertion is derived from the
master bond graph in conftest.py — if the master changes, expected values
are recomputed via the DEGREE_MAP and bond list, not re-hand-authored.
"""

from __future__ import annotations

import functools

import pytest

from trajectory_kit import sim as Sim
from trajectory_kit import _query_help as qh

from conftest import (
    MASTER_ATOMS, MASTER_BONDS, N_ATOMS, DEGREE_MAP,
    LIGAND_IDS, SOLV_IDS, IONS_IDS, OT_TYPE_IDS, HT_TYPE_IDS,
)


# ---------------------------------------------------------------------------
# Expected results derived from the master system (no hand-coded values)
# ---------------------------------------------------------------------------

# Atoms with exactly 1 bond ("terminal" atoms)
TERMINAL_IDS = sorted(i for i, d in DEGREE_MAP.items() if d == 1)

# Atoms with 0 bonds (isolated — the 4 ions)
ISOLATED_IDS = sorted(i for i, d in DEGREE_MAP.items() if d == 0)

# Atoms bonded to at least one OT (water oxygen) — these are HT atoms
def _bonded_to_ot_ids():
    ot_set = set(OT_TYPE_IDS)
    result = set()
    for (li, lj) in MASTER_BONDS:
        gi, gj = li - 1, lj - 1
        if gi in ot_set:
            result.add(gj)
        if gj in ot_set:
            result.add(gi)
    return sorted(result)

BONDED_TO_OT_IDS = _bonded_to_ot_ids()


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture(scope="session")
def sim_psf(synthetic_psf):
    return Sim(topology=synthetic_psf)


# ===========================================================================
# Shorthand — three forms must produce identical results
# ===========================================================================

class TestBondedWithShorthand:
    """bonded_with accepts (a) a bare dict, (b) a bare list of dicts, and
    (c) the full `(include, exclude)` tuple. All three must produce the
    same output for the same include-only query."""

    BLOCK = {"total": True, "count": {"eq": 1}}  # match terminal atoms

    def test_bare_dict_form(self, sim_psf):
        ids = sim_psf.get_topology(
            QUERY={"bonded_with": self.BLOCK},
            REQUEST="global_ids",
        )
        assert sorted(ids) == TERMINAL_IDS

    def test_bare_list_form(self, sim_psf):
        ids = sim_psf.get_topology(
            QUERY={"bonded_with": [self.BLOCK]},
            REQUEST="global_ids",
        )
        assert sorted(ids) == TERMINAL_IDS

    def test_full_tuple_form(self, sim_psf):
        ids = sim_psf.get_topology(
            QUERY={"bonded_with": ([self.BLOCK], [])},
            REQUEST="global_ids",
        )
        assert sorted(ids) == TERMINAL_IDS

    def test_all_three_forms_agree(self, sim_psf):
        q_dict  = {"bonded_with": self.BLOCK}
        q_list  = {"bonded_with": [self.BLOCK]}
        q_tuple = {"bonded_with": ([self.BLOCK], [])}
        r_dict  = sim_psf.get_topology(QUERY=q_dict,  REQUEST="global_ids")
        r_list  = sim_psf.get_topology(QUERY=q_list,  REQUEST="global_ids")
        r_tuple = sim_psf.get_topology(QUERY=q_tuple, REQUEST="global_ids")
        assert r_dict == r_list == r_tuple


# ===========================================================================
# Universal application across request types
# ===========================================================================

class TestBondedWithUniversal:
    """
    The bond filter must apply to every per-atom topology request — not
    just global_ids. Previously this was a silent footgun where only the
    global_ids branch saw the filter.
    """

    BLOCK = {"total": True, "count": {"eq": 1}}

    @pytest.mark.parametrize("request_string", [
        "global_ids", "local_ids", "residue_ids",
        "atom_names", "atom_types", "residue_names", "segment_names",
        "charges", "masses",
    ])
    def test_request_has_expected_length(self, sim_psf, request_string):
        result = sim_psf.get_topology(
            QUERY={"bonded_with": self.BLOCK},
            REQUEST=request_string,
        )
        assert len(result) == len(TERMINAL_IDS)

    def test_names_correspond_to_global_ids(self, sim_psf):
        """Cross-consistency: atom_names returned with the filter must be
        the names of the atoms whose global_ids are returned with the
        same filter."""
        ids   = sim_psf.get_topology(QUERY={"bonded_with": self.BLOCK}, REQUEST="global_ids")
        names = sim_psf.get_topology(QUERY={"bonded_with": self.BLOCK}, REQUEST="atom_names")
        expected_names = [MASTER_ATOMS[i]["name"] for i in ids]
        assert names == expected_names

    def test_charges_correspond_to_global_ids(self, sim_psf):
        ids     = sim_psf.get_topology(QUERY={"bonded_with": self.BLOCK}, REQUEST="global_ids")
        charges = sim_psf.get_topology(QUERY={"bonded_with": self.BLOCK}, REQUEST="charges")
        expected = [MASTER_ATOMS[i]["charge"] for i in ids]
        for c, e in zip(charges, expected):
            assert abs(c - e) < 1e-4


# ===========================================================================
# No-bond-filter path (regression: shorthand doesn't break the streaming path)
# ===========================================================================

class TestBondedWithNoFilter:

    def test_no_bonded_with_returns_all(self, sim_psf):
        ids = sim_psf.get_topology(QUERY={}, REQUEST="global_ids")
        assert len(ids) == N_ATOMS

    def test_none_bonded_with_treated_as_no_filter(self, sim_psf):
        # Explicit None should be accepted
        ids = sim_psf.get_topology(QUERY={"bonded_with": None}, REQUEST="global_ids")
        assert len(ids) == N_ATOMS

    def test_atom_type_only_no_bond_filter(self, sim_psf):
        ids = sim_psf.get_topology(QUERY={"atom_type": "OT"}, REQUEST="global_ids")
        assert sorted(ids) == OT_TYPE_IDS


# ===========================================================================
# Neighbour sub-queries (Level-2 bonded_with)
# ===========================================================================

class TestBondedWithNeighbour:

    def test_bonded_to_ot(self, sim_psf):
        """Atoms bonded to at least one OT (water oxygen)."""
        ids = sim_psf.get_topology(
            QUERY={"bonded_with": {
                "neighbor": {"atom_type": "OT"},
                "count": {"ge": 1},
            }},
            REQUEST="global_ids",
        )
        assert sorted(ids) == BONDED_TO_OT_IDS

    def test_bonded_to_ot_gives_ht_atoms(self, sim_psf):
        """Derived expectation: atoms bonded to water O should be HT hydrogens."""
        ids = sim_psf.get_topology(
            QUERY={"bonded_with": {
                "neighbor": {"atom_type": "OT"},
                "count": {"ge": 1},
            }},
            REQUEST="global_ids",
        )
        assert sorted(ids) == sorted(HT_TYPE_IDS)

    def test_isolated_atoms(self, sim_psf):
        """Degree-0 filter should find only the ions."""
        ids = sim_psf.get_topology(
            QUERY={"bonded_with": {"total": True, "count": {"eq": 0}}},
            REQUEST="global_ids",
        )
        assert sorted(ids) == ISOLATED_IDS
        assert sorted(ids) == IONS_IDS


# ===========================================================================
# Recursive bonded_with — graph-pattern matching
# ===========================================================================

class TestBondedWithRecursion:

    def test_two_hop_query_works(self, sim_psf):
        """
        "Atoms bonded to an OT that is itself bonded to ≥2 HT atoms."
        The inner predicate picks OT atoms (each has 2 HT neighbours).
        The outer picks the neighbours of those OT atoms — the HT atoms.
        """
        ids = sim_psf.get_topology(
            QUERY={
                "bonded_with": {
                    "neighbor": {
                        "atom_type": "OT",
                        "bonded_with": {
                            "neighbor": {"atom_type": "HT"},
                            "count": {"ge": 2},
                        },
                    },
                    "count": {"ge": 1},
                },
            },
            REQUEST="global_ids",
        )
        # Expected: all HT atoms (every OT in the system has 2 HT neighbours)
        assert sorted(ids) == sorted(HT_TYPE_IDS)

    def test_two_hop_matches_one_hop_when_predicate_trivial(self, sim_psf):
        """When the inner bonded_with is trivially True for all OT, the
        two-hop result must equal the one-hop result."""
        one_hop = sim_psf.get_topology(
            QUERY={"bonded_with": {
                "neighbor": {"atom_type": "OT"},
                "count": {"ge": 1},
            }},
            REQUEST="global_ids",
        )
        two_hop = sim_psf.get_topology(
            QUERY={
                "bonded_with": {
                    "neighbor": {
                        "atom_type": "OT",
                        "bonded_with": {"total": True, "count": {"ge": 0}},  # always true
                    },
                    "count": {"ge": 1},
                },
            },
            REQUEST="global_ids",
        )
        assert one_hop == two_hop


# ===========================================================================
# Depth limit — 16 succeeds, 17 raises RecursionError
# ===========================================================================

def _build_nested_query(depth: int, leaf_type: str = "OT") -> dict:
    """Build a query nested ``depth`` levels deep. Each level wraps the
    previous query inside a bonded_with neighbour."""
    q = {"atom_type": leaf_type}
    for _ in range(depth):
        q = {"atom_type": leaf_type,
             "bonded_with": {"neighbor": q, "count": {"ge": 0}}}
    return q


class TestBondedWithDepthLimit:

    def test_depth_16_succeeds(self, sim_psf):
        # Outer bonded_with + 15 nested = 16 total depth
        inner = _build_nested_query(15, leaf_type="OT")
        sim_psf.get_topology(
            QUERY={"bonded_with": {"neighbor": inner, "count": {"ge": 0}}},
            REQUEST="global_ids",
        )  # should not raise

    def test_depth_17_raises_recursion_error(self, sim_psf):
        # Outer + 16 nested = 17 total
        inner = _build_nested_query(16, leaf_type="OT")
        with pytest.raises(RecursionError, match="max depth"):
            sim_psf.get_topology(
                QUERY={"bonded_with": {"neighbor": inner, "count": {"ge": 0}}},
                REQUEST="global_ids",
            )

    def test_depth_constant_matches_module(self):
        assert qh.MAX_BONDED_WITH_DEPTH == 16


# ===========================================================================
# Call-spanning neighbour cache
# ===========================================================================

class TestNeighbourCache:
    """
    When two bonded_with blocks reference the same neighbour predicate,
    the predicate must be resolved exactly once — the cache is reused
    across recursion within a single user-facing call.
    """

    def test_identical_subqueries_resolved_once(self, synthetic_psf):
        from trajectory_kit import psf_parse

        call_depths = []
        original = psf_parse._get_topology_query_psf

        @functools.wraps(original)
        def counting(*args, **kwargs):
            call_depths.append(kwargs.get("_bonded_with_depth", 0))
            return original(*args, **kwargs)

        # Patch, run, restore
        psf_parse._get_topology_query_psf = counting
        try:
            s = Sim(topology=synthetic_psf)
            s.get_topology(
                QUERY={
                    "bonded_with": [
                        {"neighbor": {"atom_type": "OT"}, "count": {"ge": 1}},
                        {"neighbor": {"atom_type": "OT"}, "count": {"le": 5}},
                    ],
                },
                REQUEST="global_ids",
            )
        finally:
            psf_parse._get_topology_query_psf = original

        # Exactly one recursive call (depth >= 1) — the second block's
        # identical neighbour predicate hits the cache.
        recursive_calls = [d for d in call_depths if d >= 1]
        assert len(recursive_calls) == 1, (
            f"Expected 1 cache-missing recursive call, got {len(recursive_calls)} "
            f"at depths {call_depths}"
        )
