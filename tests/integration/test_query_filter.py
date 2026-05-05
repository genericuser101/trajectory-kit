from __future__ import annotations

import pytest

from trajectory_kit import sim as Sim

from conftest import (
    N_ATOMS, LIGAND_IDS, SOLV_IDS, IONS_IDS, MASTER_ATOMS,
)


# ===========================================================================
# global_ids filtering
# ===========================================================================

class TestGlobalIdsFilter:
    """Every parser that declares ``global_ids`` as a valid keyword must
    actually filter by it. This test exercises all five sim configurations
    that expose a ``global_ids``-capable query domain."""

    IDS = [0, 1, 2, 3, 4]  # first 5 atoms of the ligand

    def test_pdb_typing_filters_by_global_ids(self, synthetic_pdb):
        s = Sim(typing=synthetic_pdb)
        ids = s.get_types(
            QUERY={"global_ids": (self.IDS, set())},
            REQUEST="global_ids",
        )
        assert sorted(ids) == self.IDS

    def test_xyz_typing_filters_by_global_ids(self, synthetic_xyz):
        s = Sim(typing=synthetic_xyz)
        ids = s.get_types(
            QUERY={"global_ids": (self.IDS, set())},
            REQUEST="global_ids",
        )
        assert sorted(ids) == self.IDS

    def test_psf_topology_filters_by_global_ids(self, synthetic_psf):
        s = Sim(topology=synthetic_psf)
        ids = s.get_topology(
            QUERY={"global_ids": (self.IDS, set())},
            REQUEST="global_ids",
        )
        assert sorted(ids) == self.IDS

    def test_mae_typing_filters_by_global_ids(self, synthetic_mae):
        s = Sim(typing=synthetic_mae)
        ids = s.get_types(
            QUERY={"global_ids": (self.IDS, set())},
            REQUEST="global_ids",
        )
        assert sorted(ids) == self.IDS

    def test_mae_topology_filters_by_global_ids(self, synthetic_mae):
        s = Sim(topology=synthetic_mae)
        ids = s.get_topology(
            QUERY={"global_ids": (self.IDS, set())},
            REQUEST="global_ids",
        )
        assert sorted(ids) == self.IDS

    # global_ids applies uniformly across request types 

    def test_global_ids_filter_applies_to_atom_names(self, synthetic_pdb):
        """The filter must apply to every per-atom request, not just global_ids."""
        s = Sim(typing=synthetic_pdb)
        names = s.get_types(
            QUERY={"global_ids": ([0, 1, 2], set())},
            REQUEST="atom_names",
        )
        expected = [MASTER_ATOMS[i]["name"] for i in [0, 1, 2]]
        assert names == expected

    def test_global_ids_filter_applies_to_charges(self, synthetic_psf):
        s = Sim(topology=synthetic_psf)
        charges = s.get_topology(
            QUERY={"global_ids": ([0, 1, 2], set())},
            REQUEST="charges",
        )
        expected = [MASTER_ATOMS[i]["charge"] for i in [0, 1, 2]]
        for c, e in zip(charges, expected):
            assert abs(c - e) < 1e-4

    # range and membership semantics 

    def test_global_ids_as_range_tuple(self, synthetic_pdb):
        """``(0, 4)`` is a range — atoms 0 through 4 inclusive."""
        s = Sim(typing=synthetic_pdb)
        ids = s.get_types(
            QUERY={"global_ids": (0, 4)},
            REQUEST="global_ids",
        )
        assert sorted(ids) == [0, 1, 2, 3, 4]

    def test_global_ids_as_membership_list(self, synthetic_pdb):
        """``[0, 4, 10]`` is membership — exactly those three atoms."""
        s = Sim(typing=synthetic_pdb)
        ids = s.get_types(
            QUERY={"global_ids": [0, 4, 10]},
            REQUEST="global_ids",
        )
        assert sorted(ids) == [0, 4, 10]

    def test_global_ids_exclude(self, synthetic_pdb):
        """Exclude-side works too: `(None, excluded_list)` removes those ids."""
        s = Sim(typing=synthetic_pdb)
        ids = s.get_types(
            QUERY={"global_ids": (None, IONS_IDS)},  # exclude the 4 ions
            REQUEST="global_ids",
        )
        assert len(ids) == N_ATOMS - len(IONS_IDS)
        for ion_id in IONS_IDS:
            assert ion_id not in ids

    # Cross-domain intersection (the fetch() flow that was broken)

    def test_cross_domain_intersection_resolves(self, synthetic_pdb, synthetic_psf):
        """fetch() with global_ids on both domains should intersect them.
        The intersection count must appear in selection.resolved_count and
        both payloads must reflect the intersection."""
        s = Sim(typing=synthetic_pdb, topology=synthetic_psf)
        r = s.fetch(
            TYPE_Q={"global_ids": ([0, 1, 2, 3, 4], set())},
            TYPE_R="atom_names",
            TOPO_Q={"global_ids": ([2, 3, 4, 5, 6], set())},
            TOPO_R="charges",
            devFlag=True,
        )
        # {0..4} & {2..6} = {2, 3, 4}
        assert r["selection"]["resolved_count"] == 3
        assert len(r["payload"]["typing"]) == 3
        assert len(r["payload"]["topology"]) == 3

    # No filter regression check

    def test_no_global_ids_query_returns_all(self, synthetic_pdb):
        s = Sim(typing=synthetic_pdb)
        ids = s.get_types(QUERY={}, REQUEST="global_ids")
        assert len(ids) == N_ATOMS


# ===========================================================================
# (None, None) normalises to "no constraint"
# ===========================================================================

class TestNoneNoneNormalisation:
    """``(None, None)`` is an unbounded range which semantically means
    'no constraint'. Previously on the exclude side it was interpreted as
    'exclude every value' — a severe footgun where the user's intent
    (no exclusion) produced the opposite result (exclude everything).

    Fix: ``(None, None)`` on either side normalises to empty.
    """

    def test_range_with_none_none_exclude_matches_range(self, synthetic_pdb):
        """``residue_ids = ((1, 1), (None, None))`` should match residue 1,
        not zero atoms."""
        s = Sim(typing=synthetic_pdb)
        ids = s.get_types(
            QUERY={"residue_ids": ((1, 1), (None, None))},
            REQUEST="global_ids",
        )
        # Residue 1 is the entire ligand
        assert sorted(ids) == LIGAND_IDS

    def test_three_forms_of_no_exclude_equivalent(self, synthetic_pdb):
        """None, (), and (None, None) as exclude must all mean the same thing."""
        s = Sim(typing=synthetic_pdb)
        r1 = s.get_types(QUERY={"residue_ids": (1, 1)},                  REQUEST="global_ids")
        r2 = s.get_types(QUERY={"residue_ids": ((1, 1), None)},          REQUEST="global_ids")
        r3 = s.get_types(QUERY={"residue_ids": ((1, 1), ())},            REQUEST="global_ids")
        r4 = s.get_types(QUERY={"residue_ids": ((1, 1), (None, None))},  REQUEST="global_ids")
        assert sorted(r1) == sorted(r2) == sorted(r3) == sorted(r4)
        assert len(r1) > 0   # non-empty — the bug would have returned []

    def test_bare_none_none_matches_everything(self, synthetic_pdb):
        """``(None, None)`` as a bare query is an unbounded range = match all."""
        s = Sim(typing=synthetic_pdb)
        ids = s.get_types(QUERY={"x": (None, None)}, REQUEST="global_ids")
        assert len(ids) == N_ATOMS

    def test_none_none_include_side_also_normalises(self, synthetic_pdb):
        """Symmetry: ``((None, None), [99])`` should match all-except-99."""
        s = Sim(typing=synthetic_pdb)
        ids = s.get_types(
            QUERY={"residue_ids": ((None, None), [1])},  # include all, exclude residue 1
            REQUEST="global_ids",
        )
        for lig_id in LIGAND_IDS:
            assert lig_id not in ids
        assert len(ids) == N_ATOMS - len(LIGAND_IDS)
