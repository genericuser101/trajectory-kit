"""
test_topology.py
================
Complete test suite for the topology domain (PSF file type).
Covers every available request.

Atom layout in synthetic PSF:
    global_id 0  CA   CT1  ALA  PROT  charge=-0.270  mass=12.011  degree=1
    global_id 1  CB   CT3  ALA  PROT  charge=-0.270  mass=12.011  degree=2
    global_id 2  CA   CT2  GLY  PROT  charge=-0.180  mass=12.011  degree=2
    global_id 3  OW   OT   TIP  SOLV  charge=-0.834  mass=15.999  degree=1

Bonds: 0-1, 1-2, 2-3
"""

from __future__ import annotations

import pytest

# ===========================================================================
# PSF — request: global_ids
# ===========================================================================

class TestPSFGlobalIds:

    def test_empty_query_returns_all_atoms(self, sim_synthetic):
        ids = sim_synthetic.get_topology(QUERY={}, REQUEST="global_ids")
        assert sorted(ids) == [0, 1, 2, 3]

    def test_atom_name_include(self, sim_synthetic):
        ids = sim_synthetic.get_topology(
            QUERY={"atom_name": ({"CA"}, set())}, REQUEST="global_ids"
        )
        assert sorted(ids) == [0, 2]

    def test_atom_name_exclude(self, sim_synthetic):
        ids = sim_synthetic.get_topology(
            QUERY={"atom_name": (set(), {"OW"})}, REQUEST="global_ids"
        )
        assert sorted(ids) == [0, 1, 2]

    def test_atom_type_include(self, sim_synthetic):
        ids = sim_synthetic.get_topology(
            QUERY={"atom_type": ({"OT"}, set())}, REQUEST="global_ids"
        )
        assert ids == [3]

    def test_atom_type_exclude(self, sim_synthetic):
        ids = sim_synthetic.get_topology(
            QUERY={"atom_type": (set(), {"OT"})}, REQUEST="global_ids"
        )
        assert 3 not in ids

    def test_residue_name_include(self, sim_synthetic):
        ids = sim_synthetic.get_topology(
            QUERY={"residue_name": ({"ALA"}, set())}, REQUEST="global_ids"
        )
        assert sorted(ids) == [0, 1]

    def test_segment_name_include(self, sim_synthetic):
        ids = sim_synthetic.get_topology(
            QUERY={"segment_name": ({"SOLV"}, set())}, REQUEST="global_ids"
        )
        assert ids == [3]

    def test_charge_range_include(self, sim_synthetic):
        ids = sim_synthetic.get_topology(
            QUERY={"charge": ((-0.3, -0.1), (None, None))}, REQUEST="global_ids"
        )
        assert sorted(ids) == [0, 1, 2]

    def test_mass_range_include(self, sim_synthetic):
        ids = sim_synthetic.get_topology(
            QUERY={"mass": ((12.0, 13.0), (None, None))}, REQUEST="global_ids"
        )
        assert sorted(ids) == [0, 1, 2]

    def test_bonded_total_degree_eq_1(self, sim_synthetic):
        ids = sim_synthetic.get_topology(
            QUERY={
                "bonded_with": ([{"total": True, "count": {"eq": 1}}], []),
                "bonded_with_mode": ("all", None),
            },
            REQUEST="global_ids",
        )
        assert sorted(ids) == [0, 3]

    def test_bonded_total_degree_eq_2(self, sim_synthetic):
        ids = sim_synthetic.get_topology(
            QUERY={
                "bonded_with": ([{"total": True, "count": {"eq": 2}}], []),
                "bonded_with_mode": ("all", None),
            },
            REQUEST="global_ids",
        )
        assert sorted(ids) == [1, 2]

    def test_bonded_neighbor_type(self, sim_synthetic):
        ids = sim_synthetic.get_topology(
            QUERY={
                "bonded_with": (
                    [{"neighbor": {"atom_type": ({"OT"}, set())}, "count": {"ge": 1}}],
                    [],
                ),
                "bonded_with_mode": ("all", None),
            },
            REQUEST="global_ids",
        )
        assert ids == [2]

    def test_bonded_exclude(self, sim_synthetic):
        ids = sim_synthetic.get_topology(
            QUERY={
                "bonded_with": (
                    [],
                    [{"neighbor": {"atom_type": ({"OT"}, set())}, "count": {"ge": 1}}],
                ),
                "bonded_with_mode": ("all", None),
            },
            REQUEST="global_ids",
        )
        assert 2 not in ids

    def test_bonded_mode_any(self, sim_synthetic):
        ids = sim_synthetic.get_topology(
            QUERY={
                "bonded_with": (
                    [
                        {"total": True, "count": {"eq": 1}},
                        {"neighbor": {"atom_type": ({"OT"}, set())}, "count": {"ge": 1}},
                    ],
                    [],
                ),
                "bonded_with_mode": ("any", None),
            },
            REQUEST="global_ids",
        )
        assert sorted(ids) == [0, 2, 3]

# ===========================================================================
# PSF — request: local_ids
# ===========================================================================

class TestPSFLocalIds:

    def test_local_ids_all_atoms(self, sim_synthetic):
        ids = sim_synthetic.get_topology(QUERY={}, REQUEST="local_ids")
        assert len(ids) == 4
        assert all(isinstance(i, int) for i in ids)

    def test_local_ids_filtered(self, sim_synthetic):
        ids = sim_synthetic.get_topology(
            QUERY={"atom_name": ({"CA"}, set())}, REQUEST="local_ids"
        )
        assert len(ids) == 2

# ===========================================================================
# PSF — request: residue_ids
# ===========================================================================

class TestPSFResidueIds:

    def test_residue_ids_ala_atoms(self, sim_synthetic):
        ids = sim_synthetic.get_topology(
            QUERY={"residue_name": ({"ALA"}, set())}, REQUEST="residue_ids"
        )
        assert all(i == 1 for i in ids)

# ===========================================================================
# PSF — request: atom_names
# ===========================================================================

class TestPSFAtomNames:

    def test_atom_names_all(self, sim_synthetic):
        names = sim_synthetic.get_topology(QUERY={}, REQUEST="atom_names")
        assert len(names) == 4
        assert all(isinstance(n, str) for n in names)

    def test_atom_names_filtered(self, sim_synthetic):
        names = sim_synthetic.get_topology(
            QUERY={"atom_name": ({"CA"}, set())}, REQUEST="atom_names"
        )
        assert all(n == "CA" for n in names)

# ===========================================================================
# PSF — request: atom_types
# ===========================================================================

class TestPSFAtomTypes:

    def test_atom_types_all(self, sim_synthetic):
        types = sim_synthetic.get_topology(QUERY={}, REQUEST="atom_types")
        assert len(types) == 4
        assert all(isinstance(t, str) for t in types)

    def test_atom_types_filtered(self, sim_synthetic):
        types = sim_synthetic.get_topology(
            QUERY={"atom_type": ({"OT"}, set())}, REQUEST="atom_types"
        )
        assert types == ["OT"]

# ===========================================================================
# PSF — request: residue_names
# ===========================================================================

class TestPSFResidueNames:

    def test_residue_names_all(self, sim_synthetic):
        names = sim_synthetic.get_topology(QUERY={}, REQUEST="residue_names")
        assert len(names) == 4

    def test_residue_names_filtered(self, sim_synthetic):
        names = sim_synthetic.get_topology(
            QUERY={"residue_name": ({"ALA"}, set())}, REQUEST="residue_names"
        )
        assert all(n == "ALA" for n in names)

# ===========================================================================
# PSF — request: segment_names
# ===========================================================================

class TestPSFSegmentNames:

    def test_segment_names_all(self, sim_synthetic):
        names = sim_synthetic.get_topology(QUERY={}, REQUEST="segment_names")
        assert len(names) == 4

    def test_segment_names_filtered(self, sim_synthetic):
        names = sim_synthetic.get_topology(
            QUERY={"segment_name": ({"SOLV"}, set())}, REQUEST="segment_names"
        )
        assert names == ["SOLV"]

# ===========================================================================
# PSF — request: charges
# ===========================================================================

class TestPSFCharges:

    def test_charges_all_atoms(self, sim_synthetic):
        charges = sim_synthetic.get_topology(QUERY={}, REQUEST="charges")
        assert len(charges) == 4
        assert all(isinstance(c, float) for c in charges)

    def test_charges_filtered(self, sim_synthetic):
        charges = sim_synthetic.get_topology(
            QUERY={"atom_type": ({"OT"}, set())}, REQUEST="charges"
        )
        assert len(charges) == 1
        assert round(charges[0], 3) == -0.834

# ===========================================================================
# PSF — request: masses
# ===========================================================================

class TestPSFMasses:

    def test_masses_all_atoms(self, sim_synthetic):
        masses = sim_synthetic.get_topology(QUERY={}, REQUEST="masses")
        assert len(masses) == 4
        assert all(isinstance(m, float) for m in masses)

    def test_masses_filtered(self, sim_synthetic):
        masses = sim_synthetic.get_topology(
            QUERY={"residue_name": ({"ALA", "GLY"}, set())}, REQUEST="masses"
        )
        assert all(round(m, 3) == 12.011 for m in masses)

# ===========================================================================
# PSF — property requests
# ===========================================================================

class TestPSFProperties:

    def test_property_system_charge(self, sim_synthetic):
        result = sim_synthetic.select(TOPO_Q={}, TOPO_R="property-system_charge")
        total = result["payload"]["topology"]
        assert isinstance(total, float)
        assert round(total, 3) == round(-0.270 + -0.270 + -0.180 + -0.834, 3)

    def test_bonds_with_raises_not_implemented(self, sim_synthetic):
        with pytest.raises(NotImplementedError):
            sim_synthetic.get_topology(QUERY={}, REQUEST="bonds_with")

# ===========================================================================
# PSF — additional coverage
# ===========================================================================

class TestPSFGlobalIdsExtra:

    def test_atom_name_exclude_wins_over_include(self, sim_synthetic):
        ids = sim_synthetic.get_topology(QUERY={"atom_name": ({"CA"}, {"CA"})}, REQUEST="global_ids")
        assert ids == []

    def test_residue_name_exclude(self, sim_synthetic):
        ids = sim_synthetic.get_topology(QUERY={"residue_name": (set(), {"TIP"})}, REQUEST="global_ids")
        assert sorted(ids) == [0, 1, 2]

    def test_segment_name_exclude(self, sim_synthetic):
        ids = sim_synthetic.get_topology(QUERY={"segment_name": (set(), {"SOLV"})}, REQUEST="global_ids")
        assert sorted(ids) == [0, 1, 2]

    def test_charge_range_exclude(self, sim_synthetic):
        # exclude the OW atom (charge ~ -0.834)
        ids = sim_synthetic.get_topology(QUERY={"charge": ((None, None), (-1.0, -0.5))}, REQUEST="global_ids")
        assert 3 not in ids

    def test_mass_range_exclude_oxygen(self, sim_synthetic):
        ids = sim_synthetic.get_topology(QUERY={"mass": ((None, None), (15.0, 16.5))}, REQUEST="global_ids")
        assert sorted(ids) == [0, 1, 2]

    def test_combined_name_and_type_filter(self, sim_synthetic):
        ids = sim_synthetic.get_topology(
            QUERY={"atom_name": ({"CA"}, set()), "atom_type": ({"CT2"}, set())},
            REQUEST="global_ids",
        )
        assert ids == [2]

    def test_no_matches_returns_empty_list(self, sim_synthetic):
        ids = sim_synthetic.get_topology(QUERY={"atom_name": ({"ZZ"}, set())}, REQUEST="global_ids")
        assert ids == []

    def test_result_is_sorted(self, sim_synthetic):
        ids = sim_synthetic.get_topology(QUERY={}, REQUEST="global_ids")
        assert ids == sorted(ids)


class TestPSFResidueIdsExtra:

    def test_residue_ids_are_ints(self, sim_synthetic):
        ids = sim_synthetic.get_topology(QUERY={}, REQUEST="residue_ids")
        assert all(isinstance(i, int) for i in ids)

    def test_residue_ids_all_atoms_count(self, sim_synthetic):
        ids = sim_synthetic.get_topology(QUERY={}, REQUEST="residue_ids")
        assert len(ids) == 4


class TestPSFChargesExtra:

    def test_charges_carbon_atoms(self, sim_synthetic):
        charges = sim_synthetic.get_topology(
            QUERY={"residue_name": ({"ALA"}, set())}, REQUEST="charges"
        )
        assert all(round(c, 3) == -0.270 for c in charges)


class TestPSFMassesExtra:

    def test_oxygen_mass(self, sim_synthetic):
        masses = sim_synthetic.get_topology(
            QUERY={"atom_type": ({"OT"}, set())}, REQUEST="masses"
        )
        assert len(masses) == 1
        assert round(masses[0], 3) == 15.999


class TestPSFPropertiesExtra:

    def test_property_system_charge_is_negative(self, sim_synthetic):
        result = sim_synthetic.select(TOPO_Q={}, TOPO_R="property-system_charge")
        assert result["payload"]["topology"] < 0