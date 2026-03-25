"""
test_typing.py
==============
Complete test suite for the typing domain (PDB and XYZ file types).
Covers every available request for each file type.

Atom layout in synthetic PDB:
    global_id 0  CA  ALA  PROT  x=1  y=2   z=3   occ=1.00  temp=0.10
    global_id 1  CB  ALA  PROT  x=4  y=5   z=6   occ=1.00  temp=0.20
    global_id 2  CA  GLY  PROT  x=7  y=8   z=9   occ=1.00  temp=0.30
    global_id 3  OW  TIP  SOLV  x=10 y=11  z=12  occ=1.00  temp=0.40

XYZ atoms:
    global_id 0  C   x=1  y=2   z=3
    global_id 1  C   x=4  y=5   z=6
    global_id 2  N   x=7  y=8   z=9
    global_id 3  O   x=10 y=11  z=12
"""

from __future__ import annotations

import pytest

# ===========================================================================
# PDB — request: global_ids
# ===========================================================================

class TestPDBGlobalIds:

    def test_empty_query_returns_all_atoms(self, sim_synthetic):
        ids = sim_synthetic.get_types(QUERY={}, REQUEST="global_ids")
        assert sorted(ids) == [0, 1, 2, 3]

    def test_atom_name_include_single(self, sim_synthetic):
        ids = sim_synthetic.get_types(
            QUERY={"atom_name": ({"CA"}, set())}, REQUEST="global_ids"
        )
        assert sorted(ids) == [0, 2]

    def test_atom_name_include_multiple(self, sim_synthetic):
        ids = sim_synthetic.get_types(
            QUERY={"atom_name": ({"CA", "CB"}, set())}, REQUEST="global_ids"
        )
        assert sorted(ids) == [0, 1, 2]

    def test_atom_name_exclude(self, sim_synthetic):
        ids = sim_synthetic.get_types(
            QUERY={"atom_name": (set(), {"OW"})}, REQUEST="global_ids"
        )
        assert sorted(ids) == [0, 1, 2]

    def test_atom_name_exclude_wins_over_include(self, sim_synthetic):
        ids = sim_synthetic.get_types(
            QUERY={"atom_name": ({"CA"}, {"CA"})}, REQUEST="global_ids"
        )
        assert ids == []

    def test_residue_name_include(self, sim_synthetic):
        ids = sim_synthetic.get_types(
            QUERY={"residue_name": ({"ALA"}, set())}, REQUEST="global_ids"
        )
        assert sorted(ids) == [0, 1]

    def test_residue_name_exclude(self, sim_synthetic):
        ids = sim_synthetic.get_types(
            QUERY={"residue_name": (set(), {"ALA"})}, REQUEST="global_ids"
        )
        assert sorted(ids) == [2, 3]

    def test_segment_name_include(self, sim_synthetic):
        ids = sim_synthetic.get_types(
            QUERY={"segment_name": ({"SOLV"}, set())}, REQUEST="global_ids"
        )
        assert ids == [3]

    def test_segment_name_exclude(self, sim_synthetic):
        ids = sim_synthetic.get_types(
            QUERY={"segment_name": (set(), {"SOLV"})}, REQUEST="global_ids"
        )
        assert sorted(ids) == [0, 1, 2]

    def test_residue_ids_range(self, sim_synthetic):
        ids = sim_synthetic.get_types(
            QUERY={"residue_ids": ((1, 1), (None, None))}, REQUEST="global_ids"
        )
        assert sorted(ids) == [0, 1]

    def test_x_range_unbounded_upper(self, sim_synthetic):
        ids = sim_synthetic.get_types(
            QUERY={"x": ((None, 5.0), (None, None))}, REQUEST="global_ids"
        )
        assert sorted(ids) == [0, 1]

    def test_x_exact_range(self, sim_synthetic):
        ids = sim_synthetic.get_types(
            QUERY={"x": ((7.0, 7.0), (None, None))}, REQUEST="global_ids"
        )
        assert ids == [2]

    def test_y_range_filter(self, sim_synthetic):
        ids = sim_synthetic.get_types(
            QUERY={"y": ((None, 6.0), (None, None))}, REQUEST="global_ids"
        )
        assert sorted(ids) == [0, 1]

    def test_z_exclude_range(self, sim_synthetic):
        ids = sim_synthetic.get_types(
            QUERY={"z": ((None, None), (3.0, 9.0))}, REQUEST="global_ids"
        )
        assert sorted(ids) == [3]

    def test_occupancy_filter(self, sim_synthetic):
        ids = sim_synthetic.get_types(
            QUERY={"occupancy": ((1.0, 1.0), (None, None))}, REQUEST="global_ids"
        )
        assert sorted(ids) == [0, 1, 2, 3]

    def test_temperature_coeff_filter(self, sim_synthetic):
        ids = sim_synthetic.get_types(
            QUERY={"temperature_coeff": ((None, 0.25), (None, None))}, REQUEST="global_ids"
        )
        assert sorted(ids) == [0, 1]

    def test_combined_filters(self, sim_synthetic):
        ids = sim_synthetic.get_types(
            QUERY={
                "atom_name": ({"CA"}, set()),
                "segment_name": ({"PROT"}, set()),
                "x": ((5.0, None), (None, None)),
            },
            REQUEST="global_ids",
        )
        assert ids == [2]

    def test_no_matches_returns_empty_list(self, sim_synthetic):
        ids = sim_synthetic.get_types(
            QUERY={"atom_name": ({"ZZ"}, set())}, REQUEST="global_ids"
        )
        assert ids == []

    def test_result_is_sorted(self, sim_synthetic):
        ids = sim_synthetic.get_types(QUERY={}, REQUEST="global_ids")
        assert ids == sorted(ids)

# ===========================================================================
# PDB — request: local_ids
# ===========================================================================

class TestPDBLocalIds:

    def test_local_ids_all_atoms(self, sim_synthetic):
        ids = sim_synthetic.get_types(QUERY={}, REQUEST="local_ids")
        assert len(ids) == 4

    def test_local_ids_filtered(self, sim_synthetic):
        ids = sim_synthetic.get_types(
            QUERY={"atom_name": ({"OW"}, set())}, REQUEST="local_ids"
        )
        assert len(ids) == 1

    def test_local_ids_are_ints(self, sim_synthetic):
        ids = sim_synthetic.get_types(QUERY={}, REQUEST="local_ids")
        assert all(isinstance(i, int) for i in ids)

# ===========================================================================
# PDB — request: residue_ids
# ===========================================================================

class TestPDBResidueIds:

    def test_residue_ids_all_atoms(self, sim_synthetic):
        ids = sim_synthetic.get_types(QUERY={}, REQUEST="residue_ids")
        assert len(ids) == 4

    def test_residue_ids_filtered(self, sim_synthetic):
        ids = sim_synthetic.get_types(
            QUERY={"residue_name": ({"ALA"}, set())}, REQUEST="residue_ids"
        )
        assert all(i == 1 for i in ids)

# ===========================================================================
# PDB — request: atom_names
# ===========================================================================

class TestPDBAtomNames:

    def test_atom_names_all_atoms(self, sim_synthetic):
        names = sim_synthetic.get_types(QUERY={}, REQUEST="atom_names")
        assert len(names) == 4
        assert all(isinstance(n, str) for n in names)

    def test_atom_names_filtered(self, sim_synthetic):
        names = sim_synthetic.get_types(
            QUERY={"atom_name": ({"CA"}, set())}, REQUEST="atom_names"
        )
        assert all(n == "CA" for n in names)

# ===========================================================================
# PDB — request: residue_names
# ===========================================================================

class TestPDBResidueNames:

    def test_residue_names_all_atoms(self, sim_synthetic):
        names = sim_synthetic.get_types(QUERY={}, REQUEST="residue_names")
        assert len(names) == 4

    def test_residue_names_filtered(self, sim_synthetic):
        names = sim_synthetic.get_types(
            QUERY={"residue_name": ({"ALA"}, set())}, REQUEST="residue_names"
        )
        assert all(n == "ALA" for n in names)

# ===========================================================================
# PDB — request: segment_names
# ===========================================================================

class TestPDBSegmentNames:

    def test_segment_names_all_atoms(self, sim_synthetic):
        names = sim_synthetic.get_types(QUERY={}, REQUEST="segment_names")
        assert len(names) == 4

    def test_segment_names_filtered(self, sim_synthetic):
        names = sim_synthetic.get_types(
            QUERY={"segment_name": ({"SOLV"}, set())}, REQUEST="segment_names"
        )
        assert names == ["SOLV"]

# ===========================================================================
# PDB — request: x, y, z
# ===========================================================================

class TestPDBCoordinateRequests:

    def test_x_returns_floats(self, sim_synthetic):
        xs = sim_synthetic.get_types(QUERY={}, REQUEST="x")
        assert len(xs) == 4
        assert all(isinstance(v, float) for v in xs)

    def test_x_values_correct(self, sim_synthetic):
        xs = sim_synthetic.get_types(QUERY={}, REQUEST="x")
        assert xs == [1.0, 4.0, 7.0, 10.0]

    def test_y_values_correct(self, sim_synthetic):
        ys = sim_synthetic.get_types(QUERY={}, REQUEST="y")
        assert ys == [2.0, 5.0, 8.0, 11.0]

    def test_z_values_correct(self, sim_synthetic):
        zs = sim_synthetic.get_types(QUERY={}, REQUEST="z")
        assert zs == [3.0, 6.0, 9.0, 12.0]

    def test_x_filtered(self, sim_synthetic):
        xs = sim_synthetic.get_types(
            QUERY={"atom_name": ({"CA"}, set())}, REQUEST="x"
        )
        assert xs == [1.0, 7.0]

# ===========================================================================
# PDB — request: occupancy, temperature_coeff
# ===========================================================================

class TestPDBScalarFields:

    def test_occupancy_all_atoms(self, sim_synthetic):
        vals = sim_synthetic.get_types(QUERY={}, REQUEST="occupancy")
        assert len(vals) == 4
        assert all(v == 1.0 for v in vals)

    def test_temperature_coeff_all_atoms(self, sim_synthetic):
        vals = sim_synthetic.get_types(QUERY={}, REQUEST="temperature_coeff")
        assert len(vals) == 4
        assert round(vals[0], 2) == 0.10
        assert round(vals[3], 2) == 0.40

# ===========================================================================
# PDB — property requests
# ===========================================================================

class TestPDBProperties:

    def test_property_number_of_atoms(self, sim_synthetic):
        result = sim_synthetic.select(TYPE_Q={}, TYPE_R="property-number_of_atoms")
        assert result["payload"]["typing"] == 4

    def test_property_number_of_residues(self, sim_synthetic):
        result = sim_synthetic.select(TYPE_Q={}, TYPE_R="property-number_of_residues")
        assert result["payload"]["typing"] == 3

    def test_property_number_of_segments(self, sim_synthetic):
        result = sim_synthetic.select(TYPE_Q={}, TYPE_R="property-number_of_segments")
        assert result["payload"]["typing"] == 2

    def test_property_box_size_returns_bounds(self, sim_synthetic):
        result = sim_synthetic.select(TYPE_Q={}, TYPE_R="property-box_size")
        bounds = result["payload"]["typing"]
        # x: min=1, max=10  y: min=2, max=11  z: min=3, max=12
        assert bounds == (1.0, 10.0, 2.0, 11.0, 3.0, 12.0)

    def test_property_box_size_is_tuple_of_six(self, sim_synthetic):
        result = sim_synthetic.select(TYPE_Q={}, TYPE_R="property-box_size")
        bounds = result["payload"]["typing"]
        assert isinstance(bounds, tuple)
        assert len(bounds) == 6

# ===========================================================================
# XYZ — request: global_ids
# ===========================================================================

class TestXYZGlobalIds:

    def test_empty_query_returns_all_atoms(self, sim_synthetic_xyz):
        ids = sim_synthetic_xyz.get_types(QUERY={}, REQUEST="global_ids")
        assert sorted(ids) == [0, 1, 2, 3]

    def test_atom_name_include(self, sim_synthetic_xyz):
        ids = sim_synthetic_xyz.get_types(
            QUERY={"atom_name": ({"C"}, set())}, REQUEST="global_ids"
        )
        assert sorted(ids) == [0, 1]

    def test_atom_name_exclude(self, sim_synthetic_xyz):
        ids = sim_synthetic_xyz.get_types(
            QUERY={"atom_name": (set(), {"O"})}, REQUEST="global_ids"
        )
        assert sorted(ids) == [0, 1, 2]

    def test_x_range_filter(self, sim_synthetic_xyz):
        ids = sim_synthetic_xyz.get_types(
            QUERY={"x": ((None, 5.0), (None, None))}, REQUEST="global_ids"
        )
        assert sorted(ids) == [0, 1]

    def test_y_range_filter(self, sim_synthetic_xyz):
        ids = sim_synthetic_xyz.get_types(
            QUERY={"y": ((8.0, None), (None, None))}, REQUEST="global_ids"
        )
        assert sorted(ids) == [2, 3]

    def test_z_range_filter(self, sim_synthetic_xyz):
        ids = sim_synthetic_xyz.get_types(
            QUERY={"z": ((None, 7.0), (None, None))}, REQUEST="global_ids"
        )
        assert sorted(ids) == [0, 1]

    def test_no_matches_returns_empty_list(self, sim_synthetic_xyz):
        ids = sim_synthetic_xyz.get_types(
            QUERY={"atom_name": ({"Fe"}, set())}, REQUEST="global_ids"
        )
        assert ids == []

# ===========================================================================
# XYZ — request: local_ids
# ===========================================================================

class TestXYZLocalIds:

    def test_local_ids_mirror_global_ids(self, sim_synthetic_xyz):
        global_ids = sim_synthetic_xyz.get_types(QUERY={}, REQUEST="global_ids")
        local_ids  = sim_synthetic_xyz.get_types(QUERY={}, REQUEST="local_ids")
        assert global_ids == local_ids

# ===========================================================================
# XYZ — request: atom_names
# ===========================================================================

class TestXYZAtomNames:

    def test_atom_names_all(self, sim_synthetic_xyz):
        names = sim_synthetic_xyz.get_types(QUERY={}, REQUEST="atom_names")
        assert names == ["C", "C", "N", "O"]

    def test_atom_names_filtered(self, sim_synthetic_xyz):
        names = sim_synthetic_xyz.get_types(
            QUERY={"atom_name": ({"C"}, set())}, REQUEST="atom_names"
        )
        assert all(n == "C" for n in names)

# ===========================================================================
# XYZ — request: x, y, z
# ===========================================================================

class TestXYZCoordinateRequests:

    def test_x_values_correct(self, sim_synthetic_xyz):
        xs = sim_synthetic_xyz.get_types(QUERY={}, REQUEST="x")
        assert xs == [1.0, 4.0, 7.0, 10.0]

    def test_y_values_correct(self, sim_synthetic_xyz):
        ys = sim_synthetic_xyz.get_types(QUERY={}, REQUEST="y")
        assert ys == [2.0, 5.0, 8.0, 11.0]

    def test_z_values_correct(self, sim_synthetic_xyz):
        zs = sim_synthetic_xyz.get_types(QUERY={}, REQUEST="z")
        assert zs == [3.0, 6.0, 9.0, 12.0]

    def test_x_filtered(self, sim_synthetic_xyz):
        xs = sim_synthetic_xyz.get_types(
            QUERY={"atom_name": ({"C"}, set())}, REQUEST="x"
        )
        assert xs == [1.0, 4.0]

# ===========================================================================
# XYZ — property requests
# ===========================================================================

class TestXYZProperties:

    def test_property_number_of_atoms(self, sim_synthetic_xyz):
        result = sim_synthetic_xyz.select(TYPE_Q={}, TYPE_R="property-number_of_atoms")
        assert result["payload"]["typing"] == 4

    def test_property_box_size_returns_bounds(self, sim_synthetic_xyz):
        result = sim_synthetic_xyz.select(TYPE_Q={}, TYPE_R="property-box_size")
        bounds = result["payload"]["typing"]
        # x: min=1, max=10  y: min=2, max=11  z: min=3, max=12
        assert bounds == (1.0, 10.0, 2.0, 11.0, 3.0, 12.0)

    def test_property_box_size_is_tuple_of_six(self, sim_synthetic_xyz):
        result = sim_synthetic_xyz.select(TYPE_Q={}, TYPE_R="property-box_size")
        bounds = result["payload"]["typing"]
        assert isinstance(bounds, tuple)
        assert len(bounds) == 6

# ===========================================================================
# PDB — additional coverage
# ===========================================================================

class TestPDBGlobalIdsExtra:

    def test_combined_filters(self, sim_synthetic):
        ids = sim_synthetic.get_types(
            QUERY={
                "atom_name":    ({"CA"}, set()),
                "segment_name": ({"PROT"}, set()),
                "x":            ((5.0, None), (None, None)),
            },
            REQUEST="global_ids",
        )
        assert ids == [2]

    def test_no_matches_returns_empty_list(self, sim_synthetic):
        ids = sim_synthetic.get_types(QUERY={"atom_name": ({"ZZ"}, set())}, REQUEST="global_ids")
        assert ids == []

    def test_result_is_sorted(self, sim_synthetic):
        ids = sim_synthetic.get_types(QUERY={}, REQUEST="global_ids")
        assert ids == sorted(ids)

    def test_exclude_wins_over_include(self, sim_synthetic):
        ids = sim_synthetic.get_types(QUERY={"atom_name": ({"CA"}, {"CA"})}, REQUEST="global_ids")
        assert ids == []


class TestPDBResidueIdsExtra:

    def test_residue_ids_are_ints(self, sim_synthetic):
        ids = sim_synthetic.get_types(QUERY={}, REQUEST="residue_ids")
        assert all(isinstance(i, int) for i in ids)


class TestPDBScalarFields:

    def test_occupancy_all_atoms(self, sim_synthetic):
        vals = sim_synthetic.get_types(QUERY={}, REQUEST="occupancy")
        assert len(vals) == 4
        assert all(v == 1.0 for v in vals)

    def test_temperature_coeff_all_atoms(self, sim_synthetic):
        vals = sim_synthetic.get_types(QUERY={}, REQUEST="temperature_coeff")
        assert len(vals) == 4
        assert round(vals[0], 2) == 0.10
        assert round(vals[3], 2) == 0.40

    def test_temperature_coeff_monotone_increasing(self, sim_synthetic):
        vals = sim_synthetic.get_types(QUERY={}, REQUEST="temperature_coeff")
        assert vals == sorted(vals)


class TestPDBPropertiesExtra:

    def test_property_box_size_xmin_lt_xmax(self, sim_synthetic):
        result = sim_synthetic.select(TYPE_Q={}, TYPE_R="property-box_size")
        xmin, xmax, ymin, ymax, zmin, zmax = result["payload"]["typing"]
        assert xmin < xmax
        assert ymin < ymax
        assert zmin < zmax


# ===========================================================================
# XYZ — additional coverage
# ===========================================================================

class TestXYZGlobalIdsExtra:

    def test_exclude_wins_over_include(self, sim_synthetic_xyz):
        ids = sim_synthetic_xyz.get_types(QUERY={"atom_name": ({"C"}, {"C"})}, REQUEST="global_ids")
        assert ids == []

    def test_no_matches_returns_empty_list(self, sim_synthetic_xyz):
        ids = sim_synthetic_xyz.get_types(QUERY={"atom_name": ({"Fe"}, set())}, REQUEST="global_ids")
        assert ids == []


class TestXYZCoordinateRequestsExtra:

    def test_coordinates_are_floats(self, sim_synthetic_xyz):
        for req in ("x", "y", "z"):
            vals = sim_synthetic_xyz.get_types(QUERY={}, REQUEST=req)
            assert all(isinstance(v, float) for v in vals), f"{req} values not floats"


class TestXYZPropertiesExtra:

    def test_property_box_size_min_lt_max(self, sim_synthetic_xyz):
        result = sim_synthetic_xyz.select(TYPE_Q={}, TYPE_R="property-box_size")
        xmin, xmax, ymin, ymax, zmin, zmax = result["payload"]["typing"]
        assert xmin < xmax
        assert ymin < ymax
        assert zmin < zmax


# ===========================================================================
# PDB — request: positions
# ===========================================================================

class TestPDBPositions:

    def test_positions_in_requests(self, sim_synthetic):
        assert "positions" in sim_synthetic.type_file_reqs

    def test_positions_returns_ndarray(self, sim_synthetic):
        import numpy as np
        result = sim_synthetic.get_types(QUERY={}, REQUEST="positions")
        assert isinstance(result, np.ndarray)

    def test_positions_shape_1_n_3_all_atoms(self, sim_synthetic):
        result = sim_synthetic.get_types(QUERY={}, REQUEST="positions")
        assert result.shape == (1, 4, 3)

    def test_positions_dtype_float32(self, sim_synthetic):
        import numpy as np
        result = sim_synthetic.get_types(QUERY={}, REQUEST="positions")
        assert result.dtype == np.float32

    def test_positions_values_atom0(self, sim_synthetic):
        import numpy as np
        result = sim_synthetic.get_types(QUERY={}, REQUEST="positions")
        np.testing.assert_allclose(result[0, 0], [1.0, 2.0, 3.0], atol=1e-4)

    def test_positions_values_atom3(self, sim_synthetic):
        import numpy as np
        result = sim_synthetic.get_types(QUERY={}, REQUEST="positions")
        np.testing.assert_allclose(result[0, 3], [10.0, 11.0, 12.0], atol=1e-4)

    def test_positions_consistent_with_x_y_z_requests(self, sim_synthetic):
        import numpy as np
        pos = sim_synthetic.get_types(QUERY={}, REQUEST="positions")
        xs  = sim_synthetic.get_types(QUERY={}, REQUEST="x")
        ys  = sim_synthetic.get_types(QUERY={}, REQUEST="y")
        zs  = sim_synthetic.get_types(QUERY={}, REQUEST="z")
        np.testing.assert_allclose(pos[0, :, 0], xs, atol=1e-4)
        np.testing.assert_allclose(pos[0, :, 1], ys, atol=1e-4)
        np.testing.assert_allclose(pos[0, :, 2], zs, atol=1e-4)

    def test_positions_filtered_by_atom_name(self, sim_synthetic):
        import numpy as np
        result = sim_synthetic.get_types(
            QUERY={"atom_name": ({"CA"}, set())}, REQUEST="positions"
        )
        assert result.shape == (1, 2, 3)
        np.testing.assert_allclose(result[0, 0], [1.0, 2.0, 3.0], atol=1e-4)
        np.testing.assert_allclose(result[0, 1], [7.0, 8.0, 9.0], atol=1e-4)

    def test_positions_filtered_consistent_with_x_y_z(self, sim_synthetic):
        import numpy as np
        q = {"atom_name": ({"CA"}, set())}
        pos = sim_synthetic.get_types(QUERY=q, REQUEST="positions")
        xs  = sim_synthetic.get_types(QUERY=q, REQUEST="x")
        ys  = sim_synthetic.get_types(QUERY=q, REQUEST="y")
        zs  = sim_synthetic.get_types(QUERY=q, REQUEST="z")
        np.testing.assert_allclose(pos[0, :, 0], xs, atol=1e-4)
        np.testing.assert_allclose(pos[0, :, 1], ys, atol=1e-4)
        np.testing.assert_allclose(pos[0, :, 2], zs, atol=1e-4)

    def test_positions_filtered_segment_name(self, sim_synthetic):
        result = sim_synthetic.get_types(
            QUERY={"segment_name": ({"SOLV"}, set())}, REQUEST="positions"
        )
        assert result.shape == (1, 1, 3)

    def test_positions_no_matches_empty_array(self, sim_synthetic):
        result = sim_synthetic.get_types(
            QUERY={"atom_name": ({"ZZ"}, set())}, REQUEST="positions"
        )
        assert result.shape == (1, 0, 3)

    def test_positions_normalise_bare_string_query(self, sim_synthetic):
        # bare string should work via _normalise_query_pair
        result = sim_synthetic.get_types(
            QUERY={"atom_name": "CA"}, REQUEST="positions"
        )
        assert result.shape == (1, 2, 3)


# ===========================================================================
# XYZ — request: positions
# ===========================================================================

class TestXYZPositions:

    def test_positions_in_requests(self, sim_synthetic_xyz):
        assert "positions" in sim_synthetic_xyz.type_file_reqs

    def test_positions_returns_ndarray(self, sim_synthetic_xyz):
        import numpy as np
        result = sim_synthetic_xyz.get_types(QUERY={}, REQUEST="positions")
        assert isinstance(result, np.ndarray)

    def test_positions_shape_1_n_3_all_atoms(self, sim_synthetic_xyz):
        result = sim_synthetic_xyz.get_types(QUERY={}, REQUEST="positions")
        assert result.shape == (1, 4, 3)

    def test_positions_dtype_float32(self, sim_synthetic_xyz):
        import numpy as np
        result = sim_synthetic_xyz.get_types(QUERY={}, REQUEST="positions")
        assert result.dtype == np.float32

    def test_positions_values_atom0(self, sim_synthetic_xyz):
        import numpy as np
        result = sim_synthetic_xyz.get_types(QUERY={}, REQUEST="positions")
        np.testing.assert_allclose(result[0, 0], [1.0, 2.0, 3.0], atol=1e-4)

    def test_positions_values_atom3(self, sim_synthetic_xyz):
        import numpy as np
        result = sim_synthetic_xyz.get_types(QUERY={}, REQUEST="positions")
        np.testing.assert_allclose(result[0, 3], [10.0, 11.0, 12.0], atol=1e-4)

    def test_positions_consistent_with_x_y_z_requests(self, sim_synthetic_xyz):
        import numpy as np
        pos = sim_synthetic_xyz.get_types(QUERY={}, REQUEST="positions")
        xs  = sim_synthetic_xyz.get_types(QUERY={}, REQUEST="x")
        ys  = sim_synthetic_xyz.get_types(QUERY={}, REQUEST="y")
        zs  = sim_synthetic_xyz.get_types(QUERY={}, REQUEST="z")
        np.testing.assert_allclose(pos[0, :, 0], xs, atol=1e-4)
        np.testing.assert_allclose(pos[0, :, 1], ys, atol=1e-4)
        np.testing.assert_allclose(pos[0, :, 2], zs, atol=1e-4)

    def test_positions_filtered_by_atom_name(self, sim_synthetic_xyz):
        import numpy as np
        result = sim_synthetic_xyz.get_types(
            QUERY={"atom_name": ({"C"}, set())}, REQUEST="positions"
        )
        assert result.shape == (1, 2, 3)
        np.testing.assert_allclose(result[0, 0], [1.0, 2.0, 3.0], atol=1e-4)
        np.testing.assert_allclose(result[0, 1], [4.0, 5.0, 6.0], atol=1e-4)

    def test_positions_normalise_bare_string_query(self, sim_synthetic_xyz):
        result = sim_synthetic_xyz.get_types(
            QUERY={"atom_name": "C"}, REQUEST="positions"
        )
        assert result.shape == (1, 2, 3)