"""
test_queries_synthetic.py
=========================
Full pipeline integration tests using synthetic files only.

Atom layout:
    global_id 0  CA  ALA  PROT  x=1   y=2   z=3
    global_id 1  CB  ALA  PROT  x=4   y=5   z=6
    global_id 2  CA  GLY  PROT  x=7   y=8   z=9
    global_id 3  OW  TIP  SOLV  x=10  y=11  z=12

Bonds (PSF): 0-1, 1-2, 2-3

DCD (3 frames):
    frame 0: positions = base + 0.0
    frame 1: positions = base + 1.0
    frame 2: positions = base + 2.0
"""

from __future__ import annotations

import numpy as np
import pytest


class TestFullPositionsPipeline:

    def test_positions_via_type_query(self, sim_synthetic):
        out = sim_synthetic.positions(
            TYPE_Q={"atom_name": ({"CA"}, set())},
            TRAJ_Q={"frame_interval": (0, 2, 1)},
        )
        assert out["mode"] == "positions"
        assert out["selection"]["count"] == 2
        pos, meta = out["payload"]
        assert pos.shape == (3, 2, 3)

    def test_positions_via_topology_query(self, sim_synthetic):
        out = sim_synthetic.positions(
            TOPO_Q={"segment_name": ({"SOLV"}, set())},
            TRAJ_Q={"frame_interval": (0, 0, 1)},
        )
        assert out["selection"]["count"] == 1
        pos, _ = out["payload"]
        assert pos.shape == (1, 1, 3)

    def test_positions_intersection_of_type_and_topology(self, sim_synthetic):
        # TYPE_Q: CA atoms → [0, 2]
        # TOPO_Q: ALA residue → [0, 1]
        # intersection → [0]
        out = sim_synthetic.positions(
            TYPE_Q={"atom_name": ({"CA"}, set())},
            TOPO_Q={"residue_name": ({"ALA"}, set())},
            TRAJ_Q={"frame_interval": (0, 0, 1)},
        )
        assert out["selection"]["count"] == 1

    def test_positions_empty_query_selects_all_atoms(self, sim_synthetic):
        out = sim_synthetic.positions(
            TYPE_Q={},
            TRAJ_Q={"frame_interval": (0, 0, 1)},
        )
        assert out["selection"]["count"] == 4

    def test_positions_empty_intersection_raises(self, sim_synthetic):
        with pytest.raises(ValueError, match="empty intersection"):
            sim_synthetic.positions(
                TYPE_Q={"atom_name": ({"CA"}, set())},
                TOPO_Q={"segment_name": ({"SOLV"}, set())},
                TRAJ_Q={"frame_interval": (0, 0, 1)},
            )

    def test_positions_output_is_finite(self, sim_synthetic):
        out = sim_synthetic.positions(
            TYPE_Q={},
            TRAJ_Q={"frame_interval": (0, 2, 1)},
        )
        pos, _ = out["payload"]
        assert np.all(np.isfinite(pos))

    def test_positions_output_shape_n_frames_x_n_atoms_x_3(self, sim_synthetic):
        out = sim_synthetic.positions(
            TYPE_Q={"atom_name": ({"CA"}, set())},
            TRAJ_Q={"frame_interval": (0, 2, 1)},
        )
        pos, _ = out["payload"]
        assert pos.ndim == 3
        assert pos.shape[1] == out["selection"]["count"]
        assert pos.shape[2] == 3


class TestPositionsPlanFlag:

    def test_plan_flag_returns_none_payload(self, sim_synthetic):
        out = sim_synthetic.positions(
            TYPE_Q={"atom_name": ({"CA"}, set())},
            TRAJ_Q={"frame_interval": (0, 0, 1)},
            planFlag=True,
        )
        assert "plan"    in out
        assert out["payload"] is None

    def test_plan_flag_contains_expected_keys(self, sim_synthetic):
        out = sim_synthetic.positions(
            TYPE_Q={"atom_name": ({"CA"}, set())},
            TRAJ_Q={"frame_interval": (0, 0, 1)},
            planFlag=True,
        )
        assert "plan"      in out
        assert "mode"      in out
        assert "selection" in out


class TestPositionsSelectionMetadata:
    """
    Selection metadata (sources, count, merge_mode) is populated during real
    execution. planFlag=True returns count=None and sources=[].
    These tests use planFlag=False to verify metadata correctness.
    """

    def test_sources_typing_only(self, sim_synthetic):
        out = sim_synthetic.positions(
            TYPE_Q={"atom_name": ({"CA"}, set())},
            TRAJ_Q={"frame_interval": (0, 0, 1)},
        )
        assert out["selection"]["sources"] == ["typing"]

    def test_sources_topology_only(self, sim_synthetic):
        out = sim_synthetic.positions(
            TOPO_Q={"segment_name": ({"PROT"}, set())},
            TRAJ_Q={"frame_interval": (0, 0, 1)},
        )
        assert out["selection"]["sources"] == ["topology"]

    def test_sources_both_when_both_provided(self, sim_synthetic):
        out = sim_synthetic.positions(
            TYPE_Q={"atom_name": ({"CA"}, set())},
            TOPO_Q={"residue_name": ({"ALA"}, set())},
            TRAJ_Q={"frame_interval": (0, 0, 1)},
        )
        assert "typing"   in out["selection"]["sources"]
        assert "topology" in out["selection"]["sources"]

    def test_merge_mode_is_intersection(self, sim_synthetic):
        out = sim_synthetic.positions(
            TYPE_Q={"atom_name": ({"CA"}, set())},
            TRAJ_Q={"frame_interval": (0, 0, 1)},
        )
        assert out["selection"]["merge_mode"] == "intersection"

    def test_selection_count_matches_payload_atom_axis(self, sim_synthetic):
        out = sim_synthetic.positions(
            TYPE_Q={"atom_name": ({"CA"}, set())},
            TRAJ_Q={"frame_interval": (0, 0, 1)},
        )
        pos, _ = out["payload"]
        assert out["selection"]["count"] == pos.shape[1]

    def test_empty_query_count_is_all_atoms(self, sim_synthetic):
        out = sim_synthetic.positions(
            TYPE_Q={},
            TRAJ_Q={"frame_interval": (0, 0, 1)},
        )
        assert out["selection"]["count"] == 4


class TestSelectPropertyRequests:

    def test_select_single_type_property(self, sim_synthetic):
        out = sim_synthetic.select(TYPE_Q={}, TYPE_R="property-number_of_atoms")
        assert out["mode"] == "property"
        assert out["payload"]["typing"] == 4

    def test_select_single_topo_property(self, sim_synthetic):
        out = sim_synthetic.select(TOPO_Q={}, TOPO_R="property-system_charge")
        assert "topology" in out["payload"]
        assert isinstance(out["payload"]["topology"], float)

    def test_select_multiple_domains_simultaneously(self, sim_synthetic):
        out = sim_synthetic.select(
            TYPE_Q={},  TYPE_R="property-number_of_atoms",
            TOPO_Q={},  TOPO_R="property-system_charge",
        )
        assert "typing"   in out["payload"]
        assert "topology" in out["payload"]

    def test_select_plan_flag(self, sim_synthetic):
        out = sim_synthetic.select(
            TYPE_Q={}, TYPE_R="property-number_of_atoms", planFlag=True
        )
        assert out["payload"] is None
        assert "plan" in out


# ===========================================================================
# positions() — static fallback (no trajectory loaded)
# ===========================================================================

class TestPositionsStaticFallbackPDB:
    """
    When no trajectory file is loaded, positions() falls back to reading
    coordinates from the typing file. The return dict uses mode="positions"
    and payload is a raw ndarray of shape (1, n_atoms, 3).
    """

    def test_mode_is_positions(self, sim_pdb_only):
        out = sim_pdb_only.positions()
        assert out["mode"] == "positions"

    def test_payload_is_ndarray(self, sim_pdb_only):
        out = sim_pdb_only.positions()
        assert isinstance(out["payload"], np.ndarray)

    def test_payload_shape_all_atoms(self, sim_pdb_only):
        out = sim_pdb_only.positions()
        arr = out["payload"]
        assert arr.shape == (1, 4, 3)

    def test_payload_dtype_float32(self, sim_pdb_only):
        out = sim_pdb_only.positions()
        arr = out["payload"]
        assert arr.dtype == np.float32

    def test_payload_values_atom0(self, sim_pdb_only):
        out = sim_pdb_only.positions()
        arr = out["payload"]
        np.testing.assert_allclose(arr[0, 0], [1.0, 2.0, 3.0], atol=1e-4)

    def test_payload_values_atom3(self, sim_pdb_only):
        out = sim_pdb_only.positions()
        arr = out["payload"]
        np.testing.assert_allclose(arr[0, 3], [10.0, 11.0, 12.0], atol=1e-4)

    def test_selection_count_all_atoms(self, sim_pdb_only):
        out = sim_pdb_only.positions()
        assert out["selection"]["count"] == 4

    def test_type_q_filter_reduces_atom_count(self, sim_pdb_only):
        out = sim_pdb_only.positions(TYPE_Q={"atom_name": ({"CA"}, set())})
        arr = out["payload"]
        assert arr.shape == (1, 2, 3)
        assert out["selection"]["count"] == 2

    def test_type_q_filter_values_correct(self, sim_pdb_only):
        out = sim_pdb_only.positions(TYPE_Q={"atom_name": ({"CA"}, set())})
        arr = out["payload"]
        np.testing.assert_allclose(arr[0, 0], [1.0, 2.0, 3.0], atol=1e-4)
        np.testing.assert_allclose(arr[0, 1], [7.0, 8.0, 9.0], atol=1e-4)

    def test_static_consistent_with_x_y_z_requests(self, sim_pdb_only):
        out = sim_pdb_only.positions()
        arr = out["payload"]
        xs = sim_pdb_only.get_types(QUERY={}, REQUEST="x")
        ys = sim_pdb_only.get_types(QUERY={}, REQUEST="y")
        zs = sim_pdb_only.get_types(QUERY={}, REQUEST="z")
        np.testing.assert_allclose(arr[0, :, 0], xs, atol=1e-4)
        np.testing.assert_allclose(arr[0, :, 1], ys, atol=1e-4)
        np.testing.assert_allclose(arr[0, :, 2], zs, atol=1e-4)

    def test_plan_flag_returns_no_payload(self, sim_pdb_only):
        out = sim_pdb_only.positions(planFlag=True)
        assert out["payload"] is None
        assert out["plan"] is not None

    def test_normalise_bare_string_query(self, sim_pdb_only):
        out = sim_pdb_only.positions(TYPE_Q={"atom_name": "CA"})
        arr = out["payload"]
        assert arr.shape == (1, 2, 3)

    def test_no_traj_file_loaded(self, sim_pdb_only):
        assert sim_pdb_only.traj_file is None


class TestPositionsStaticFallbackXYZ:

    def test_mode_is_positions(self, sim_xyz_only):
        out = sim_xyz_only.positions()
        assert out["mode"] == "positions"

    def test_payload_is_ndarray(self, sim_xyz_only):
        out = sim_xyz_only.positions()
        assert isinstance(out["payload"], np.ndarray)

    def test_payload_shape_all_atoms(self, sim_xyz_only):
        out = sim_xyz_only.positions()
        arr = out["payload"]
        assert arr.shape == (1, 4, 3)

    def test_type_q_filter_reduces_atom_count(self, sim_xyz_only):
        out = sim_xyz_only.positions(TYPE_Q={"atom_name": ({"C"}, set())})
        arr = out["payload"]
        assert arr.shape == (1, 2, 3)

    def test_normalise_bare_string_query(self, sim_xyz_only):
        out = sim_xyz_only.positions(TYPE_Q={"atom_name": "C"})
        arr = out["payload"]
        assert arr.shape == (1, 2, 3)

    def test_static_consistent_with_x_y_z_requests(self, sim_xyz_only):
        out = sim_xyz_only.positions()
        arr = out["payload"]
        xs = sim_xyz_only.get_types(QUERY={}, REQUEST="x")
        ys = sim_xyz_only.get_types(QUERY={}, REQUEST="y")
        zs = sim_xyz_only.get_types(QUERY={}, REQUEST="z")
        np.testing.assert_allclose(arr[0, :, 0], xs, atol=1e-4)
        np.testing.assert_allclose(arr[0, :, 1], ys, atol=1e-4)
        np.testing.assert_allclose(arr[0, :, 2], zs, atol=1e-4)


class TestPositionsStaticFallbackErrors:

    def test_raises_when_no_file_loaded(self):
        from trajectory_kit import sim
        s = sim()
        with pytest.raises(ValueError):
            s.positions()
