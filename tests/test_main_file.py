"""
test_main_file.py
=================
Tests for sim class internals: file loading, validation, query normalisation,
request classification, and parse mode resolution.
"""

from __future__ import annotations

import pytest
from trajectory_kit import sim as Sim


# ===========================================================================
# File loading
# ===========================================================================

class TestFileLoading:

    def test_pdb_loads_as_typing(self, synthetic_pdb, synthetic_psf, synthetic_dcd):
        s = Sim(typing=synthetic_pdb, topology=synthetic_psf, trajectory=synthetic_dcd)
        assert s.type_type == ".pdb"
        assert s.top_type  == ".psf"
        assert s.traj_type == ".dcd"

    def test_xyz_loads_as_typing(self, synthetic_xyz, synthetic_psf, synthetic_dcd):
        s = Sim(typing=synthetic_xyz, topology=synthetic_psf, trajectory=synthetic_dcd)
        assert s.type_type == ".xyz"

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            Sim(typing=tmp_path / "does_not_exist.pdb")

    def test_directory_as_file_raises(self, tmp_path):
        with pytest.raises(IsADirectoryError):
            Sim(typing=tmp_path)

    def test_unsupported_typing_format_raises(self, tmp_path):
        bad = tmp_path / "file.xtc"
        bad.write_bytes(b"")
        with pytest.raises(ValueError, match="Unsupported typing file format"):
            Sim(typing=bad)

    def test_keys_populated_after_load(self, sim_synthetic):
        assert "atom_name"      in sim_synthetic.type_file_keys
        assert "global_ids"     in sim_synthetic.type_file_reqs
        assert "atom_name"      in sim_synthetic.topo_file_keys
        assert "global_ids"     in sim_synthetic.topo_file_reqs
        assert "frame_interval" in sim_synthetic.traj_file_keys
        assert "positions"      in sim_synthetic.traj_file_reqs

    def test_load_typing_only(self, synthetic_pdb):
        s = Sim(typing=synthetic_pdb)
        assert s.type_file is not None
        assert s.top_file  is None
        assert s.traj_file is None

    def test_load_topology_only(self, synthetic_psf):
        s = Sim(topology=synthetic_psf)
        assert s.top_file  is not None
        assert s.type_file is None
        assert s.traj_file is None

    def test_load_trajectory_only(self, synthetic_dcd):
        s = Sim(trajectory=synthetic_dcd)
        assert s.traj_file is not None
        assert s.type_file is None
        assert s.top_file  is None

    def test_incremental_load_matches_constructor(self, synthetic_pdb, synthetic_psf, synthetic_dcd):
        s_all = Sim(typing=synthetic_pdb, topology=synthetic_psf, trajectory=synthetic_dcd)
        s_inc = Sim()
        s_inc.load_typing(synthetic_pdb)
        s_inc.load_topology(synthetic_psf)
        s_inc.load_trajectory(synthetic_dcd)
        assert s_all.type_type      == s_inc.type_type
        assert s_all.top_type       == s_inc.top_type
        assert s_all.traj_type      == s_inc.traj_type
        assert s_all.type_file_keys == s_inc.type_file_keys
        assert s_all.topo_file_keys  == s_inc.topo_file_keys


# ===========================================================================
# _get_filetype
# ===========================================================================

class TestGetFiletype:

    def test_pdb_extension(self, sim_synthetic):
        assert sim_synthetic._get_filetype("some/path/file.pdb") == ".pdb"

    def test_psf_extension(self, sim_synthetic):
        assert sim_synthetic._get_filetype("some/path/file.psf") == ".psf"

    def test_dcd_extension(self, sim_synthetic):
        assert sim_synthetic._get_filetype("some/path/file.dcd") == ".dcd"

    def test_xyz_extension(self, sim_synthetic):
        assert sim_synthetic._get_filetype("some/path/file.xyz") == ".xyz"

    def test_uppercase_normalised(self, sim_synthetic):
        assert sim_synthetic._get_filetype("FILE.PDB") == ".pdb"

    def test_windows_path(self, sim_synthetic):
        assert sim_synthetic._get_filetype(r"E:\PhD\ARCHIVE\CHARMM\example_pdb.pdb") == ".pdb"


# ===========================================================================
# _normalise_query and _normalise_request
# ===========================================================================

class TestNormalise:

    def test_none_query_returns_empty_dict(self, sim_synthetic):
        assert sim_synthetic._normalise_query(None) == {}

    def test_dict_query_returns_copy(self, sim_synthetic):
        q = {"atom_name": ({"CA"}, set())}
        result = sim_synthetic._normalise_query(q)
        assert result == q
        assert result is not q

    def test_non_dict_query_raises(self, sim_synthetic):
        with pytest.raises(TypeError):
            sim_synthetic._normalise_query("bad_query")

    def test_list_query_raises(self, sim_synthetic):
        with pytest.raises(TypeError):
            sim_synthetic._normalise_query(["atom_name"])

    def test_none_request_returns_none(self, sim_synthetic):
        assert sim_synthetic._normalise_request(None) is None

    def test_request_stripped(self, sim_synthetic):
        assert sim_synthetic._normalise_request("  global_ids  ") == "global_ids"

    def test_empty_request_raises(self, sim_synthetic):
        with pytest.raises(ValueError):
            sim_synthetic._normalise_request("   ")

    def test_non_string_request_raises(self, sim_synthetic):
        with pytest.raises(TypeError):
            sim_synthetic._normalise_request(42)


# ===========================================================================
# _validate_query and _validate_request
# ===========================================================================

class TestValidation:

    def test_valid_typing_query_passes(self, sim_synthetic):
        assert sim_synthetic._validate_query({"atom_name": ({"CA"}, set())}, "typing") is True

    def test_invalid_typing_key_raises(self, sim_synthetic):
        with pytest.raises(ValueError, match="Invalid query keyword"):
            sim_synthetic._validate_query({"not_a_key": (set(), set())}, "typing")

    def test_valid_topology_request_passes(self, sim_synthetic):
        assert sim_synthetic._validate_request("global_ids", "topology") is True

    def test_invalid_topology_request_raises(self, sim_synthetic):
        with pytest.raises(ValueError, match="Invalid request"):
            sim_synthetic._validate_request("not_a_request", "topology")

    def test_invalid_query_type_raises(self, sim_synthetic):
        with pytest.raises(ValueError, match="Invalid query_type"):
            sim_synthetic._validate_query({}, "made_up_domain")

    def test_none_query_passes_validation(self, sim_synthetic):
        assert sim_synthetic._validate_query(None, "typing") is True

    def test_valid_trajectory_request_passes(self, sim_synthetic):
        assert sim_synthetic._validate_request("positions", "trajectory") is True

    def test_invalid_trajectory_request_raises(self, sim_synthetic):
        with pytest.raises(ValueError, match="Invalid request"):
            sim_synthetic._validate_request("not_a_traj_req", "trajectory")


# ===========================================================================
# _ensure_domain_loaded
# ===========================================================================

class TestEnsureDomainLoaded:

    def test_query_without_typing_raises(self):
        s = Sim()
        with pytest.raises(ValueError, match="No typing file"):
            s.get_types(QUERY={}, REQUEST="global_ids")

    def test_query_without_topology_raises(self):
        s = Sim()
        with pytest.raises(ValueError, match="No topology file"):
            s.get_topology(QUERY={}, REQUEST="global_ids")

    def test_query_without_trajectory_raises(self):
        s = Sim()
        with pytest.raises(ValueError, match="No trajectory file"):
            s.get_trajectory(QUERY={"global_ids": ([0], set())}, REQUEST="positions")


# ===========================================================================
# Request classification
# ===========================================================================

class TestClassification:
    """
    The API exposes a single _classify_request(domain, request) method.
    It returns "property" for property-* requests, and "payload" for all
    other recognised requests (including global_ids and positions).
    """

    def test_typing_global_ids_is_payload(self, sim_synthetic):
        assert sim_synthetic._classify_request("typing", "global_ids") == "payload"

    def test_typing_property_is_property(self, sim_synthetic):
        assert sim_synthetic._classify_request("typing", "property-number_of_atoms") == "property"

    def test_typing_positions_is_payload(self, sim_synthetic):
        assert sim_synthetic._classify_request("typing", "positions") == "payload"

    def test_typing_unknown_raises(self, sim_synthetic):
        with pytest.raises(ValueError):
            sim_synthetic._classify_request("typing", "made_up")

    def test_topology_global_ids_is_payload(self, sim_synthetic):
        assert sim_synthetic._classify_request("topology", "global_ids") == "payload"

    def test_topology_property_is_property(self, sim_synthetic):
        assert sim_synthetic._classify_request("topology", "property-system_charge") == "property"

    def test_trajectory_positions_is_payload(self, sim_synthetic):
        assert sim_synthetic._classify_request("trajectory", "positions") == "payload"

    def test_trajectory_property_prefix_is_property(self, sim_synthetic):
        # Any property-* string returns "property" regardless of whether it is
        # a real trajectory request, because the classification is prefix-based.
        assert sim_synthetic._classify_request("trajectory", "property-box_size-timeline") == "property"


# ===========================================================================
# positions() interface
# ===========================================================================

class TestPositionsGuard:

    def test_positions_empty_queries_selects_all_atoms(self, sim_synthetic):
        out = sim_synthetic.positions(TYPE_Q={}, TRAJ_Q={"frame_interval": (0, 0, 1)})
        assert out["selection"]["count"] == 4

    def test_positions_plan_flag_returns_plan_dict(self, sim_synthetic):
        out = sim_synthetic.positions(
            TYPE_Q={"atom_name": ({"CA"}, set())},
            TRAJ_Q={"frame_interval": (0, 0, 1)},
            planFlag=True,
        )
        assert "plan"    in out
        assert "payload" in out
        assert out["payload"] is None


# ===========================================================================
# select() interface
# ===========================================================================

class TestSelectInterface:

    def test_select_requires_at_least_one_request(self, sim_synthetic):
        with pytest.raises(ValueError):
            sim_synthetic.select(TYPE_Q={})

    def test_select_type_r_positions_raises(self, sim_synthetic):
        with pytest.raises(ValueError, match="not a property request"):
            sim_synthetic.select(TYPE_Q={}, TYPE_R="positions")

    def test_select_plan_flag_returns_none_payload(self, sim_synthetic):
        out = sim_synthetic.select(TYPE_Q={}, TYPE_R="property-number_of_atoms", planFlag=True)
        assert out["payload"] is None

    def test_select_returns_mode_property(self, sim_synthetic):
        out = sim_synthetic.select(TYPE_Q={}, TYPE_R="property-number_of_atoms")
        assert out["mode"] == "property"


# ===========================================================================
# global_system_properties
# ===========================================================================

class TestSystemProperties:

    def test_add_valid_info(self, synthetic_pdb, synthetic_psf, synthetic_dcd):
        s = Sim(typing=synthetic_pdb, topology=synthetic_psf, trajectory=synthetic_dcd)
        result = s.add_info({"sim_name": "test_run", "timestep": 2.0})
        assert result is True
        assert s.global_system_properties["sim_name"] == "test_run"
        assert s.global_system_properties["timestep"] == 2.0

    def test_add_invalid_key_prints_warning_but_does_not_raise(self, sim_synthetic, capsys):
        sim_synthetic.add_info({"not_a_real_property": 99})
        captured = capsys.readouterr()
        assert "Key Warning" in captured.out

    def test_add_info_returns_false_for_none(self, sim_synthetic):
        assert sim_synthetic.add_info(None) is False

    def test_globals_dictionary_kwarg_applied_on_construction(self):
        s = Sim(globals_dictionary={"sim_name": "my_run", "timestep": 1.0})
        assert s.global_system_properties["sim_name"] == "my_run"
        assert s.global_system_properties["timestep"] == 1.0
