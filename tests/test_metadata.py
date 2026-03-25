"""
test_metadata.py
================
Tests for global system property auto-population and cross-file atom count
validation.

Covers:
  - Auto-population of globals on file load (num_atoms, num_residues,
    num_frames, start_box_size) for every supported file type
  - Correct values against known synthetic file content
  - num_atoms consistency across typing / topology / trajectory
  - Mismatch detection raises ValueError with an informative message
  - Manual add_info() interaction with auto-populated values
  - Partial load scenarios

Atom layout in synthetic files (4 atoms, 3 residues, 3 frames):
    global_id 0  CA  ALA  PROT  resid=1  x=1   y=2   z=3
    global_id 1  CB  ALA  PROT  resid=1  x=4   y=5   z=6
    global_id 2  CA  GLY  PROT  resid=2  x=7   y=8   z=9
    global_id 3  OW  TIP  SOLV  resid=3  x=10  y=11  z=12
"""

from __future__ import annotations

import pytest
from trajectory_kit import sim as Sim


# ===========================================================================
# PDB globals auto-population
# ===========================================================================

class TestPDBGlobalsAutoPopulation:

    def test_num_atoms_populated_on_load(self, synthetic_pdb):
        s = Sim(typing=synthetic_pdb)
        assert s.global_system_properties["num_atoms"] == 4

    def test_num_residues_populated_on_load(self, synthetic_pdb):
        s = Sim(typing=synthetic_pdb)
        assert s.global_system_properties["num_residues"] == 3  # resids 1, 2, 3

    def test_start_box_size_populated_on_load(self, synthetic_pdb):
        s = Sim(typing=synthetic_pdb)
        box = s.global_system_properties["start_box_size"]
        assert box is not None
        xmin, xmax, ymin, ymax, zmin, zmax = box
        assert xmin == pytest.approx(1.0)
        assert xmax == pytest.approx(10.0)
        assert ymin == pytest.approx(2.0)
        assert ymax == pytest.approx(11.0)
        assert zmin == pytest.approx(3.0)
        assert zmax == pytest.approx(12.0)

    def test_start_box_size_is_six_tuple(self, synthetic_pdb):
        s = Sim(typing=synthetic_pdb)
        box = s.global_system_properties["start_box_size"]
        assert isinstance(box, tuple)
        assert len(box) == 6

    def test_start_box_size_min_lt_max(self, synthetic_pdb):
        s = Sim(typing=synthetic_pdb)
        xmin, xmax, ymin, ymax, zmin, zmax = s.global_system_properties["start_box_size"]
        assert xmin < xmax
        assert ymin < ymax
        assert zmin < zmax

    def test_globals_not_set_before_any_load(self):
        s = Sim()
        assert s.global_system_properties["num_atoms"]      is None
        assert s.global_system_properties["start_box_size"] is None
        assert s.global_system_properties["num_residues"]   is None
        assert s.global_system_properties["num_frames"]     is None

    def test_unrelated_globals_remain_none_after_pdb_load(self, synthetic_pdb):
        s = Sim(typing=synthetic_pdb)
        assert s.global_system_properties["num_frames"] is None
        assert s.global_system_properties["timestep"]   is None


# ===========================================================================
# XYZ globals auto-population
# ===========================================================================

class TestXYZGlobalsAutoPopulation:

    def test_num_atoms_populated_on_load(self, synthetic_xyz):
        s = Sim(typing=synthetic_xyz)
        assert s.global_system_properties["num_atoms"] == 4

    def test_start_box_size_populated_on_load(self, synthetic_xyz):
        s = Sim(typing=synthetic_xyz)
        box = s.global_system_properties["start_box_size"]
        assert box is not None
        xmin, xmax, ymin, ymax, zmin, zmax = box
        assert xmin == pytest.approx(1.0)
        assert xmax == pytest.approx(10.0)
        assert ymin == pytest.approx(2.0)
        assert ymax == pytest.approx(11.0)
        assert zmin == pytest.approx(3.0)
        assert zmax == pytest.approx(12.0)

    def test_start_box_size_is_six_tuple(self, synthetic_xyz):
        s = Sim(typing=synthetic_xyz)
        box = s.global_system_properties["start_box_size"]
        assert isinstance(box, tuple)
        assert len(box) == 6

    def test_start_box_size_matches_pdb(self, synthetic_pdb, synthetic_xyz):
        s_pdb = Sim(typing=synthetic_pdb)
        s_xyz = Sim(typing=synthetic_xyz)
        assert s_pdb.global_system_properties["start_box_size"] == \
               s_xyz.global_system_properties["start_box_size"]

    def test_num_residues_not_populated_by_xyz(self, synthetic_xyz):
        s = Sim(typing=synthetic_xyz)
        assert s.global_system_properties["num_residues"] is None

    def test_num_frames_not_populated_by_xyz(self, synthetic_xyz):
        s = Sim(typing=synthetic_xyz)
        assert s.global_system_properties["num_frames"] is None


# ===========================================================================
# PSF globals auto-population
# ===========================================================================

class TestPSFGlobalsAutoPopulation:

    def test_num_atoms_populated_on_load(self, synthetic_psf):
        s = Sim(topology=synthetic_psf)
        assert s.global_system_properties["num_atoms"] == 4

    def test_num_residues_populated_on_load(self, synthetic_psf):
        s = Sim(topology=synthetic_psf)
        assert s.global_system_properties["num_residues"] == 3

    def test_start_box_size_not_populated_by_psf(self, synthetic_psf):
        s = Sim(topology=synthetic_psf)
        assert s.global_system_properties["start_box_size"] is None

    def test_num_frames_not_populated_by_psf(self, synthetic_psf):
        s = Sim(topology=synthetic_psf)
        assert s.global_system_properties["num_frames"] is None


# ===========================================================================
# DCD globals auto-population
# ===========================================================================

class TestDCDGlobalsAutoPopulation:

    def test_num_atoms_populated_on_load(self, synthetic_dcd):
        s = Sim(trajectory=synthetic_dcd)
        assert s.global_system_properties["num_atoms"] == 4

    def test_num_frames_populated_on_load(self, synthetic_dcd):
        s = Sim(trajectory=synthetic_dcd)
        assert s.global_system_properties["num_frames"] == 3

    def test_start_box_size_not_populated_by_dcd(self, synthetic_dcd):
        s = Sim(trajectory=synthetic_dcd)
        assert s.global_system_properties["start_box_size"] is None

    def test_timestep_not_populated_by_dcd(self, synthetic_dcd):
        s = Sim(trajectory=synthetic_dcd)
        assert s.global_system_properties["timestep"] is None


# ===========================================================================
# Full stack — all three files
# ===========================================================================

class TestGlobalsFullStack:

    def test_all_globals_present_pdb_psf_dcd(self, synthetic_pdb, synthetic_psf, synthetic_dcd):
        s = Sim(typing=synthetic_pdb, topology=synthetic_psf, trajectory=synthetic_dcd)
        g = s.global_system_properties
        assert g["num_atoms"]      == 4
        assert g["num_frames"]     == 3
        assert g["start_box_size"] is not None

    def test_all_globals_present_xyz_psf_dcd(self, synthetic_xyz, synthetic_psf, synthetic_dcd):
        s = Sim(typing=synthetic_xyz, topology=synthetic_psf, trajectory=synthetic_dcd)
        g = s.global_system_properties
        assert g["num_atoms"]      == 4
        assert g["num_frames"]     == 3
        assert g["start_box_size"] is not None

    def test_num_atoms_consistent_across_all_three(self, synthetic_pdb, synthetic_psf, synthetic_dcd):
        s = Sim(typing=synthetic_pdb, topology=synthetic_psf, trajectory=synthetic_dcd)
        assert s.global_system_properties["num_atoms"] == 4

    def test_incremental_load_matches_constructor(self, synthetic_pdb, synthetic_psf, synthetic_dcd):
        s_all = Sim(typing=synthetic_pdb, topology=synthetic_psf, trajectory=synthetic_dcd)
        s_inc = Sim()
        s_inc.load_typing(synthetic_pdb)
        s_inc.load_topology(synthetic_psf)
        s_inc.load_trajectory(synthetic_dcd)
        assert s_all.global_system_properties == s_inc.global_system_properties


# ===========================================================================
# Partial load scenarios
# ===========================================================================

class TestGlobalsPartialLoad:

    def test_typing_only_populates_atoms_and_box(self, synthetic_pdb):
        s = Sim(typing=synthetic_pdb)
        assert s.global_system_properties["num_atoms"]      == 4
        assert s.global_system_properties["start_box_size"] is not None
        assert s.global_system_properties["num_frames"]     is None

    def test_typing_plus_topology_no_frames(self, synthetic_pdb, synthetic_psf):
        s = Sim(typing=synthetic_pdb, topology=synthetic_psf)
        assert s.global_system_properties["num_atoms"]  == 4
        assert s.global_system_properties["num_frames"] is None

    def test_trajectory_only_populates_atoms_and_frames(self, synthetic_dcd):
        s = Sim(trajectory=synthetic_dcd)
        assert s.global_system_properties["num_atoms"]      == 4
        assert s.global_system_properties["num_frames"]     == 3
        assert s.global_system_properties["start_box_size"] is None

    def test_topology_only_populates_atoms_and_residues(self, synthetic_psf):
        s = Sim(topology=synthetic_psf)
        assert s.global_system_properties["num_atoms"]      == 4
        assert s.global_system_properties["num_residues"]   == 3
        assert s.global_system_properties["start_box_size"] is None


# ===========================================================================
# Atom count cross-validation
# ===========================================================================

class TestAtomCountCrossValidation:

    def test_matching_counts_do_not_raise(self, synthetic_pdb, synthetic_psf, synthetic_dcd):
        Sim(typing=synthetic_pdb, topology=synthetic_psf, trajectory=synthetic_dcd)

    def test_mismatched_dcd_vs_pdb_raises(self, synthetic_pdb, mismatched_dcd):
        with pytest.raises(ValueError, match="Atom count mismatch"):
            Sim(typing=synthetic_pdb, trajectory=mismatched_dcd)

    def test_mismatched_psf_vs_pdb_raises(self, synthetic_pdb, mismatched_psf):
        with pytest.raises(ValueError, match="Atom count mismatch"):
            Sim(typing=synthetic_pdb, topology=mismatched_psf)

    def test_mismatched_dcd_vs_xyz_raises(self, synthetic_xyz, mismatched_dcd):
        with pytest.raises(ValueError, match="Atom count mismatch"):
            Sim(typing=synthetic_xyz, trajectory=mismatched_dcd)

    def test_mismatched_psf_vs_dcd_raises(self, synthetic_dcd, mismatched_psf):
        with pytest.raises(ValueError, match="Atom count mismatch"):
            Sim(topology=mismatched_psf, trajectory=synthetic_dcd)

    def test_error_message_contains_both_counts(self, synthetic_pdb, mismatched_dcd):
        with pytest.raises(ValueError) as exc_info:
            Sim(typing=synthetic_pdb, trajectory=mismatched_dcd)
        msg = str(exc_info.value)
        assert "4"  in msg
        assert "99" in msg

    def test_error_message_contains_filenames(self, synthetic_pdb, mismatched_dcd):
        with pytest.raises(ValueError) as exc_info:
            Sim(typing=synthetic_pdb, trajectory=mismatched_dcd)
        msg = str(exc_info.value)
        assert "synthetic.pdb" in msg or "pdb" in msg.lower()
        assert "mismatch.dcd"  in msg or "dcd" in msg.lower()

    def test_mismatch_caught_on_second_file_incremental(self, synthetic_pdb, mismatched_dcd):
        s = Sim(typing=synthetic_pdb)
        with pytest.raises(ValueError, match="Atom count mismatch"):
            s.load_trajectory(mismatched_dcd)

    def test_mismatch_caught_on_third_file_incremental(self, synthetic_pdb, synthetic_psf, mismatched_dcd):
        s = Sim(typing=synthetic_pdb, topology=synthetic_psf)
        with pytest.raises(ValueError, match="Atom count mismatch"):
            s.load_trajectory(mismatched_dcd)

    def test_single_file_never_raises_pdb(self, synthetic_pdb):
        Sim(typing=synthetic_pdb)

    def test_single_file_never_raises_dcd(self, synthetic_dcd):
        Sim(trajectory=synthetic_dcd)

    def test_single_file_never_raises_psf(self, synthetic_psf):
        Sim(topology=synthetic_psf)


# ===========================================================================
# add_info() interaction with auto-populated globals
# ===========================================================================

class TestAddInfoInteraction:

    def test_add_info_overwrites_auto_populated_value(self, synthetic_pdb):
        s = Sim(typing=synthetic_pdb)
        assert s.global_system_properties["num_atoms"] == 4
        s.add_info({"num_atoms": 9999})
        assert s.global_system_properties["num_atoms"] == 9999

    def test_add_info_fills_unpopulated_fields(self, synthetic_pdb):
        s = Sim(typing=synthetic_pdb)
        assert s.global_system_properties["timestep"] is None
        s.add_info({"timestep": 2.0, "sim_name": "myrun"})
        assert s.global_system_properties["timestep"] == 2.0
        assert s.global_system_properties["sim_name"] == "myrun"

    def test_add_info_returns_true_on_success(self, synthetic_pdb):
        s = Sim(typing=synthetic_pdb)
        assert s.add_info({"sim_name": "ok"}) is True

    def test_add_info_invalid_key_warns_does_not_raise(self, synthetic_pdb, capsys):
        s = Sim(typing=synthetic_pdb)
        s.add_info({"not_a_property": 42})
        captured = capsys.readouterr()
        assert "Key Warning" in captured.out

    def test_globals_dict_kwarg_merges_with_file_values(self, synthetic_pdb):
        s = Sim(
            typing=synthetic_pdb,
            globals_dictionary={"sim_name": "ctor_run", "timestep": 1.0},
        )
        assert s.global_system_properties["sim_name"]  == "ctor_run"
        assert s.global_system_properties["timestep"]  == 1.0
        assert s.global_system_properties["num_atoms"] == 4

    def test_auto_populated_values_do_not_crash_when_manual_set_first(self, synthetic_psf):
        s = Sim(globals_dictionary={"num_residues": 999})
        s.load_topology(synthetic_psf)
        assert s.global_system_properties["num_residues"] is not None
