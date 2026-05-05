"""
tests/unit/test_main.py
=======================
Unit tests for pure/class-level helpers in trajectory_kit.main:

- _PLAN_DROP_KEYS / _PLAN_TIER_1_KEYS class-level contracts
- _standardise_plan — compute estimated_bytes from plan_shape, drop keys,
  tier ordering
- _normalise_query / _normalise_request static-style methods

These tests use a tiny sim instance built from nothing (no files attached)
to access the instance methods. File-dependent behavior is covered by
integration tests.
"""

from __future__ import annotations

import pytest

from trajectory_kit import sim as Sim


# ===========================================================================
# Class-level plan-key contracts
# ===========================================================================

class TestPlanKeyContracts:
    """These frozen sets/tuples encode a public contract with the planner.
    If they drift, downstream consumers of the plan dict silently break."""

    def test_tier1_includes_essential_sizing_keys(self):
        for required in ("n_atoms", "n_frames", "estimated_bytes"):
            assert required in Sim._PLAN_TIER_1_KEYS, f"tier-1 missing {required!r}"

    def test_tier1_includes_identification_keys(self):
        for required in ("source", "file_format", "request"):
            assert required in Sim._PLAN_TIER_1_KEYS

    def test_drop_keys_contains_mib(self):
        """estimated_mib is legacy and must be stripped from output plans
        (bytes is canonical)."""
        assert "estimated_mib" in Sim._PLAN_DROP_KEYS

    def test_drop_keys_contains_bytes_per_atom_per_frame(self):
        """Dropped because derivable from plan_shape, no need to echo
        back the raw planner output."""
        assert "bytes_per_atom_per_frame" in Sim._PLAN_DROP_KEYS

    def test_drop_keys_contains_file_type(self):
        assert "file_type" in Sim._PLAN_DROP_KEYS

    def test_drop_keys_contains_query_dictionary(self):
        """The full query dict is not echoed back in plans."""
        assert "query_dictionary" in Sim._PLAN_DROP_KEYS


# ===========================================================================
# Sim construction with no files — minimal instance for method access
# ===========================================================================

@pytest.fixture
def empty_sim():
    """A sim with no files loaded — enough to reach instance methods."""
    return Sim()


# ===========================================================================
# _standardise_plan — bytes computation and key discipline
# ===========================================================================

class TestStandardisePlan:

    def test_returns_dict(self, empty_sim):
        raw = {"planner_mode": "header", "n_atoms": 10, "n_frames": 1}
        plan_shape = ("per_atom", (3,), 12)
        out = empty_sim._standardise_plan(
            domain="typing", file_format="pdb", request="positions",
            raw_plan=raw, plan_shape=plan_shape,
        )
        assert isinstance(out, dict)

    def test_adds_source_and_file_format_and_request(self, empty_sim):
        out = empty_sim._standardise_plan(
            domain="typing", file_format="pdb", request="positions",
            raw_plan={"planner_mode": "header", "n_atoms": 10, "n_frames": 1},
            plan_shape=("per_atom", (3,), 12),
        )
        assert out["source"] == "typing"
        assert out["file_format"] == "pdb"
        assert out["request"] == "positions"

    def test_computes_estimated_bytes_from_shape(self, empty_sim):
        """bytes = n_atoms * n_frames * bytes_per_match from plan_shape."""
        raw = {"planner_mode": "header", "n_atoms": 100, "n_frames": 10}
        out = empty_sim._standardise_plan(
            domain="trajectory", file_format="dcd", request="positions",
            raw_plan=raw, plan_shape=("per_atom_per_frame", (3,), 12),
        )
        assert out["estimated_bytes"] == 100 * 10 * 12

    def test_scalar_request_raises_if_called_with_none_bytes(self, empty_sim):
        """_standardise_plan enforces a contract: if bytes_per_match is None
        (scalar_property or selector), the caller is expected to short-circuit
        BEFORE calling this function. Reaching the standardiser with None
        bytes is a contract violation and raises ValueError."""
        raw = {"planner_mode": "header", "n_atoms": 10, "n_frames": 1}
        with pytest.raises(ValueError, match="plan_shape contract violated"):
            empty_sim._standardise_plan(
                domain="typing", file_format="pdb",
                request="property-number_of_atoms",
                raw_plan=raw, plan_shape=("scalar_property", (), None),
            )

    def test_selector_request_raises_if_called_with_none_bytes(self, empty_sim):
        """Same contract: selector kind with None bytes should have
        short-circuited before this call."""
        raw = {"planner_mode": "header", "n_atoms": 10, "n_frames": 1}
        with pytest.raises(ValueError, match="plan_shape contract violated"):
            empty_sim._standardise_plan(
                domain="trajectory", file_format="dcd", request="global_ids",
                raw_plan=raw, plan_shape=("selector", (), None),
            )

    def test_drops_legacy_raw_values_not_output_keys(self, empty_sim):
        """_PLAN_DROP_KEYS controls what's filtered from raw_plan input (so
        the parser cannot override standardiser-owned values). The
        standardiser then emits its own canonical bytes_per_atom_per_frame
        in tier-1 output. So the rule is: after standardisation, drop-keys
        in the output come from the standardiser's computation, not from
        the raw_plan passthrough."""
        raw = {
            "planner_mode": "header",
            "n_atoms": 10, "n_frames": 1,
            "estimated_mib": 1.5,            # raw value must be dropped
            "bytes_per_atom_per_frame": 999, # raw value must be dropped (standardiser uses plan_shape)
            "file_type": "pdb",              # raw value must be dropped
            "query_dictionary": {"some": "thing"},  # raw value must be dropped
        }
        out = empty_sim._standardise_plan(
            domain="typing", file_format="pdb", request="positions",
            raw_plan=raw, plan_shape=("per_atom", (3,), 12),
        )
        # Standardiser's own bytes_per_atom_per_frame is in tier-1, not the raw 999
        assert out["bytes_per_atom_per_frame"] == 12
        # These should never appear anywhere:
        for leaked in ("estimated_mib", "file_type", "query_dictionary"):
            assert leaked not in out, f"dropped raw-key {leaked!r} leaked into output"

    def test_zero_atoms_zero_bytes(self, empty_sim):
        raw = {"planner_mode": "header", "n_atoms": 0, "n_frames": 1}
        out = empty_sim._standardise_plan(
            domain="typing", file_format="pdb", request="positions",
            raw_plan=raw, plan_shape=("per_atom", (3,), 12),
        )
        assert out["estimated_bytes"] == 0

    def test_single_frame_for_static_file(self, empty_sim):
        """Static (non-trajectory) files plan with n_frames=1."""
        raw = {"planner_mode": "header", "n_atoms": 54, "n_frames": 1}
        out = empty_sim._standardise_plan(
            domain="typing", file_format="pdb", request="positions",
            raw_plan=raw, plan_shape=("per_atom", (3,), 12),
        )
        assert out["n_frames"] == 1
        assert out["estimated_bytes"] == 54 * 1 * 12


# ===========================================================================
# _normalise_query / _normalise_request
# ===========================================================================

class TestNormaliseQuery:

    def test_none_becomes_empty_dict(self, empty_sim):
        assert empty_sim._normalise_query(None) == {}

    def test_dict_returned_as_copy(self, empty_sim):
        q = {"atom_name": ({"C1"}, set())}
        out = empty_sim._normalise_query(q)
        assert out == q
        assert out is not q  # must be a copy, not a reference

    def test_non_dict_raises(self, empty_sim):
        with pytest.raises(TypeError):
            empty_sim._normalise_query("bad")

    def test_list_raises(self, empty_sim):
        with pytest.raises(TypeError):
            empty_sim._normalise_query(["atom_name"])


class TestNormaliseRequest:

    def test_none_returns_none(self, empty_sim):
        assert empty_sim._normalise_request(None) is None

    def test_strips_whitespace(self, empty_sim):
        assert empty_sim._normalise_request("  global_ids  ") == "global_ids"

    def test_empty_raises(self, empty_sim):
        with pytest.raises(ValueError):
            empty_sim._normalise_request("   ")

    def test_non_string_raises(self, empty_sim):
        with pytest.raises(TypeError):
            empty_sim._normalise_request(42)


# ===========================================================================
# _get_filetype — extension detection
# ===========================================================================

class TestGetFiletype:

    def test_pdb_extension(self, empty_sim):
        assert empty_sim._get_filetype("file.pdb") == ".pdb"

    def test_psf_extension(self, empty_sim):
        assert empty_sim._get_filetype("path/to/file.psf") == ".psf"

    def test_dcd_extension(self, empty_sim):
        assert empty_sim._get_filetype("file.dcd") == ".dcd"

    def test_uppercase_normalised(self, empty_sim):
        assert empty_sim._get_filetype("FILE.PDB") == ".pdb"

    def test_windows_style_path(self, empty_sim):
        assert empty_sim._get_filetype(r"E:\some\dir\file.psf") == ".psf"
