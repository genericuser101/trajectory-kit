"""
tests/integration/test_envelope.py
===================================
Integration tests for the dev/plan envelope emitted by fetch(),
positions(), select() when ``devFlag=True`` and/or ``planFlag=True``.

Covers:
- Bare payload returned when devFlag=False (regardless of planFlag)
- Full 5-key envelope when devFlag=True
- planFlag silent without devFlag
- planFlag=True + devFlag=True returns plan but None payload
- Combined cross-domain plan estimate present and correctly totalled
- Required keys/subkeys at each envelope level
"""

from __future__ import annotations

import numpy as np
import pytest

from trajectory_kit import sim as Sim

from conftest import N_ATOMS, N_FRAMES


@pytest.fixture(scope="session")
def sim_full(synthetic_pdb, synthetic_psf, synthetic_dcd):
    return Sim(typing=synthetic_pdb, topology=synthetic_psf, trajectory=synthetic_dcd)


# ===========================================================================
# Bare vs envelope output — devFlag controls shape
# ===========================================================================

class TestDevFlagBehavior:

    def test_default_returns_bare_payload(self, sim_full):
        """Without devFlag, fetch returns bare payload (no envelope)."""
        result = sim_full.fetch(TYPE_Q={}, TYPE_R="atom_names")
        # bare payload = dict keyed by domain, no envelope keys
        assert "mode" not in result
        assert "plan" not in result
        assert "metadata" not in result

    def test_devflag_true_returns_envelope(self, sim_full):
        result = sim_full.fetch(TYPE_Q={}, TYPE_R="atom_names", devFlag=True)
        for key in ("mode", "selection", "metadata", "plan", "payload"):
            assert key in result, f"envelope missing {key!r}"

    def test_devflag_payload_contains_data(self, sim_full):
        result = sim_full.fetch(TYPE_Q={}, TYPE_R="atom_names", devFlag=True)
        assert result["payload"] is not None
        assert "typing" in result["payload"]
        assert len(result["payload"]["typing"]) == N_ATOMS

    def test_devflag_mode_is_fetch(self, sim_full):
        result = sim_full.fetch(TYPE_Q={}, TYPE_R="atom_names", devFlag=True)
        assert result["mode"] == "fetch"

    def test_positions_mode_is_positions(self, sim_full):
        result = sim_full.positions(
            TYPE_Q={}, TRAJ_Q={"frame_interval": (0, 0, 1)}, devFlag=True,
        )
        assert result["mode"] == "positions"

    def test_select_mode_is_property_or_select(self, sim_full):
        result = sim_full.select(
            TYPE_Q={}, TYPE_R="property-number_of_atoms", devFlag=True,
        )
        # mode is "property" for property requests
        assert result["mode"] in {"property", "select"}


# ===========================================================================
# planFlag behavior
# ===========================================================================

class TestPlanFlag:

    def test_planflag_silent_without_devflag(self, sim_full):
        """planFlag=True with devFlag=False should still return bare payload."""
        result = sim_full.fetch(TYPE_Q={}, TYPE_R="atom_names", planFlag=True)
        assert "plan" not in result    # no envelope => no plan exposed

    def test_planflag_true_devflag_true_payload_none(self, sim_full):
        result = sim_full.fetch(
            TYPE_Q={}, TYPE_R="atom_names",
            devFlag=True, planFlag=True,
        )
        assert result["payload"] is None

    def test_planflag_true_includes_plan(self, sim_full):
        result = sim_full.fetch(
            TYPE_Q={}, TYPE_R="atom_names",
            devFlag=True, planFlag=True,
        )
        assert result["plan"] is not None
        assert isinstance(result["plan"], dict)

    def test_planflag_false_executes_and_returns_payload(self, sim_full):
        result = sim_full.fetch(
            TYPE_Q={}, TYPE_R="atom_names",
            devFlag=True, planFlag=False,
        )
        assert result["payload"] is not None


# ===========================================================================
# Plan structure — single-domain
# ===========================================================================

class TestPlanStructureSingleDomain:

    def test_typing_plan_has_domain_subkey(self, sim_full):
        result = sim_full.fetch(TYPE_Q={}, TYPE_R="atom_names",
                                devFlag=True, planFlag=True)
        assert "typing" in result["plan"]

    def test_typing_plan_has_n_atoms(self, sim_full):
        result = sim_full.fetch(TYPE_Q={}, TYPE_R="atom_names",
                                devFlag=True, planFlag=True)
        plan_t = result["plan"]["typing"]
        assert "n_atoms" in plan_t
        assert plan_t["n_atoms"] == N_ATOMS

    def test_typing_plan_has_n_frames(self, sim_full):
        """Every plan advertises n_frames — static files emit 1."""
        result = sim_full.fetch(TYPE_Q={}, TYPE_R="atom_names",
                                devFlag=True, planFlag=True)
        plan_t = result["plan"]["typing"]
        assert plan_t.get("n_frames") == 1

    def test_typing_plan_estimated_bytes(self, sim_full):
        """For 'atom_names' (string categorical) estimated_bytes may be None
        (selector kind) OR an int. Accept either."""
        result = sim_full.fetch(TYPE_Q={}, TYPE_R="atom_names",
                                devFlag=True, planFlag=True)
        plan_t = result["plan"]["typing"]
        # Test only that the key exists
        assert "estimated_bytes" in plan_t

    def test_positions_plan_estimated_bytes_is_int(self, sim_full):
        """positions is a per-atom float32x3 → bytes are known."""
        result = sim_full.fetch(TYPE_Q={}, TYPE_R="positions",
                                devFlag=True, planFlag=True)
        plan_t = result["plan"]["typing"]
        assert plan_t["estimated_bytes"] is not None
        # 54 atoms * 1 frame * 12 bytes per point = 648 bytes
        assert plan_t["estimated_bytes"] == N_ATOMS * 1 * 12


# ===========================================================================
# Combined cross-domain plan estimate
# ===========================================================================

class TestCombinedPlan:

    def test_combined_present_for_single_domain(self, sim_full):
        result = sim_full.fetch(TYPE_Q={}, TYPE_R="positions",
                                devFlag=True, planFlag=True)
        assert "combined" in result["plan"]

    def test_combined_has_n_atoms_upper_bound(self, sim_full):
        result = sim_full.fetch(TYPE_Q={}, TYPE_R="positions",
                                devFlag=True, planFlag=True)
        combined = result["plan"]["combined"]
        assert "n_atoms_upper_bound" in combined
        assert combined["n_atoms_upper_bound"] == N_ATOMS

    def test_combined_has_total_estimated_bytes(self, sim_full):
        result = sim_full.fetch(TYPE_Q={}, TYPE_R="positions",
                                devFlag=True, planFlag=True)
        combined = result["plan"]["combined"]
        assert "total_estimated_bytes" in combined

    def test_combined_multi_domain_sums_bytes(self, sim_full):
        """When asking for positions in typing and trajectory, the combined
        total must equal the sum of the per-domain bytes (within the
        n_atoms upper-bound logic)."""
        result = sim_full.fetch(
            TYPE_Q={}, TYPE_R="positions",
            TRAJ_Q={"global_ids": (list(range(N_ATOMS)), set()),
                    "frame_interval": (0, N_FRAMES - 1, 1)},
            TRAJ_R="positions",
            devFlag=True, planFlag=True,
        )
        plan = result["plan"]
        type_bytes = plan["typing"]["estimated_bytes"]
        traj_bytes = plan["trajectory"]["estimated_bytes"]
        assert plan["combined"]["total_estimated_bytes"] == type_bytes + traj_bytes

    def test_combined_upper_bound_is_file_natom(self, sim_full):
        """n_atoms_upper_bound is the minimum n_atoms advertised across the
        requested per-atom domains. Planners look at file headers and do
        not pre-resolve global_ids-list queries, so even a 3-id trajectory
        query shows the full file's natom (54) in the upper bound."""
        result = sim_full.fetch(
            TYPE_Q={}, TYPE_R="positions",
            TRAJ_Q={"global_ids": ([0, 1, 2], set()),
                    "frame_interval": (0, 0, 1)},
            TRAJ_R="positions",
            devFlag=True, planFlag=True,
        )
        # All domains advertise 54 atoms from their headers.
        assert result["plan"]["combined"]["n_atoms_upper_bound"] == N_ATOMS


# ===========================================================================
# Selection block — top-level envelope key
# ===========================================================================

class TestSelectionBlock:

    def test_selection_present(self, sim_full):
        result = sim_full.fetch(TYPE_Q={}, TYPE_R="atom_names", devFlag=True)
        assert "selection" in result
        assert isinstance(result["selection"], dict)

    def test_selection_has_resolved_count_key(self, sim_full):
        """Selection always carries the resolved_count key. Its value is
        populated for cross-domain fetches and None for single-domain
        fetches or for planFlag short-circuited calls."""
        result = sim_full.fetch(TYPE_Q={}, TYPE_R="atom_names", devFlag=True)
        assert "resolved_count" in result["selection"]

    def test_resolved_count_populated_for_positions(self, sim_full):
        """positions() always intersects across loaded domains and populates
        resolved_count. Plain fetch() without id-list intersection leaves
        it None because there's no selection to resolve."""
        result = sim_full.positions(
            TYPE_Q={}, TOPO_Q={},
            TRAJ_Q={"frame_interval": (0, 0, 1)},
            devFlag=True,
        )
        assert result["selection"]["resolved_count"] == N_ATOMS

    def test_resolved_count_populated_when_ids_provided(self, sim_full):
        """Explicit global_ids in the query triggers id-list intersection
        and populates resolved_count with the intersection size."""
        result = sim_full.fetch(
            TYPE_Q={"global_ids": ([0, 1, 2, 3, 4], set())},
            TYPE_R="atom_names",
            TOPO_Q={"global_ids": ([2, 3, 4, 5, 6], set())},
            TOPO_R="charges",
            devFlag=True,
        )
        # {0..4} & {2..6} = {2, 3, 4} → 3 atoms survive intersection
        assert result["selection"]["resolved_count"] == 3

    def test_selection_per_domain_flags_present(self, sim_full):
        """Each active domain carries query_provided / ids_provided / n_matched."""
        result = sim_full.fetch(TYPE_Q={}, TYPE_R="atom_names", devFlag=True)
        sel = result["selection"]
        assert "typing" in sel
        for key in ("query_provided", "ids_provided", "n_matched"):
            assert key in sel["typing"]

    def test_selection_resolved_count_none_with_planflag(self, sim_full):
        """With planFlag, no execution happens, so resolved_count stays None."""
        result = sim_full.fetch(
            TYPE_Q={}, TYPE_R="atom_names",
            devFlag=True, planFlag=True,
        )
        assert result["selection"]["resolved_count"] is None


# ===========================================================================
# Metadata block
# ===========================================================================

class TestMetadataBlock:

    def test_metadata_present(self, sim_full):
        result = sim_full.fetch(TYPE_Q={}, TYPE_R="atom_names", devFlag=True)
        assert "metadata" in result
        assert isinstance(result["metadata"], dict)
