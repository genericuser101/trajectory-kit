"""
tests/unit/test_mae_parse.py
============================
Unit tests for pure (file-free) functions in trajectory_kit.mae_parse.

Functions under test
--------------------
_tokenise_mae_line
_get_m_atom_predicate
_does_m_atom_query_match
_get_type_plan_shape_mae
_get_topology_plan_shape_mae

File-dependent functions (block readers, planners, query executors,
ffio extractors, _filter_mae_by_bonded_with) live in integration tests.
"""

from __future__ import annotations

import pytest

from trajectory_kit.mae_parse import (
    _tokenise_mae_line,
    _get_m_atom_predicate,
    _does_m_atom_query_match,
    _get_type_plan_shape_mae,
    _get_topology_plan_shape_mae,
)


# ===========================================================================
# _tokenise_mae_line
# ===========================================================================

class TestTokeniseMaeLine:

    def test_simple_tokens(self):
        assert _tokenise_mae_line("1  C1  DRG") == ["1", "C1", "DRG"]

    def test_quoted_string(self):
        assert _tokenise_mae_line('1  "a quoted string"  foo') == ["1", "a quoted string", "foo"]

    def test_empty_line(self):
        assert _tokenise_mae_line("   \n") == []

    def test_quoted_empty(self):
        assert _tokenise_mae_line('1  ""  x') == ["1", "", "x"]


# ===========================================================================
# _get_m_atom_predicate / _does_m_atom_query_match
# ===========================================================================

class TestMAEAtomPredicate:

    def test_empty_query_query_controlled_need_flags_false(self):
        """Same principle as PSF: only test flags that are flipped by
        query keys. MAE defaults several 'need_' flags True because the
        parser always emits those fields; that's by design."""
        ps = _get_m_atom_predicate({})
        query_controlled = {
            "need_atom", "need_atf", "need_resn", "need_chain", "need_seg",
            "need_mres", "need_grow", "need_ins",
        }
        for flag in query_controlled:
            assert ps[flag] is False, f"{flag} should be False for empty query"

    def test_atom_name_sets_need_atom(self):
        ps = _get_m_atom_predicate({"atom_name": ({"C1"}, set())})
        assert ps["need_atom"] is True

    def test_residue_name_sets_need_resn(self):
        ps = _get_m_atom_predicate({"residue_name": ({"DRG"}, set())})
        assert ps["need_resn"] is True

    def test_empty_matches_full_atom(self):
        """With a fully populated atom dict, empty-query predicate matches."""
        ps = _get_m_atom_predicate({})
        full = {
            "local_id": 1, "atom_name": "C1", "atom_name_full": "C1",
            "residue_name": "DRG", "chain_name": "L", "segment_name": "LIG",
            "mmod_res": "", "grow_name": "", "insertion_code": "",
            "residue_id": 1, "atomic_number": 6, "mmod_type": 1,
            "color": 0, "visibility": 1, "formal_charge": 0,
            "secondary_structure": 0, "h_count": 0, "representation": 0,
            "template_index": 0,
            "x": 0.0, "y": 0.0, "z": 0.0,
            "v_x": 0.0, "v_y": 0.0, "v_z": 0.0,
            "partial_charge_1": 0.0, "partial_charge_2": 0.0,
            "pdb_tfactor": 0.1, "pdb_occupancy": 1.0,
        }
        assert _does_m_atom_query_match(full, ps) is True


# ===========================================================================
# _get_type_plan_shape_mae / _get_topology_plan_shape_mae
# ===========================================================================

class TestMAEPlanShape:

    def test_type_global_ids(self):
        kind, _, _ = _get_type_plan_shape_mae("global_ids")
        assert kind == "per_atom"

    def test_topology_charges(self):
        kind, _, _ = _get_topology_plan_shape_mae("charges")
        assert kind == "per_atom"

    def test_topology_system_charge_scalar(self):
        kind, _, bpe = _get_topology_plan_shape_mae("property-system_charge")
        assert kind == "scalar_property"
        assert bpe is None

    def test_type_unknown_raises(self):
        with pytest.raises(ValueError):
            _get_type_plan_shape_mae("bad_request")

    def test_topology_unknown_raises(self):
        with pytest.raises(ValueError):
            _get_topology_plan_shape_mae("bad_request")
