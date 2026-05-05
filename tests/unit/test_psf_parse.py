"""
tests/unit/test_psf_parse.py
============================
Unit tests for pure (file-free) functions in trajectory_kit.psf_parse.

Functions under test
--------------------
_parse_psf_atom_row
_is_psf_natom_record_line
_get_psf_topology_predicate_state
_psf_atom_matches_query
_get_topology_plan_shape_psf

File-dependent functions (planners, query executors, the !NBOND walker,
_build_local_to_global_to_type_map) live in integration tests.
"""

from __future__ import annotations

import pytest

from trajectory_kit.psf_parse import (
    _parse_psf_atom_row,
    _is_psf_natom_record_line,
    _get_psf_topology_predicate_state,
    _psf_atom_matches_query,
    _get_topology_plan_shape_psf,
)


# ===========================================================================
# _parse_psf_atom_row
# ===========================================================================

class TestParsePSFAtomRow:

    def _row(self, serial, seg, resi, resn, name, atype, charge, mass):
        return (f"{serial:10d} {seg:<8s} {resi:<8d} {resn:<8s} "
                f"{name:<8s} {atype:<8s} {charge:13.6f} {mass:13.4f}           0\n")

    def test_returns_dict(self):
        r = _parse_psf_atom_row(self._row(1, "LIG", 1, "DRG", "C1", "CA", -0.1, 12.0), 0)
        assert isinstance(r, dict)

    def test_global_id_from_index(self):
        r = _parse_psf_atom_row(self._row(1, "LIG", 1, "DRG", "C1", "CA", 0.0, 12.0), 5)
        assert r["global_id"] == 5

    def test_local_id_from_serial(self):
        r = _parse_psf_atom_row(self._row(42, "LIG", 1, "DRG", "C1", "CA", 0.0, 12.0), 0)
        assert r["local_id"] == 42

    def test_atom_name_parsed(self):
        r = _parse_psf_atom_row(self._row(1, "LIG", 1, "DRG", "OH2", "OT", 0.0, 16.0), 0)
        assert r["atom_name"] == "OH2"

    def test_atom_type_parsed(self):
        r = _parse_psf_atom_row(self._row(1, "LIG", 1, "DRG", "C1", "CA", 0.0, 12.0), 0)
        assert r["atom_type"] == "CA"

    def test_charge_float(self):
        r = _parse_psf_atom_row(self._row(1, "LIG", 1, "DRG", "C1", "CA", -0.115, 12.0), 0)
        assert abs(r["charge"] - (-0.115)) < 1e-4

    def test_mass_float(self):
        r = _parse_psf_atom_row(self._row(1, "LIG", 1, "DRG", "C1", "CA", 0.0, 12.011), 0)
        assert abs(r["mass"] - 12.011) < 1e-4


# ===========================================================================
# _is_psf_natom_record_line
# ===========================================================================

class TestIsPSFNatomRecordLine:

    def test_valid_row(self):
        row = ("         1 LIG      1        DRG      "
               "C1       CA         -0.115000       12.0110           0\n")
        assert _is_psf_natom_record_line(row) is True

    def test_natom_header_rejected(self):
        assert _is_psf_natom_record_line("      54 !NATOM\n") is False

    def test_nbond_header_rejected(self):
        assert _is_psf_natom_record_line("      40 !NBOND: bonds\n") is False

    def test_empty_line_rejected(self):
        assert _is_psf_natom_record_line("\n") is False

    def test_comment_rejected(self):
        assert _is_psf_natom_record_line(" REMARKS synthetic\n") is False


# ===========================================================================
# _get_psf_topology_predicate_state
# ===========================================================================

class TestPSFTopologyPredicateState:

    KWS = {"atom_name", "atom_type", "residue_name", "segment_name",
           "residue_ids", "local_ids", "charge", "mass", "is_virtual"}

    def test_empty_query_query_controlled_need_flags_false(self):
        """Flags that are only set by a matching query key should be False
        for an empty query. Some flags (e.g. need_charge, need_mass) default
        True because the parser always emits those fields — that's by
        design and not tested here."""
        ps = _get_psf_topology_predicate_state({}, self.KWS)
        query_controlled = {
            "need_atom", "need_atomt", "need_resn", "need_seg", "need_virt",
        }
        for flag in query_controlled:
            assert ps[flag] is False, f"{flag} should be False for empty query"

    def test_atom_name_sets_need(self):
        ps = _get_psf_topology_predicate_state({"atom_name": ({"C1"}, set())}, self.KWS)
        assert ps["need_atom"] is True

    def test_atom_type_sets_need(self):
        ps = _get_psf_topology_predicate_state({"atom_type": ({"OT"}, set())}, self.KWS)
        assert ps["need_atomt"] is True

    def test_charge_range_sets_need(self):
        ps = _get_psf_topology_predicate_state({"charge": ((0.0, None), (None, None))}, self.KWS)
        assert ps["need_charge"] is True


# ===========================================================================
# _psf_atom_matches_query
# ===========================================================================

class TestPSFAtomMatchesQuery:

    KWS = {"atom_name", "atom_type", "residue_name", "segment_name",
           "residue_ids", "local_ids", "charge", "mass", "is_virtual"}

    def _atom(self, **kw):
        base = {
            "global_id": 0, "local_id": 1,
            "atom_name": "C1", "atom_type": "CA",
            "residue_name": "DRG", "residue_id": 1, "segment_name": "LIG",
            "charge": -0.115, "mass": 12.011,
            "is_virtual": 0, "drude_alpha": 0.0, "drude_thole": 0.0,
        }
        base.update(kw)
        return base

    def test_empty_query_matches(self):
        ps = _get_psf_topology_predicate_state({}, self.KWS)
        assert _psf_atom_matches_query(self._atom(), ps) is True

    def test_atom_name_include(self):
        ps = _get_psf_topology_predicate_state({"atom_name": ({"C1"}, set())}, self.KWS)
        assert _psf_atom_matches_query(self._atom(atom_name="C1"), ps) is True
        assert _psf_atom_matches_query(self._atom(atom_name="C2"), ps) is False

    def test_exclude_wins(self):
        ps = _get_psf_topology_predicate_state({"atom_name": ({"C1"}, {"C1"})}, self.KWS)
        assert _psf_atom_matches_query(self._atom(atom_name="C1"), ps) is False

    def test_charge_range(self):
        # exclude side: empty tuple means "no exclude" in canonical form.
        # (None, None) on the exclude side would be normalised as an
        # unbounded range, which matches everything and thus excludes all.
        ps = _get_psf_topology_predicate_state({"charge": ((0.0, None),)}, self.KWS)
        assert _psf_atom_matches_query(self._atom(charge=0.5),  ps) is True
        assert _psf_atom_matches_query(self._atom(charge=-0.5), ps) is False


# ===========================================================================
# _get_topology_plan_shape_psf
# ===========================================================================

class TestPSFPlanShape:

    def test_global_ids_per_atom(self):
        kind, shape, bpe = _get_topology_plan_shape_psf("global_ids")
        assert kind == "per_atom"

    def test_charges_per_atom(self):
        kind, shape, bpe = _get_topology_plan_shape_psf("charges")
        assert kind == "per_atom"
        assert bpe is not None

    def test_system_charge_scalar(self):
        kind, shape, bpe = _get_topology_plan_shape_psf("property-system_charge")
        assert kind == "scalar_property"
        assert bpe is None

    def test_unknown_raises(self):
        with pytest.raises(ValueError):
            _get_topology_plan_shape_psf("not_a_request")

    @pytest.mark.parametrize("req", [
        "global_ids", "local_ids", "residue_ids", "atom_names", "atom_types",
        "residue_names", "segment_names", "charges", "masses",
        "property-system_charge",
    ])
    def test_all_known_return_triple(self, req):
        result = _get_topology_plan_shape_psf(req)
        assert isinstance(result, tuple) and len(result) == 3
