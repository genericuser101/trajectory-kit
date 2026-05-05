"""
tests/unit/test_xyz_parse.py
============================
Unit tests for pure (file-free) functions in trajectory_kit.xyz_parse.

Functions under test
--------------------
_parse_xyz_atom_row
_get_xyz_type_predicate_state
_xyz_atom_matches_query
_get_type_plan_shape_xyz

File-dependent functions (planner, query executor, global-updater)
live in integration tests.
"""

from __future__ import annotations

import pytest

from trajectory_kit.xyz_parse import (
    _parse_xyz_atom_row,
    _get_xyz_type_predicate_state,
    _xyz_atom_matches_query,
    _get_type_plan_shape_xyz,
)


# ===========================================================================
# _parse_xyz_atom_row
# ===========================================================================

class TestParseXYZAtomRow:

    def _row(self, name="C1", x=0.0, y=0.0, z=0.0):
        return f"{name:<6s} {x:10.4f} {y:10.4f} {z:10.4f}\n"

    def test_returns_dict(self):
        r = _parse_xyz_atom_row(self._row(), 0)
        assert isinstance(r, dict)

    def test_global_id(self):
        assert _parse_xyz_atom_row(self._row(), 3)["global_id"] == 3

    def test_atom_name(self):
        assert _parse_xyz_atom_row(self._row(name="OH2"), 0)["atom_name"] == "OH2"

    def test_coordinates(self):
        r = _parse_xyz_atom_row(self._row(x=1.0, y=2.0, z=3.0), 0)
        assert abs(r["x"] - 1.0) < 1e-3
        assert abs(r["y"] - 2.0) < 1e-3
        assert abs(r["z"] - 3.0) < 1e-3


# ===========================================================================
# _get_type_plan_shape_xyz
# ===========================================================================

class TestXYZPlanShape:

    def test_global_ids(self):
        kind, _, _ = _get_type_plan_shape_xyz("global_ids")
        assert kind == "per_atom"

    def test_positions(self):
        kind, shape, bpe = _get_type_plan_shape_xyz("positions")
        assert kind == "per_atom"
        assert shape == (3,)

    def test_number_of_atoms_scalar(self):
        kind, _, _ = _get_type_plan_shape_xyz("property-number_of_atoms")
        assert kind == "scalar_property"

    def test_unknown_raises(self):
        with pytest.raises(ValueError):
            _get_type_plan_shape_xyz("bad")
