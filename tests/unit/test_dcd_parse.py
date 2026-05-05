"""
tests/unit/test_dcd_parse.py
============================
Unit tests for pure (file-free) functions in trajectory_kit.dcd_parse.

Functions under test
--------------------
_get_trajectory_plan_shape_dcd

The DCD reader itself requires a real binary file and is covered by
integration tests.
"""

from __future__ import annotations

import pytest

from trajectory_kit.dcd_parse import _get_trajectory_plan_shape_dcd


# ===========================================================================
# _get_trajectory_plan_shape_dcd
# ===========================================================================

class TestDCDPlanShape:

    def test_positions(self):
        kind, shape, bpe = _get_trajectory_plan_shape_dcd("positions")
        assert kind == "per_atom_per_frame"
        assert shape == (3,)
        assert bpe == 12

    def test_global_ids_selector(self):
        kind, _, bpe = _get_trajectory_plan_shape_dcd("global_ids")
        assert kind == "selector"
        assert bpe is None

    def test_unknown_raises(self):
        with pytest.raises(ValueError):
            _get_trajectory_plan_shape_dcd("bad")
