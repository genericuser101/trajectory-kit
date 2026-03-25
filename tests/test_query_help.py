"""
test_query_help.py
==================
Unit tests for the primitive matching functions in query_help.py.
No files required — runs on every machine.
"""

from __future__ import annotations

import pytest
from trajectory_kit import query_help as qh


# ===========================================================================
# _match  (categorical include/exclude)
# ===========================================================================

class TestMatch:

    def test_include_hit(self):
        assert qh._match("CA", {"CA", "CB"}, set()) is True

    def test_include_miss(self):
        assert qh._match("N", {"CA", "CB"}, set()) is False

    def test_empty_include_accepts_all(self):
        assert qh._match("anything", set(), set()) is True

    def test_exclude_wins_over_include(self):
        assert qh._match("CA", {"CA"}, {"CA"}) is False

    def test_exclude_only_hit(self):
        assert qh._match("CB", set(), {"CB"}) is False

    def test_exclude_only_miss(self):
        assert qh._match("CA", set(), {"CB"}) is True

    def test_empty_both_accepts_all(self):
        assert qh._match("whatever", set(), set()) is True

    def test_case_sensitive(self):
        # match must be exact — "ca" != "CA"
        assert qh._match("ca", {"CA"}, set()) is False


# ===========================================================================
# _match_range_scalar  (numeric include/exclude range)
# ===========================================================================

class TestMatchRangeScalar:

    # basic include
    def test_value_inside_include_range(self):
        assert qh._match_range_scalar(5.0, (0.0, 10.0), (None, None)) is True

    def test_value_at_lower_bound_inclusive(self):
        assert qh._match_range_scalar(0.0, (0.0, 10.0), (None, None)) is True

    def test_value_at_upper_bound_inclusive(self):
        assert qh._match_range_scalar(10.0, (0.0, 10.0), (None, None)) is True

    def test_value_outside_include_range(self):
        assert qh._match_range_scalar(11.0, (0.0, 10.0), (None, None)) is False

    # unbounded ranges
    def test_unbounded_lower(self):
        assert qh._match_range_scalar(-999.0, (None, 10.0), (None, None)) is True

    def test_unbounded_upper(self):
        assert qh._match_range_scalar(999.0, (0.0, None), (None, None)) is True

    def test_fully_unbounded_include(self):
        assert qh._match_range_scalar(42.0, (None, None), (None, None)) is True

    # exclude
    def test_excluded_value_rejected(self):
        assert qh._match_range_scalar(5.0, (None, None), (-1.0, 10.0)) is False

    def test_value_outside_exclude_range_accepted(self):
        assert qh._match_range_scalar(15.0, (None, None), (-1.0, 10.0)) is True

    # exclude wins over include
    def test_exclude_wins_over_include(self):
        assert qh._match_range_scalar(5.0, (0.0, 10.0), (4.0, 6.0)) is False

    # no constraint = accept all
    def test_no_constraint_accepts_anything(self):
        assert qh._match_range_scalar(42.0, (None, None), (None, None)) is True

    # integer values
    def test_integer_value_in_range(self):
        assert qh._match_range_scalar(50, (1, 100), (None, None)) is True

    def test_integer_value_excluded(self):
        assert qh._match_range_scalar(15, (1, 100), (10, 20)) is False

    def test_exact_boundary_values(self):
        assert qh._match_range_scalar(7, (7, 7), (None, None)) is True
        assert qh._match_range_scalar(6, (7, 7), (None, None)) is False
        assert qh._match_range_scalar(8, (7, 7), (None, None)) is False


# ===========================================================================
# _merge_global_ids
# ===========================================================================

class TestMergeGlobalIds:

    def test_single_input(self):
        assert qh._merge_global_ids([1, 2, 3]) == {1, 2, 3}

    def test_two_disjoint_inputs(self):
        assert qh._merge_global_ids([1, 2], [3, 4]) == {1, 2, 3, 4}

    def test_overlapping_inputs_deduplicates(self):
        assert qh._merge_global_ids([1, 2, 3], [3, 4, 5]) == {1, 2, 3, 4, 5}

    def test_empty_inputs(self):
        assert qh._merge_global_ids([], []) == set()

    def test_single_empty_input(self):
        assert qh._merge_global_ids([]) == set()

    def test_set_input(self):
        assert qh._merge_global_ids({10, 20}, {20, 30}) == {10, 20, 30}

    def test_mixed_types(self):
        import numpy as np
        arr = np.array([1, 2, 3], dtype=np.int32)
        assert qh._merge_global_ids(arr, [4, 5]) == {1, 2, 3, 4, 5}

    def test_three_inputs(self):
        assert qh._merge_global_ids([1], [2], [3]) == {1, 2, 3}

    def test_returns_set(self):
        result = qh._merge_global_ids([1, 2])
        assert isinstance(result, set)


# ===========================================================================
# _normalise_query_pair
# ===========================================================================

class TestNormaliseQueryPairSetStyle:
    """range_style=False (default) — membership / set semantics."""

    def test_none_returns_empty_sets(self):
        inc, exc = qh._normalise_query_pair(None)
        assert inc == set() and exc == set()

    def test_empty_tuple_returns_empty_sets(self):
        inc, exc = qh._normalise_query_pair(())
        assert inc == set() and exc == set()

    def test_empty_list_returns_empty_sets(self):
        inc, exc = qh._normalise_query_pair([])
        assert inc == set() and exc == set()

    def test_bare_string_becomes_include_set(self):
        inc, exc = qh._normalise_query_pair("CA")
        assert inc == {"CA"} and exc == set()

    def test_bare_set_becomes_include(self):
        inc, exc = qh._normalise_query_pair({"CA", "CB"})
        assert inc == {"CA", "CB"} and exc == set()

    def test_bare_list_becomes_include_set(self):
        inc, exc = qh._normalise_query_pair(["CA", "CB"])
        assert inc == {"CA", "CB"} and exc == set()

    def test_one_element_tuple_treated_as_include(self):
        inc, exc = qh._normalise_query_pair(({"CA"},))
        assert inc == {"CA"} and exc == set()

    def test_two_element_tuple_inc_exc(self):
        inc, exc = qh._normalise_query_pair(({"CA"}, {"CB"}))
        assert inc == {"CA"} and exc == {"CB"}

    def test_two_element_tuple_empty_exc(self):
        inc, exc = qh._normalise_query_pair(({"CA", "CB"}, set()))
        assert inc == {"CA", "CB"} and exc == set()

    def test_two_element_tuple_empty_inc(self):
        inc, exc = qh._normalise_query_pair((set(), {"CB"}))
        assert inc == set() and exc == {"CB"}

    def test_list_sides_coerced_to_sets(self):
        inc, exc = qh._normalise_query_pair((["CA"], ["CB"]))
        assert inc == {"CA"} and exc == {"CB"}

    def test_none_sides_become_empty_sets(self):
        inc, exc = qh._normalise_query_pair((None, None))
        assert inc == set() and exc == set()

    def test_exclude_wins_in_downstream_match(self):
        inc, exc = qh._normalise_query_pair(({"CA"}, {"CA"}))
        assert qh._match("CA", inc, exc) is False


class TestNormaliseQueryPairRangeStyle:
    """range_style=True — single (lo, hi) pair."""

    def test_none_returns_none_none_tuples(self):
        inc, exc = qh._normalise_query_pair(None, range_style=True)
        assert inc == (None, None) and exc == (None, None)

    def test_empty_tuple_returns_none_none_tuples(self):
        inc, exc = qh._normalise_query_pair((), range_style=True)
        assert inc == (None, None) and exc == (None, None)

    def test_bare_lo_hi_pair_becomes_include(self):
        inc, exc = qh._normalise_query_pair((0.0, 10.0), range_style=True)
        assert inc == (0.0, 10.0) and exc == (None, None)

    def test_bare_int_pair_becomes_include(self):
        inc, exc = qh._normalise_query_pair((1, 5), range_style=True)
        assert inc == (1, 5) and exc == (None, None)

    def test_full_inc_exc_pair(self):
        inc, exc = qh._normalise_query_pair(((0.0, 5.0), (3.0, 4.0)), range_style=True)
        assert inc == (0.0, 5.0) and exc == (3.0, 4.0)

    def test_one_element_tuple_treated_as_include(self):
        inc, exc = qh._normalise_query_pair(((0.0, 5.0),), range_style=True)
        assert inc == (0.0, 5.0) and exc == (None, None)

    def test_empty_list_sides(self):
        inc, exc = qh._normalise_query_pair(([], []), range_style=True)
        assert inc == (None, None) and exc == (None, None)

    def test_bare_pair_used_in_downstream_match(self):
        inc, exc = qh._normalise_query_pair((0.0, 10.0), range_style=True)
        assert qh._match_range_scalar(5.0, inc, exc) is True
        assert qh._match_range_scalar(15.0, inc, exc) is False