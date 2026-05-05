"""
tests/unit/test_query_help.py
==============================
Unit tests for trajectory_kit._query_help — pure functions only.

Functions under test
--------------------
_normalise_query_pair         range_style=True and False
_match                        categorical matcher
_match_range_any_scalar       range matcher (with alias _match_range_scalar)
_merge_global_ids             set union helper
_freeze_query                 deterministic hashable representation
_normalise_bonded_with_pair   shorthand -> canonical pair
MAX_BONDED_WITH_DEPTH         module constant
"""

from __future__ import annotations

import pytest

from trajectory_kit import _query_help as qh


# ===========================================================================
# _normalise_query_pair — categorical (range_style=False)
# ===========================================================================

class TestNormaliseQueryPairCategorical:

    def test_none_returns_empty_sets(self):
        assert qh._normalise_query_pair(None) == (set(), set())

    def test_bare_string(self):
        inc, exc = qh._normalise_query_pair("CA")
        assert inc == {"CA"}
        assert exc == set()

    def test_set_passthrough(self):
        inc, exc = qh._normalise_query_pair({"CA", "CB"})
        assert inc == {"CA", "CB"}
        assert exc == set()

    def test_list_coerced_to_set(self):
        inc, exc = qh._normalise_query_pair(["CA", "CB", "N"])
        assert inc == {"CA", "CB", "N"}

    def test_include_exclude_tuple(self):
        inc, exc = qh._normalise_query_pair(({"CA"}, {"HN"}))
        assert inc == {"CA"} and exc == {"HN"}

    def test_single_element_tuple_wrapper(self):
        inc, exc = qh._normalise_query_pair((["CA", "CB"],))
        assert inc == {"CA", "CB"} and exc == set()


# ===========================================================================
# _normalise_query_pair — range/membership (range_style=True)
# ===========================================================================

class TestNormaliseQueryPairRange:

    def test_none_returns_empty(self):
        assert qh._normalise_query_pair(None, range_style=True) == ((), ())

    def test_tuple_is_range(self):
        inc, exc = qh._normalise_query_pair((1, 100), range_style=True)
        assert inc == ((1, 100),) and exc == ()

    def test_list_is_membership(self):
        inc, exc = qh._normalise_query_pair([1, 3, 5], range_style=True)
        assert inc == ((1, 1), (3, 3), (5, 5))

    def test_tuple_two_and_list_two_differ(self):
        """(1, 5) = range 1..5 ; [1, 5] = membership {1, 5} — crucial distinction"""
        r_range, _      = qh._normalise_query_pair((1, 5), range_style=True)
        r_membership, _ = qh._normalise_query_pair([1, 5], range_style=True)
        assert r_range == ((1, 5),)
        assert r_membership == ((1, 1), (5, 5))
        assert r_range != r_membership

    def test_open_low(self):
        inc, _ = qh._normalise_query_pair((None, 100), range_style=True)
        assert inc == ((None, 100),)

    def test_open_high(self):
        inc, _ = qh._normalise_query_pair((5, None), range_style=True)
        assert inc == ((5, None),)

    def test_inc_range_exc_membership(self):
        inc, exc = qh._normalise_query_pair(((1, 100), [50]), range_style=True)
        assert inc == ((1, 100),)
        assert exc == ((50, 50),)

    def test_bare_scalar_is_point(self):
        inc, _ = qh._normalise_query_pair(42, range_style=True)
        assert inc == ((42, 42),)

    def test_three_tuple_raises(self):
        with pytest.raises(ValueError, match="2-tuple"):
            qh._normalise_query_pair((1, 2, 3), range_style=True)


# ===========================================================================
# _match — categorical
# ===========================================================================

class TestMatch:

    def test_empty_include_matches_all(self):
        assert qh._match("CA", set(), set()) is True

    def test_include_hit(self):
        assert qh._match("CA", {"CA", "CB"}, set()) is True

    def test_include_miss(self):
        assert qh._match("HN", {"CA", "CB"}, set()) is False

    def test_exclude_rejects(self):
        assert qh._match("CA", {"CA"}, {"CA"}) is False

    def test_exclude_without_include(self):
        assert qh._match("CA", set(), {"HN"}) is True
        assert qh._match("HN", set(), {"HN"}) is False


# ===========================================================================
# _match_range_any_scalar — range matcher
# ===========================================================================

class TestMatchRange:

    def test_empty_matches_all(self):
        assert qh._match_range_any_scalar(50, (), ()) is True

    def test_in_range(self):
        assert qh._match_range_any_scalar(50, ((1, 100),), ()) is True

    def test_on_low_bound(self):
        assert qh._match_range_any_scalar(1, ((1, 100),), ()) is True

    def test_on_high_bound(self):
        assert qh._match_range_any_scalar(100, ((1, 100),), ()) is True

    def test_below_range(self):
        assert qh._match_range_any_scalar(0, ((1, 100),), ()) is False

    def test_above_range(self):
        assert qh._match_range_any_scalar(101, ((1, 100),), ()) is False

    def test_open_low(self):
        assert qh._match_range_any_scalar(-999, ((None, 100),), ()) is True

    def test_open_high(self):
        assert qh._match_range_any_scalar(999999, ((5, None),), ()) is True

    def test_membership_hit(self):
        assert qh._match_range_any_scalar(3, ((1, 1), (3, 3), (5, 5)), ()) is True

    def test_membership_miss(self):
        assert qh._match_range_any_scalar(2, ((1, 1), (3, 3), (5, 5)), ()) is False

    def test_exclude_overrides(self):
        assert qh._match_range_any_scalar(50, ((1, 100),), ((50, 50),)) is False

    def test_alias_is_same_function(self):
        assert qh._match_range_scalar is qh._match_range_any_scalar


# ===========================================================================
# _merge_global_ids
# ===========================================================================

class TestMergeGlobalIds:

    def test_returns_set(self):
        assert isinstance(qh._merge_global_ids([1, 2, 3]), set)

    def test_empty_input(self):
        assert qh._merge_global_ids() == set()

    def test_single_list(self):
        assert qh._merge_global_ids([1, 2, 3]) == {1, 2, 3}

    def test_multiple_lists_union(self):
        assert qh._merge_global_ids([1, 2], [2, 3], [4]) == {1, 2, 3, 4}

    def test_mixed_types(self):
        assert qh._merge_global_ids({1, 2}, [2, 3]) == {1, 2, 3}

    def test_duplicates_collapsed(self):
        assert qh._merge_global_ids([1, 1, 2, 2, 3]) == {1, 2, 3}


# ===========================================================================
# _freeze_query — deterministic hashable representation
# ===========================================================================

class TestFreezeQuery:

    def test_scalar_returned_as_is(self):
        assert qh._freeze_query(42) == 42
        assert qh._freeze_query("x") == "x"

    def test_dict_becomes_sorted_tuple(self):
        d = {"b": 2, "a": 1}
        frozen = qh._freeze_query(d)
        assert frozen == (("a", 1), ("b", 2))

    def test_dict_insertion_order_independent(self):
        d1 = {"a": 1, "b": 2, "c": 3}
        d2 = {"c": 3, "a": 1, "b": 2}
        assert qh._freeze_query(d1) == qh._freeze_query(d2)

    def test_nested_dict(self):
        d = {"outer": {"inner": "value"}}
        frozen = qh._freeze_query(d)
        assert frozen == (("outer", (("inner", "value"),)),)

    def test_list_becomes_tuple(self):
        assert qh._freeze_query([1, 2, 3]) == (1, 2, 3)

    def test_set_becomes_sorted_tuple(self):
        assert qh._freeze_query({3, 1, 2}) == (1, 2, 3)

    def test_frozen_result_is_hashable(self):
        """Cache keys must be hashable."""
        d = {"a": {"b": [1, 2, 3]}, "c": {4, 5}}
        frozen = qh._freeze_query(d)
        hash(frozen)  # should not raise

    def test_none_passthrough(self):
        assert qh._freeze_query(None) is None


# ===========================================================================
# _normalise_bonded_with_pair — shorthand normaliser
# ===========================================================================

class TestNormaliseBondedWithPair:

    BLOCK_A = {"total": True, "count": {"eq": 1}}
    BLOCK_B = {"neighbor": {"atom_type": "OT"}, "count": {"ge": 1}}

    # Accepted shapes
    def test_none_returns_empty_lists(self):
        assert qh._normalise_bonded_with_pair(None) == ([], [])

    def test_single_dict_becomes_single_include_block(self):
        inc, exc = qh._normalise_bonded_with_pair(self.BLOCK_A)
        assert inc == [self.BLOCK_A]
        assert exc == []

    def test_list_of_dicts_becomes_include_list(self):
        inc, exc = qh._normalise_bonded_with_pair([self.BLOCK_A, self.BLOCK_B])
        assert inc == [self.BLOCK_A, self.BLOCK_B]
        assert exc == []

    def test_empty_list_returns_empty_include(self):
        inc, exc = qh._normalise_bonded_with_pair([])
        assert inc == [] and exc == []

    def test_full_tuple_form(self):
        inc, exc = qh._normalise_bonded_with_pair(([self.BLOCK_A], [self.BLOCK_B]))
        assert inc == [self.BLOCK_A]
        assert exc == [self.BLOCK_B]

    def test_tuple_with_empty_exclude(self):
        inc, exc = qh._normalise_bonded_with_pair(([self.BLOCK_A], []))
        assert inc == [self.BLOCK_A] and exc == []

    # Validation / error cases
    def test_string_raises(self):
        with pytest.raises(ValueError, match="bonded_with must"):
            qh._normalise_bonded_with_pair("hello")

    def test_int_raises(self):
        with pytest.raises(ValueError, match="bonded_with must"):
            qh._normalise_bonded_with_pair(42)

    def test_list_of_nondict_raises(self):
        with pytest.raises(ValueError, match="must be a dict"):
            qh._normalise_bonded_with_pair([1, 2, 3])

    def test_list_of_mixed_raises(self):
        with pytest.raises(ValueError, match="must be a dict"):
            qh._normalise_bonded_with_pair([self.BLOCK_A, "not_a_dict"])

    def test_three_tuple_raises(self):
        with pytest.raises(ValueError, match="must be"):
            qh._normalise_bonded_with_pair((self.BLOCK_A, [], []))

    def test_tuple_with_nonlist_raises(self):
        with pytest.raises(ValueError, match="must be"):
            qh._normalise_bonded_with_pair(([self.BLOCK_A], "bad"))

    def test_tuple_with_nondict_block_raises(self):
        with pytest.raises(ValueError, match="must be a dict"):
            qh._normalise_bonded_with_pair(([self.BLOCK_A], [42]))

    # Shorthand semantic equivalences — all forms produce same canonical output
    def test_dict_and_single_list_equivalent(self):
        r_dict = qh._normalise_bonded_with_pair(self.BLOCK_A)
        r_list = qh._normalise_bonded_with_pair([self.BLOCK_A])
        assert r_dict == r_list

    def test_single_list_and_tuple_equivalent(self):
        r_list  = qh._normalise_bonded_with_pair([self.BLOCK_A])
        r_tuple = qh._normalise_bonded_with_pair(([self.BLOCK_A], []))
        assert r_list == r_tuple


# ===========================================================================
# Regression Int Bug — (None, None) normalises to "no constraint"
# ===========================================================================

class TestNoneNoneNormalisation:
    """``(None, None)`` is the unbounded range. It semantically means "no
    constraint" — not "exclude everything". All forms that a user could
    reasonably write to mean "no exclude" must produce identical canonical
    output."""

    def test_none_none_as_bare_range_is_empty(self):
        """The bare ``(None, None)`` input → both sides empty."""
        inc, exc = qh._normalise_query_pair((None, None), range_style=True)
        assert inc == ()
        assert exc == ()

    def test_none_none_on_exclude_side(self):
        """``((1, 5), (None, None))`` — exclude side should normalise to empty,
        leaving only the include range."""
        inc, exc = qh._normalise_query_pair(((1, 5), (None, None)), range_style=True)
        assert inc == ((1, 5),)
        assert exc == ()

    def test_none_none_on_include_side(self):
        """``((None, None), [99])`` — include side should normalise to empty,
        leaving only the exclude."""
        inc, exc = qh._normalise_query_pair(((None, None), [99]), range_style=True)
        assert inc == ()
        assert exc == ((99, 99),)

    def test_three_forms_of_no_exclude_equivalent(self):
        """These three forms all mean the same thing — no exclude constraint."""
        inc1, exc1 = qh._normalise_query_pair(((1, 5), None),         range_style=True)
        inc2, exc2 = qh._normalise_query_pair(((1, 5), ()),           range_style=True)
        inc3, exc3 = qh._normalise_query_pair(((1, 5), (None, None)), range_style=True)
        assert (inc1, exc1) == (inc2, exc2) == (inc3, exc3)
        assert exc1 == ()

    def test_both_sides_none_none(self):
        """``((None, None), (None, None))`` → fully unconstrained."""
        inc, exc = qh._normalise_query_pair(((None, None), (None, None)), range_style=True)
        assert inc == ()
        assert exc == ()


# ===========================================================================
# Regression Int Bug — empty set on exclude side in range-style
# ===========================================================================

class TestEmptyContainersInRangeStyle:
    """Range-style normalisation now accepts empty sets (they were previously
    mistaken for scalar point ranges). Also accepts populated sets as
    membership lists, same as Python list input."""

    def test_empty_set_exclude_is_no_constraint(self):
        """``([0, 1, 2], set())`` — exclude side is empty set, should produce
        canonical empty exclude."""
        inc, exc = qh._normalise_query_pair(([0, 1, 2], set()), range_style=True)
        assert inc == ((0, 0), (1, 1), (2, 2))
        assert exc == ()

    def test_empty_set_include_is_no_constraint(self):
        inc, exc = qh._normalise_query_pair((set(), [99]), range_style=True)
        assert inc == ()
        assert exc == ((99, 99),)

    def test_populated_set_becomes_membership(self):
        """A populated set on either side is semantically the same as a list —
        each element becomes a point range."""
        inc, exc = qh._normalise_query_pair(({3, 1, 2}, set()), range_style=True)
        # Sorted for determinism
        assert inc == ((1, 1), (2, 2), (3, 3))

    def test_frozen_set_handled(self):
        inc, exc = qh._normalise_query_pair((frozenset({5, 7}), frozenset()), range_style=True)
        assert inc == ((5, 5), (7, 7))
        assert exc == ()


# ===========================================================================
# Module constants
# ===========================================================================

class TestModuleConstants:

    def test_max_bonded_with_depth_is_16(self):
        assert qh.MAX_BONDED_WITH_DEPTH == 16

    def test_max_depth_is_int(self):
        assert isinstance(qh.MAX_BONDED_WITH_DEPTH, int)
