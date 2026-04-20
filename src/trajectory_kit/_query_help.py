from __future__ import annotations
from typing import Iterable, TypeVar

TNum = TypeVar('TNum', int, float)

def _normalise_query_pair(value, *, range_style: bool = False):

    '''
    Coerce a raw query value into a canonical ``(include, exclude)`` pair.

    The container type is the sole disambiguation signal — no ordering
    heuristics are needed.

    Input contract — range-style keywords
    --------------------------------------
    **tuple**  → range.
        ``(lo, hi)``     a single inclusive range.  Either bound may be
                         ``None`` for open-ended: ``(None, 100)`` means
                         "up to 100", ``(5, None)`` means "5 or above".
        Tuples must always be exactly 2 elements; any other length raises
        ``ValueError``.  Only one range is supported per side — multiple
        ranges are not a supported feature.

    **list**  → membership.
        ``[1, 3, 5, 77]``  match exactly these values.
        ``[5]``            match exactly this one value.
        Stored internally as point ranges ``(v, v)`` so the output flows
        through ``_match_range_any_scalar`` unchanged.

    **bare scalar**  → equivalent to a single-element list.
        ``5``  is  ``[5]``

    **None** or empty sequence  → no constraint (match everything).

    Include / exclude wrapper
    -------------------------
    Any of the above can be wrapped in a two-element tuple to supply both
    an include side and an exclude side:

        ``(inc, exc)``

    Each side is resolved independently using the rules above.
    The wrapper is distinguished from a bare ``(lo, hi)`` range by inspecting
    the elements: if *both* elements are plain scalars (not list / tuple /
    None) the outer tuple is treated as a range; if either element is a
    container or None the outer tuple is an ``(inc, exc)`` wrapper.

    Examples::

        (1, 100)                 -> include range  1..100
        (None, 100)              -> include range  up to 100
        (5, None)                -> include range  5 and above
        [1, 3, 5]                -> include members {1, 3, 5}
        5                        -> include member  {5}
        ([1, 3, 5], [77])        -> include {1,3,5},     exclude {77}
        ((1, 100), [77])         -> include range 1..100, exclude {77}
        ([1, 3, 5], (10, 20))    -> include {1,3,5},     exclude range 10..20
        ((1, 100), (200, 300))   -> include range 1..100, exclude range 200..300
        (None, (200, 300))       -> no include constraint, exclude range 200..300

    Input contract — set-style keywords
    -------------------------------------
    Bare scalar, list, or set  -> include set.
    ``(inc, exc)``             -> include set, exclude set.
    Both sides coerced to ``set``.

    Parameters
    ----------
    value :
        Raw value from the query dictionary.
    range_style : bool
        ``True``  -> range/membership semantics as described above.
        ``False`` -> pure set/membership semantics.

    Returns
    -------
    tuple
        ``(include_side, exclude_side)`` in canonical form.
        Range-style : each side is a tuple of ``(lo, hi)`` pairs, or ``()``.
        Set-style   : each side is a ``set``, or ``set()``.
    '''

    empty = () if range_style else set()

    def _norm_side(side):
        '''Normalise one side (inc or exc) to its canonical form.'''

        if side is None:
            return empty

        if range_style:
            # Empty sequence / empty set -> no constraint
            if isinstance(side, (list, tuple, set, frozenset)) and len(side) == 0:
                return ()

            # set/frozenset -> membership (same semantics as list)
            if isinstance(side, (set, frozenset)):
                return tuple((v, v) for v in sorted(side))

            # tuple -> range: must be exactly (lo, hi)
            if isinstance(side, tuple):
                if len(side) != 2:
                    raise ValueError(
                        f"A range must be a 2-tuple (lo, hi); "
                        f"got {side!r} with {len(side)} element(s). "
                        f"To pass a membership list use a list, not a tuple."
                    )
                # (None, None) is the unbounded range, which semantically
                # means "no constraint" — normalise it to empty on both
                # include and exclude sides so users cannot accidentally
                # exclude every value by writing ((1,5), (None,None)).
                if side[0] is None and side[1] is None:
                    return ()
                return (side,)

            # list -> membership: each scalar becomes a point range (v, v)
            if isinstance(side, list):
                return tuple((v, v) for v in side)

            # bare scalar -> single-point membership
            return ((side, side),)

        else:
            # set-style
            if isinstance(side, set):
                return side
            if isinstance(side, (list, tuple)) and len(side) == 0:
                return set()
            if isinstance(side, (list, tuple)):
                return set(side)
            return {side}

    # ------------------------------------------------------------------ #
    # Top-level dispatch                                                   #
    # ------------------------------------------------------------------ #

    # No constraint
    if value is None or (isinstance(value, (list, tuple)) and len(value) == 0):
        return empty, empty

    # Two-element tuple: either a (lo, hi) range or an (inc, exc) wrapper.
    #
    # Range if neither element is a container (list / tuple / set):
    #   (scalar, scalar)  e.g. (1, 100)
    #   (None, scalar)    e.g. (None, 100)  open lo-bound
    #   (scalar, None)    e.g. (5, None)    open hi-bound
    #   (None, None)      normalised to "no constraint" (empty, empty)
    #
    # Wrapper if either element is a container (list, tuple, set, frozenset).
    if isinstance(value, tuple) and len(value) == 2:
        first, second = value
        if range_style:
            _container_types = (list, tuple, set, frozenset)
            first_is_container  = isinstance(first,  _container_types)
            second_is_container = isinstance(second, _container_types)
            if not first_is_container and not second_is_container:
                # (None, None) is the unbounded range -> no constraint.
                if first is None and second is None:
                    return empty, empty
                # bare (lo, hi) range — include only
                return ((first, second),), ()
        # at least one side is a container -> (inc, exc) wrapper
        return _norm_side(first), _norm_side(second)

    # Single-element tuple -> include only, unwrap
    if isinstance(value, tuple) and len(value) == 1:
        return _norm_side(value[0]), empty

    # Anything else (list, bare scalar, set) -> include only
    return _norm_side(value), empty


def _match(value: str,
           inc: set[str],
           exc: set[str],) -> bool:

    '''
    Check if a string value matches the include and exclude sets.
    Exclusion wins over inclusion.

    Parameters:
    ----------
    value: str
        The value to check.
    inc: set[str]
        Values to include. Empty means no positive constraint.
    exc: set[str]
        Values to exclude. Empty means no negative constraint.

    Returns:
    -------
    bool
        True if the value passes both constraints.
    '''

    if value in exc:
        return False
    if not inc:
        return True
    return value in inc


def _match_range_any_scalar(value: TNum,
                            include_ranges: Iterable[tuple[TNum | None, TNum | None]],
                            exclude_ranges: Iterable[tuple[TNum | None, TNum | None]],) -> bool:

    '''
    Check if a numeric value falls within any of the include ranges and
    none of the exclude ranges.  Bounds are inclusive.  Exclusion wins.

    Parameters:
    ----------
    value: float
        The value to check.
    include_ranges: iterable of (lo, hi) pairs
        Each pair defines an inclusive range.  Either bound may be None
        for open-ended.  Empty means no positive constraint (match all).
    exclude_ranges: iterable of (lo, hi) pairs
        Same structure.  A match here causes the function to return False
        regardless of include_ranges.

    Returns:
    -------
    bool
        True if value is matched by at least one include range (or there
        are no include ranges) and is not matched by any exclude range.
    '''

    for lo, hi in exclude_ranges:
        if (lo is None or value >= lo) and (hi is None or value <= hi):
            return False

    any_include = False
    for lo, hi in include_ranges:
        any_include = True
        if (lo is None or value >= lo) and (hi is None or value <= hi):
            return True

    return not any_include


def _merge_global_ids(*global_ids):

    '''
    Merge an arbitrary number of global-id collections into one unique set.

    Parameters:
    ----------
    global_ids: set[int] | list[int] | np.ndarray[int]
        Any number of collections of global atom indices.

    Returns:
    -------
    set[int]
        A single merged set of unique global ids.
    '''

    unique_globals = set()
    for globs in global_ids:
        unique_globals.update(globs)
    return unique_globals


# ---------------------------------------------------------------------------
# bonded_with normalisation
# ---------------------------------------------------------------------------

# Maximum recursion depth for nested bonded_with neighbour queries. A neighbour
# sub-query may itself contain a bonded_with constraint, which spawns another
# query resolution. This ceiling guards against pathological inputs causing
# stack overflow. 16 is far above any realistic chemistry pattern (depth-5 is
# already exotic) but allows graph-walking patterns to compose freely.
MAX_BONDED_WITH_DEPTH = 16


def _freeze_query(obj):

    '''
    Recursively convert a query value into a deterministic, hashable form
    suitable for use as a cache key.

    Dicts become tuples of sorted (key, frozen_value) pairs; lists and tuples
    become tuples of frozen elements; sets become tuples of sorted elements;
    other values are returned as-is.

    Used by parser-level neighbour caches so that semantically identical
    queries hash to the same key regardless of dict insertion order or set
    iteration order.

    Parameters
    ----------
    obj :
        Any query value (dict / list / tuple / set / scalar).

    Returns
    -------
    A hashable, deterministically-ordered representation of ``obj``.
    '''
    if isinstance(obj, dict):
        return tuple(sorted((k, _freeze_query(v)) for k, v in obj.items()))
    if isinstance(obj, (list, tuple)):
        return tuple(_freeze_query(x) for x in obj)
    if isinstance(obj, set):
        return tuple(sorted(obj))
    return obj


def _normalise_bonded_with_pair(value):

    '''
    Normalise a ``bonded_with`` query value to a canonical
    ``(include_blocks, exclude_blocks)`` pair of lists.

    Accepted shapes
    ---------------
    | Input                            | Result                       | Notes                                   |
    |----------------------------------|------------------------------|-----------------------------------------|
    | ``None``                         | ``([], [])``                 | no bond filter (full pass)              |
    | ``{...}`` (a single dict)        | ``([{...}], [])``            | shorthand: single include block         |
    | ``[{...}, {...}]``               | ``([...], [])``              | shorthand: multi-block include only     |
    | ``([...], [...])``               | passthrough (lists)          | full include + exclude                  |

    The shorthand forms make the common case (one or more include blocks, no
    excludes) trivially short to write. The explicit tuple form is always
    accepted for queries that need both sides.

    Parameters
    ----------
    value :
        Raw value from ``query_dictionary["bonded_with"]``. May be missing
        (caller should pass ``None`` in that case).

    Returns
    -------
    tuple[list[dict], list[dict]]
        ``(include_blocks, exclude_blocks)`` — both always lists of dicts,
        possibly empty.

    Raises
    ------
    ValueError
        If ``value`` is not one of the accepted shapes, or if any block
        within an include/exclude list is not a dict.
    '''

    if value is None:
        return [], []

    # Bare dict — single include block shorthand.
    if isinstance(value, dict):
        return [value], []

    # Bare list — multi-block include only shorthand.
    if isinstance(value, list):
        for i, blk in enumerate(value):
            if not isinstance(blk, dict):
                raise ValueError(
                    f"bonded_with: each block must be a dict; "
                    f"got {type(blk).__name__} at index {i}: {blk!r}"
                )
        return list(value), []

    # 2-tuple — full (include, exclude) form.
    if isinstance(value, tuple):
        if len(value) != 2:
            raise ValueError(
                f"bonded_with: tuple form must be (include_list, exclude_list); "
                f"got {len(value)}-tuple: {value!r}"
            )
        inc, exc = value
        if not isinstance(inc, (list, tuple)) or not isinstance(exc, (list, tuple)):
            raise ValueError(
                f"bonded_with tuple must be (include_list, exclude_list) where "
                f"each side is a list of block dicts; "
                f"got inc={type(inc).__name__}, exc={type(exc).__name__}"
            )
        for label, side in (("include", inc), ("exclude", exc)):
            for i, blk in enumerate(side):
                if not isinstance(blk, dict):
                    raise ValueError(
                        f"bonded_with: each block must be a dict; "
                        f"got {type(blk).__name__} in {label} list at index {i}: {blk!r}"
                    )
        return list(inc), list(exc)

    raise ValueError(
        f"bonded_with must be a dict (single block), list of dicts (include "
        f"blocks), or (include_list, exclude_list) tuple; "
        f"got {type(value).__name__}: {value!r}"
    )

# ---------------------------------------------------------------------------
# Alias — the parse modules reference this shorter name.
# _match_range_any_scalar is the canonical name; _match_range_scalar is kept
# for backwards compatibility with all existing parser call sites.
# ---------------------------------------------------------------------------
_match_range_scalar = _match_range_any_scalar