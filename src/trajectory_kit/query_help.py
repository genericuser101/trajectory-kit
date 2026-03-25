from __future__ import annotations
from typing import TypeVar

TNum = TypeVar('TNum', int, float)

def _normalise_query_pair(value, *, range_style: bool = False):

    '''
    Coerce a raw query value into a canonical ``(include, exclude)`` pair.

    Accepted shapes
    ---------------
    ``None`` or ``()``
        No constraint → ``(set(), set())`` for set-style,
                         ``(None, None)``  for range-style.

    Single non-tuple value  (str / int / float / set / list)
        Treated as the *include* side only →
        ``(normalised_include, empty_exclude)``.

    ``(inc,)``  — one-element tuple
        Also treated as include-only.

    ``(inc, exc)`` — two-element tuple
        Used as-is after normalising each side.

    Normalisation of each side
    --------------------------
    *set-style* (``range_style=False``):
        Each side is coerced to a ``set``.  A bare string/int/float becomes
        ``{value}``.  ``None`` / ``()`` / ``[]`` become ``set()``.

    *range-style* (``range_style=True``):
        Each side is a single ``(lo, hi)`` pair where either bound may be
        ``None`` (meaning unbounded on that end).
        A bare ``(lo, hi)`` pair is used directly.
        ``None`` / ``()`` / ``[]`` become ``(None, None)`` (no constraint).

    Parameters
    ----------
    value :
        Raw value from the query dictionary.
    range_style : bool
        ``True``  → range semantics: a single ``(lo, hi)`` pair.
        ``False`` → set/membership semantics.

    Returns
    -------
    tuple
        ``(include_side, exclude_side)`` in canonical form.
    '''

    empty = (None, None) if range_style else set()

    def _norm_side(side):
        if side is None:
            return empty
        if range_style:
            if isinstance(side, (list, tuple)) and len(side) == 0:
                return (None, None)
            # a single (lo, hi) pair
            if isinstance(side, (list, tuple)) and len(side) == 2:
                return (side[0], side[1])
            raise ValueError(
                f"range-style query side must be a (lo, hi) pair, got {side!r}"
            )
        else:
            # set-style
            if isinstance(side, set):
                return side
            if isinstance(side, (list, tuple)) and len(side) == 0:
                return set()
            if isinstance(side, (list, tuple)):
                return set(side)
            # bare scalar
            return {side}

    #dispatch on shape of value 

    if value is None or (isinstance(value, (tuple, list)) and len(value) == 0):
        return empty, empty

    if isinstance(value, tuple) and len(value) == 2:
        first, second = value

        # Distinguish a plain (lo, hi) range pair from a (inc, exc) pair.
        # Heuristic: if range_style and neither element is itself a
        # tuple/list/set/None, treat it as a single range pair, not (inc,exc).
        if range_style:
            first_is_container = isinstance(first, (tuple, list, set))
            second_is_container = isinstance(second, (tuple, list, set))
            if not first_is_container and not second_is_container:
                # bare (lo, hi) pair → include only
                return (first, second), (None, None)

        return _norm_side(first), _norm_side(second)

    if isinstance(value, tuple) and len(value) == 1:
        return _norm_side(value[0]), empty

    # anything else (bare scalar, set, list) → include-only
    return _norm_side(value), empty

def _match(value: str, 
           inc: set[str], 
           exc: set[str],) -> bool:
    
    '''
    This function checks if a given value matches the include and exclude sets. Exclusion wins over inclusion.

    Parameters:
    ----------
    value: str
        The value to check for a match against the include and exclude sets.
    inc: set[str]
        A set of values that should be included in the match. If empty, there are no positive constraints.
    exc: set[str]
        A set of values that should be excluded from the match. If empty, there are no negative constraints.

    Returns:
    -------
    bool
        True if the value matches the include and exclude sets, False otherwise.
    
    '''

    if value in exc:
        return False
    if not inc:
        return True
    return value in inc

def _match_range_scalar(value: TNum,
                        include_range: tuple[TNum | None, TNum | None],
                        exclude_range: tuple[TNum | None, TNum | None],) -> bool:
    
    '''
    The function checks if a given value falls within a specified range defined by lower and upper bounds. The bounds are inclusive.
    The bounds are inclusive because of faster hotpaths, I will not be changing this. 

    Parameters:
    ----------
    value: TNum
        The value to check against the specified range.
    include_range: tuple[TNum | None, TNum | None]
        A single (lo, hi) pair. Either bound may be None (unbounded).
        (None, None) means no inclusion constraint.
    exclude_range: tuple[TNum | None, TNum | None]
        A single (lo, hi) pair. Either bound may be None (unbounded).
        (None, None) means no exclusion constraint.

    Returns:
    -------
    bool
        True if the value falls within the specified range, False otherwise.
    '''

    exc_lo, exc_hi = exclude_range
    if exc_lo is not None or exc_hi is not None: # exclusion wins over inclusion
        if (exc_lo is None or value >= exc_lo) and (exc_hi is None or value <= exc_hi):
            return False

    inc_lo, inc_hi = include_range
    if inc_lo is None and inc_hi is None: # no include constraint => include everything not excluded
        return True

    return (inc_lo is None or value >= inc_lo) and (inc_hi is None or value <= inc_hi)

def _merge_global_ids(*global_ids): 

    '''
    This function merges a pool of global ids into a single unique set.

    Parameters:
    ----------
    globals_ids: set[int] | list[int] | nd.array[int]
        An unspecified number of global id lists, sets of numpy arrays.

    Returns:
    -------
    unique_globals: set[int]
        A single merged 
    '''

    unique_globals = set()

    for globs in global_ids:

        unique_globals.update(globs)

    return unique_globals