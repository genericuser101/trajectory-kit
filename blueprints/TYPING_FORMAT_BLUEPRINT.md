# Adding a New Typing Format

A typing file provides per-atom identity and coordinate information —
atom names, residue names, segment names, positions, etc. Use
`pdb_parse.py` (most complete reference) or `xyz_parse.py` (minimal)
as the reference implementation.

For the end-to-end workflow including test wiring, see
`ADDING_A_FORMAT.md`. This document specifies the **parser contract**
only.

---

## Registration

In `main.py`, add the new extension to the typing domain registry:

```python
"typing": {
    "supported_formats": {".pdb", ".xyz", ".mae", ".your_ext"},
    ...
}
```

Nothing else changes. The five function templates (`keys_fn_template`,
`plan_fn_template`, `plan_shape_fn_template`, `query_fn_template`,
`update_fn_template`) auto-resolve to your new functions by name.

---

## Required Functions (5)

Every typing parser implements **five** functions. The fifth — the plan
shape contract — is mandatory and was missing from older versions of
this blueprint.

### 1. `_get_type_keys_reqs_{fmt}`

Returns queryable keywords and available request strings for this format.

```python
def _get_type_keys_reqs_{fmt}(filepath: str | Path) -> tuple[set[str], set[str]]:
```

**Keywords** — fields users can filter on in `TYPE_Q`. Standard names:

| Keyword | Style | Description |
|---------|-------|-------------|
| `"global_ids"` | range | positional atom index (0-based, framework-canonical) |
| `"local_ids"` | range | file-internal serial number (format-specific indexing) |
| `"atom_name"` | set | atom name string |
| `"residue_name"` | set | residue name string |
| `"residue_ids"` | range | residue sequence number |
| `"segment_name"` | set | segment / chain identifier |
| `"x"`, `"y"`, `"z"` | range | coordinates in Å |

**Style** controls how `qh._normalise_query_pair` interprets the user's
input. Use `range_style=True` for numeric ranges and integer membership;
the default (`range_style=False`) is set membership for strings.

**Requests** — outputs users can ask for. Must include `"global_ids"` and
`"positions"`. Standard names:

| Request | Returns |
|---------|---------|
| `"global_ids"` | `list[int]` — matching atom indices |
| `"local_ids"` | `list[int]` |
| `"atom_names"` | `list[str]` |
| `"residue_ids"` | `list[int]` |
| `"residue_names"` | `list[str]` |
| `"segment_names"` | `list[str]` |
| `"x"`, `"y"`, `"z"` | `list[float]` |
| `"positions"` | `np.ndarray` shape `(1, n, 3)` `float32` |
| `"property-number_of_atoms"` | `int` |
| `"property-box_size"` | `tuple[float, ...]` |

---

### 2. `_get_type_plan_shape_{fmt}` *(mandatory)*

A pure function from request string → output sizing contract. The
planner uses this to compute `estimated_bytes` without reading the file.
The standardiser enforces the contract by raising `ValueError` if a
non-scalar request returns `None` for `bytes_per_match`.

```python
def _get_type_plan_shape_{fmt}(request_string: str
                              ) -> tuple[str, tuple | None, int | None]:
    match request_string:
        case "global_ids":   return "per_atom",        (),    8
        case "atom_names":   return "per_atom",        (),    16
        case "x" | "y" | "z": return "per_atom",       (),    8
        case "positions":    return "per_atom",        (3,),  12
        case "property-number_of_atoms":
            return "scalar_property", (), None
        case _:
            raise ValueError(f"Unsupported request_string for {fmt}: {request_string!r}")
```

**Output kinds:**

| `output_kind` | Meaning | `bytes_per_match` |
|---|---|---|
| `"per_atom"` | one entry per matched atom | int (always > 0) |
| `"per_atom_per_frame"` | one entry per matched atom × frame | int |
| `"scalar_property"` | single scalar regardless of selection | `None` |
| `"selector"` | id list used for cross-domain intersection | `None` |

`trailing_shape` describes the per-element shape beyond the atom-count
axis: `()` for scalars per atom, `(3,)` for 3-vectors per atom.

---

### 3. `_plan_type_query_{fmt}`

Returns a stochastic execution plan without reading the full file.

```python
def _plan_type_query_{fmt}(
    filepath: str | Path,
    query_dictionary: dict,
    request_string: str,
    keywords_available: set,
    requests_available: set,
) -> dict:
```

Use `fph.iter_records_sample` to sample the file, evaluate the predicate
on sampled rows, and emit a raw plan dict. The standardiser in `main.py`
takes care of flattening it into the canonical envelope; you just
provide the raw fields.

**Required raw_plan keys:**

```python
{
    "planner_mode":     "stochastic" | "header",
    "n_atoms":          int,   # estimated or exact
    "n_frames":         1,     # static files always emit 1
    "supported":        bool,
    # plus any sampling-related fields the standardiser will absorb
    # into a `sampling` sub-block automatically (see _PLAN_SAMPLING_KEYS).
}
```

For `property-*` requests that cannot be stochastically estimated, set
`"supported": False` and add a `"reason"` string.

The standardiser will compute `estimated_bytes` from
`n_atoms * n_frames * bytes_per_match`, where `bytes_per_match` comes
from your `_get_type_plan_shape_{fmt}`. Do not compute or echo
`estimated_bytes`, `estimated_mib`, or `bytes_per_atom_per_frame`
yourself — the standardiser ignores raw_plan values for those keys
(see `_PLAN_DROP_KEYS`).

---

### 4. `_get_type_query_{fmt}`

Executes the query and returns the **bare** payload — never a tuple of
`(payload, metadata)`. The framework wraps it in the response envelope.

```python
def _get_type_query_{fmt}(
    filepath: str | Path,
    query_dictionary: dict,
    request_string: str,
    keywords_available: set,
    requests_available: set,
) -> list | int | tuple | np.ndarray:
```

Use `fph.iter_records` in predicate mode, apply the parser's predicate
state, and dispatch on `request_string`.

For `"positions"`, return shape `(1, n, 3)` `float32`. Use
`.reshape(-1, 3)` before adding the frame axis to handle empty matches:

```python
arr = np.array(rows, dtype=np.float32).reshape(-1, 3)
return arr[np.newaxis, :, :]
```

---

### 5. `_update_type_globals_{fmt}`

Extracts global system properties. Called automatically on file load.

```python
def _update_type_globals_{fmt}(filepath: str | Path) -> dict:
```

Return any subset of:

```python
{
    "num_atoms":      int,
    "num_residues":   int,
    "start_box_size": tuple,   # (xmin, xmax, ymin, ymax, zmin, zmax)
}
```

Return `{}` on parse failure — never raise.

---

## Internal Helpers (recommended pattern)

Each parser has a **predicate state** function and an **atom matcher**
function. Both must be pure and shared between the planner (sampled
rows) and the executor (full pass) so semantics stay identical.

```python
def _parse_{fmt}_atom_row(line: str, global_id: int) -> dict:
    return {
        "global_id":   global_id,
        "local_id":    ...,
        "atom_name":   ...,
        "residue_name": ...,
        "residue_id":  ...,
        "segment_name": ...,
        "x": ..., "y": ..., "z": ...,
    }


def _get_{fmt}_type_predicate_state(query_dictionary: dict) -> dict:
    # ALWAYS use qh._normalise_query_pair — never query_dictionary.get(k, default)
    gid_inc, gid_exc = qh._normalise_query_pair(
        query_dictionary.get("global_ids"), range_style=True)
    li_inc,  li_exc  = qh._normalise_query_pair(
        query_dictionary.get("local_ids"),  range_style=True)
    atom_inc, atom_exc = qh._normalise_query_pair(
        query_dictionary.get("atom_name"))
    # ... etc.

    return {
        "gid_inc": gid_inc, "gid_exc": gid_exc,
        "li_inc":  li_inc,  "li_exc":  li_exc,
        "atom_inc": atom_inc, "atom_exc": atom_exc,
        # ...
        # need_* flags use bool() — _normalise_query_pair returns empty
        # () / set() for "no constraint" so a simple truthiness test works.
        "need_gid":  bool(gid_inc  or gid_exc),
        "need_li":   bool(li_inc   or li_exc),
        "need_atom": bool(atom_inc or atom_exc),
        # ...
    }


def _{fmt}_atom_matches_query(atom: dict, predicate_state: dict) -> bool:
    ok = True
    # Test global_ids first — most selective, exact integer match
    if predicate_state["need_gid"]:
        ok = qh._match_range_scalar(atom["global_id"],
                                    predicate_state["gid_inc"],
                                    predicate_state["gid_exc"])
    if ok and predicate_state["need_li"]:
        ok = qh._match_range_scalar(atom["local_id"],
                                    predicate_state["li_inc"],
                                    predicate_state["li_exc"])
    # ... etc.
    return ok
```

**Mandatory rules** — violating these causes silent wrong results:

1. Always extract include/exclude pairs via `qh._normalise_query_pair`.
   Never `query_dictionary.get("k", (default, default))` directly — that
   bypasses the canonical normalisation and breaks `(None, None)`,
   empty-set, and shorthand forms.
2. Every keyword listed in `_get_type_keys_reqs_{fmt}` must have a
   matching slot in the predicate state and a check in the matcher.
   A keyword that exists but isn't filtered on is a silent bug —
   the validator says "yes, this is allowed" while the executor
   returns all atoms.
3. Use `bool(inc or exc)` for `need_*` flags. The normaliser guarantees
   "no constraint" produces empty `()` or `set()`, so truthiness works.
4. Prefer testing integer ID fields (`gid`, `li`, `ri`) before string
   fields. Integer comparisons are cheaper and most selective.
