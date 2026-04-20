# Adding a New Topology Format

A topology file provides per-atom chemical identity and bonding
information — atom types, charges, masses, bond graphs, etc. Use
`psf_parse.py` as the reference implementation.

For the end-to-end workflow including test wiring, see
`ADDING_A_FORMAT.md`. This document specifies the **parser contract**
only.

---

## Registration

In `main.py`, add the new extension to the topology domain registry:

```python
"topology": {
    "supported_formats": {".psf", ".mae", ".your_ext"},
    ...
}
```

Nothing else changes. The five function templates auto-resolve to your
new functions by name.

---

## Required Functions (5)

Every topology parser implements **five** functions. The fifth — the
plan shape contract — is mandatory and was missing from older versions
of this blueprint.

### 1. `_get_topology_keys_reqs_{fmt}`

Returns queryable keywords and available request strings.

```python
def _get_topology_keys_reqs_{fmt}(filepath: str | Path
                                  ) -> tuple[set[str], set[str]]:
```

**Keywords** — fields users can filter on in `TOPO_Q`. Standard names:

| Keyword | Style | Description |
|---------|-------|-------------|
| `"global_ids"` | range | positional atom index |
| `"local_ids"` | range | file-internal serial number |
| `"atom_name"` | set | atom name string |
| `"atom_type"` | set | force-field atom type string |
| `"residue_name"` | set | residue name string |
| `"residue_ids"` | range | residue sequence number |
| `"segment_name"` | set | segment identifier |
| `"charge"` | range | partial charge |
| `"mass"` | range | atomic mass in Da |
| `"is_virtual"` | set | 0 or 1 |
| `"bonded_with"` | special | bond graph filter — see below |
| `"bonded_with_mode"` | special | `"all"` or `"any"` |

Only include keywords the format actually provides. Drude-specific
fields (`drude_alpha`, `drude_thole`) should be added conditionally
based on file content.

**Requests** — outputs users can ask for. Must include `"global_ids"`.

| Request | Returns |
|---------|---------|
| `"global_ids"` | `list[int]` |
| `"local_ids"` | `list[int]` |
| `"atom_names"` | `list[str]` |
| `"atom_types"` | `list[str]` |
| `"residue_ids"` | `list[int]` |
| `"residue_names"` | `list[str]` |
| `"segment_names"` | `list[str]` |
| `"charges"` | `list[float]` |
| `"masses"` | `list[float]` |
| `"bonds_with"` | `list` — adjacency for matched atoms |
| `"property-system_charge"` | `float` |
| `"property-residue_charge"` | `dict[int, float]` |

---

### 2. `_get_topology_plan_shape_{fmt}` *(mandatory)*

A pure function from request string → output sizing contract.

```python
def _get_topology_plan_shape_{fmt}(request_string: str
                                  ) -> tuple[str, tuple | None, int | None]:
    match request_string:
        case "global_ids":   return "per_atom", (), 8
        case "atom_names":   return "per_atom", (), 16
        case "atom_types":   return "per_atom", (), 16
        case "charges":      return "per_atom", (), 8
        case "masses":       return "per_atom", (), 8
        case "property-system_charge":
            return "scalar_property", (), None
        case _:
            raise ValueError(f"Unsupported request_string for {fmt}: {request_string!r}")
```

See `TYPING_FORMAT_BLUEPRINT.md` for the full output-kind table — same
semantics apply here.

---

### 3. `_plan_topology_query_{fmt}`

Same structure as the typing planner. Return raw_plan with
`planner_mode`, `n_atoms`, `n_frames=1`, `supported`, plus any
sampling-related fields.

For requests that involve `bonded_with`, the planner cannot
stochastically estimate match counts (bond traversal is not sample-able).
Set `"supported": False` with a `"reason"` key when the query
dictionary contains a non-empty `bonded_with` constraint.

---

### 4. `_get_topology_query_{fmt}`

Executes the query and returns the **bare** payload.

```python
def _get_topology_query_{fmt}(
    filepath: str | Path,
    query_dictionary: dict,
    request_string: str,
    keywords_available: set,
    requests_available: set,
    *,
    _bonded_with_depth: int = 0,
    _neighbor_cache: dict | None = None,
) -> list | float:
```

The two underscore-prefixed parameters are **part of the public
contract** for topology parsers but are private to the framework.
They support `bonded_with` recursion (see below). External callers
never set them; the parser passes them through to itself when
recursing.

**Allocate the cache at the top-level entry point:**

```python
if _neighbor_cache is None:
    _neighbor_cache = {}
```

This per-call cache spans all recursive `bonded_with` sub-queries
within a single user-facing call, deduplicating identical neighbour
resolutions.

#### bonded_with — universal application

`bonded_with` filtering applies **uniformly across all per-atom
topology requests**, not just `global_ids`. The reference
implementation in `psf_parse._get_topology_query_psf` does this with a
single closure (`_matched_atoms`) that:

1. Streams the file, applying the predicate state to each atom.
2. If `bonded_with` is present, materialises the matched set, resolves
   neighbour constraints via `_resolve_neighbor_set`, and intersects.
3. Returns the surviving atom set, which downstream switch-cases then
   project onto the requested field.

This means there is **one** filtering pipeline for all requests, not
one per request. Adding a new request is a one-line case in the
dispatch switch — no bond-filtering boilerplate to duplicate.

#### bonded_with — accepted shorthand forms

`qh._normalise_bonded_with_pair` accepts four user-input shapes,
normalising them all into the canonical `(include_blocks, exclude_blocks)`
pair:

| User input | Meaning |
|---|---|
| `None` | no constraint |
| `dict` | single include block |
| `list[dict]` | multiple include blocks (additive) |
| `(inc, exc)` tuple | full include / exclude pair |

Inside each block, neighbour sub-queries are themselves topology
queries (recursive). The dict accepts:

```python
{
    "atom_name":   "CA",         # any normaliser-accepted form
    "residue_ids": (1, 5),       # range-style
    "count":       (2, "ge"),    # bond degree comparator: (n, op)
    # comparators: "eq", "ne", "ge", "le", "gt", "lt"
}
```

#### bonded_with — recursion and depth limit

Nested neighbour sub-queries are allowed. The framework guarantees:

- `MAX_BONDED_WITH_DEPTH = 16` (defined in `_query_help`).
- A `RecursionError` is raised when the depth would exceed this.
- The neighbour cache (`_neighbor_cache`) deduplicates identical
  sub-queries within a single user call. Use `qh._freeze_query` to
  produce hashable cache keys.

Your topology parser does **not** need to re-implement any of this —
follow the `_resolve_neighbor_set` pattern in `psf_parse.py`. The
helpers do the heavy lifting; your parser just plumbs the depth
counter and cache through recursive calls.

---

### 5. `_update_topology_globals_{fmt}`

Extracts global system properties.

```python
def _update_topology_globals_{fmt}(filepath: str | Path) -> dict:
```

Return any subset of:

```python
{
    "num_atoms":    int,
    "num_residues": int,
}
```

Return `{}` on parse failure — never raise.

---

## Internal Helpers (recommended pattern)

```python
def _parse_{fmt}_atom_row(line: str, global_id: int) -> dict:
    return {
        "global_id":    global_id,
        "local_id":     ...,
        "segment_name": ...,
        "residue_id":   ...,
        "residue_name": ...,
        "atom_name":    ...,
        "atom_type":    ...,
        "charge":       ...,
        "mass":         ...,
        "is_virtual":   ...,
    }


def _get_{fmt}_topology_predicate_state(query_dictionary: dict,
                                         keywords_available: set) -> dict:
    gid_inc, gid_exc = qh._normalise_query_pair(
        query_dictionary.get("global_ids"), range_style=True)
    li_inc,  li_exc  = qh._normalise_query_pair(
        query_dictionary.get("local_ids"),  range_style=True)
    atom_inc, atom_exc = qh._normalise_query_pair(
        query_dictionary.get("atom_name"))
    charge_inc, charge_exc = qh._normalise_query_pair(
        query_dictionary.get("charge"), range_style=True)
    # ... etc.

    return {
        "gid_inc": gid_inc, "gid_exc": gid_exc,
        "li_inc":  li_inc,  "li_exc":  li_exc,
        "atom_inc": atom_inc, "atom_exc": atom_exc,
        "charge_inc": charge_inc, "charge_exc": charge_exc,
        # ...
        "need_gid":    bool(gid_inc    or gid_exc),
        "need_li":     bool(li_inc     or li_exc),
        "need_atom":   bool(atom_inc   or atom_exc),
        "need_charge": bool(charge_inc or charge_exc),
        # ...
    }


def _{fmt}_atom_matches_query(atom: dict, predicate_state: dict) -> bool:
    ok = True
    # Most selective first: integer-id checks bail out early
    if predicate_state["need_gid"]:
        ok = qh._match_range_scalar(atom["global_id"],
                                    predicate_state["gid_inc"],
                                    predicate_state["gid_exc"])
    if ok and predicate_state["need_li"]:
        ok = qh._match_range_scalar(atom["local_id"],
                                    predicate_state["li_inc"],
                                    predicate_state["li_exc"])
    # ... string + range checks for the remaining fields
    return ok
```

**Mandatory rules** — same as typing, plus:

1. Always extract include/exclude pairs via `qh._normalise_query_pair`.
2. Every declared keyword must have a matching slot and matcher check.
3. Use `bool(inc or exc)` for `need_*` flags.
4. Test integer IDs (`gid`, `li`, `ri`) before strings.
5. **bonded_with applies uniformly** to all per-atom requests via the
   `_matched_atoms` closure pattern. Do not implement per-request
   bonded_with filtering — that is a known anti-pattern; see the
   `_get_topology_query_psf` reference for the universal approach.
6. **Pass `_bonded_with_depth` and `_neighbor_cache` through recursion.**
   Never strip them when forwarding to a sub-query. The depth check
   and cache lookup happen at the top of `_resolve_neighbor_set`.
