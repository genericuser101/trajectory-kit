# Adding a New Typing Format

A typing file provides per-atom identity and coordinate information — atom names, residue names, segment names, positions, etc. Use `pdb_parse.py` as the reference implementation.

---

## Registration

In `main.py`, add the new extension to the typing domain registry:

```python
"typing": {
    "supported_formats": {".pdb", ".xyz", ".your_ext"},
    ...
}
```

---

## Required Functions (4)

### 1. `_get_type_keys_reqs_{fmt}`

Returns queryable keywords and available request strings for this format.

```python
def _get_type_keys_reqs_{fmt}(filepath: str | Path) -> tuple[set[str], set[str]]:
```

**Keywords** — fields users can filter on in `TYPE_Q`. Should include any column the format exposes per-atom. Standard names to reuse where applicable:

| Keyword | Type | Description |
|---------|------|-------------|
| `"global_ids"` | range | positional atom index (0-based) |
| `"local_ids"` | range | file-internal serial number |
| `"atom_name"` | set | atom name string |
| `"residue_name"` | set | residue name string |
| `"residue_ids"` | range | residue sequence number |
| `"segment_name"` | set | segment/chain identifier |
| `"x"`, `"y"`, `"z"` | range | coordinates in Å |

**Requests** — outputs users can ask for. Must include `"global_ids"` and `"positions"`. Standard names to reuse:

| Request | Returns |
|---------|---------|
| `"global_ids"` | `list[int]` — matching atom indices |
| `"local_ids"` | `list[int]` |
| `"atom_names"` | `list[str]` |
| `"residue_ids"` | `list[int]` |
| `"residue_names"` | `list[str]` |
| `"segment_names"` | `list[str]` |
| `"x"`, `"y"`, `"z"` | `list[float]` |
| `"positions"` | `np.ndarray` shape `(1, n, 3)` float32 |
| `"property-number_of_atoms"` | `int` |
| `"property-box_size"` | `tuple[float, ...]` — `(xmin, xmax, ymin, ymax, zmin, zmax)` |

---

### 2. `_plan_type_query_{fmt}`

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

Use `fph.iter_records_sample` to sample the file, evaluate the predicate on sampled rows, and estimate the payload size. Return a dict with at minimum:

```python
{
    "planner_mode":     "stochastic",
    "file_type":        str,
    "request":          str,
    "supported":        bool,
    "query_dictionary": dict,
    "sampling_metadata": { ... },
    "estimates": {
        "estimated_matches_rounded": int,
        "estimated_output_shape":    tuple,
        "estimated_payload_bytes":   int,
        "estimated_payload_mib":     float,
    },
    "confidence": "none" | "low" | "medium" | "high",
}
```

For `property-*` requests that cannot be stochastically estimated, return `"supported": False` with a `"reason"` key.

---

### 3. `_get_type_query_{fmt}`

Executes the query and returns the requested payload.

```python
def _get_type_query_{fmt}(
    filepath: str | Path,
    query_dictionary: dict,
    request_string: str,
    keywords_available: set,
    requests_available: set,
) -> list | int | tuple | np.ndarray:
```

Use `fph.iter_records` in predicate mode, apply `_get_{fmt}_type_predicate_state` and `_{fmt}_atom_matches_query`, and return the appropriate type per request string.

**For `"positions"`**, return `np.ndarray` of shape `(1, n, 3)` dtype `float32`. Use `.reshape(-1, 3)` before adding the frame axis to handle the empty-match case safely:

```python
arr = np.array(rows, dtype=np.float32).reshape(-1, 3)
return arr[np.newaxis, :, :]
```

---

### 4. `_update_type_globals_{fmt}`

Extracts global system properties. Called automatically on file load.

```python
def _update_type_globals_{fmt}(filepath: str | Path) -> dict:
```

Return a dict with any subset of these keys (omit what the format cannot provide):

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

```python
def _parse_{fmt}_atom_row(line: str, global_id: int) -> dict:
    # parse one record line into a standard atom dict
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
    # precompute include/exclude pairs using qh._normalise_query_pair
    # return a dict of (inc, exc) pairs + boolean need_* flags

def _{fmt}_atom_matches_query(atom: dict, predicate_state: dict) -> bool:
    # apply predicate_state to a single atom dict
    # used by both the exact query and the stochastic planner
```

Use `qh._normalise_query_pair(value, range_style=False/True)` for all include/exclude extraction — never `query_dictionary.get("key", (default, default))` directly.
