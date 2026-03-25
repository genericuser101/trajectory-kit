# Adding a New Trajectory Format

A trajectory file provides per-frame positional (and optionally per-atom property) data. Use `dcd_parse.py` as the reference implementation.

---

## Registration

In `main.py`, add the new extension to the trajectory domain registry:

```python
"trajectory": {
    "supported_formats": {".dcd", ".your_ext"},
    ...
}
```

That is the only change required outside the new parser file. The contract is validated automatically at import time.

---

## Required Functions (4)

Trajectory parsers follow exactly the same four-function contract as typing and topology parsers. There is no special fifth function — atom selection is handled through the standard `"global_ids"` request.

---

### 1. `_get_trajectory_keys_reqs_{fmt}`

Returns queryable keywords and available request strings.

```python
def _get_trajectory_keys_reqs_{fmt}(filepath: str | Path) -> tuple[set[str], set[str]]:
```

**Must always include:**

| Category | Value | Purpose |
|----------|-------|---------|
| keyword | `"global_ids"` | atom index filter injected by the framework |
| keyword | `"frame_interval"` | frame range selector |
| request | `"positions"` | coordinate extraction |
| request | `"global_ids"` | atom selection for cross-domain intersection |

Additional keywords and requests for per-atom trajectory properties (e.g. `"charge"`, `"occupancy"`) may be added freely for formats that encode them per-frame.

---

### 2. `_plan_trajectory_query_{fmt}`

Returns an execution plan without reading frame data. Must be fast — read the header only.

```python
def _plan_trajectory_query_{fmt}(
    filepath: str | Path,
    query_dictionary: dict,
    request_string: str,
    keywords_available: set,
    requests_available: set,
) -> dict:
```

Return a dict containing at minimum:

```python
{
    "planner_mode": "header",
    "file_type":    str,
    "request":      str,
    "supported":    bool,
    "query_dictionary": dict,
    "estimates": {
        "n_atoms_total":           int,
        "n_atoms_selected":        int,
        "n_frames_total":          int,
        "n_frames_selected":       int,
        "start":                   int,
        "stop":                    int,
        "step":                    int,
        "start_inclusive":         True,
        "stop_inclusive":          False,
        "bytes_per_element":       int,
        "estimated_payload_bytes": int,
        "estimated_payload_mib":   float,
    },
}
```

---

### 3. `_get_trajectory_query_{fmt}`

Executes the query and returns the payload.

```python
def _get_trajectory_query_{fmt}(
    filepath: str | Path,
    query_dictionary: dict,
    request_string: str,
    keywords_available: set,
    requests_available: set,
) -> tuple[np.ndarray, dict] | list | None:
```

**`"global_ids"` request — atom selection for cross-domain intersection**

This is how `fetch()` asks the trajectory domain which atoms match the `TRAJ_Q` query. The return value drives the intersection logic:

- Return `None` if this format has no per-atom properties to query — meaning the trajectory imposes no atom constraint and all atoms pass through to the TYPE/TOPO intersection. This is the DCD hotpath.
- Return `list[None | list[int]]` (one entry per selected frame) for formats that encode per-atom properties. Each entry is either `None` (all atoms pass at that frame) or a sorted `list[int]` of matching global_ids for that frame. `fetch()` then collapses this to a union across frames before intersecting with typing/topology.

```python
case "global_ids":
    # DCD — no per-atom properties, no constraint
    return None

    # Future format with per-frame charges:
    # charge_inc, charge_exc = qh._normalise_query_pair(
    #     query_dictionary.get("charge"), range_style=True
    # )
    # if not charge_inc and not charge_exc:
    #     return None   # no charge constraint — same hotpath
    # ... iterate frames, apply predicate, return list[list[int] | None]
```

**`"positions"` request**

The framework always injects `query_dictionary["global_ids"]` as `(selected_list, exclude_set)` before this call. Extract `global_ids = query_dictionary["global_ids"][0]` and respect `query_dictionary.get("frame_interval", ())`.

Return `(positions_array, meta_dict)` where:
- `positions_array` shape `(n_frames, n_atoms, 3)`, dtype `float32`
- `meta_dict` contains at minimum:

```python
{
    "first_frame_read": int | None,
    "last_frame_read":  int | None,
    "n_frames_read":    int,
    "start":            int,
    "stop":             int,
    "step":             int,
    "start_inclusive":  True,
    "stop_inclusive":   False,
}
```

---

### 4. `_update_trajectory_globals_{fmt}`

Extracts global system properties from the header. Must be O(1) — never iterate frames.

```python
def _update_trajectory_globals_{fmt}(filepath: str | Path) -> dict:
```

Return a dict with any subset of:

```python
{
    "num_atoms":  int,
    "num_frames": int,
}
```

Return `{}` on parse failure — never raise.

---

## Frame interval helper

Every trajectory parser resolves frame intervals using the shared `fph.resolve_frame_interval` function from `file_parse_help.py`. You do not need to implement this yourself — just call it:

```python
start, stop, step = fph.resolve_frame_interval(
    query_dictionary.get("frame_interval", ())
)
```

This converts the user-facing `(start, stop[, step])` tuple into Python range semantics `(start_inclusive, stop_exclusive, step)`. The user-facing stop is always inclusive — this is trajectory-kit's API contract and is consistent across all formats.

If your format needs a thin local wrapper (e.g. for legacy call sites), follow the DCD pattern:

```python
def _resolve_{fmt}_frame_interval(frame_inc):
    return fph.resolve_frame_interval(frame_inc)
```
