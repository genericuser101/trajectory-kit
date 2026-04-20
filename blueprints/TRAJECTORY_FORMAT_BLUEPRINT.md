# Adding a New Trajectory Format

A trajectory file provides per-frame positional (and optionally per-atom
property) data. Use `dcd_parse.py` (frame-streaming binary) or
`coor_parse.py` (single-frame static) as the reference implementation.

For the end-to-end workflow including test wiring, see
`ADDING_A_FORMAT.md`. This document specifies the **parser contract**
only.

---

## Registration

In `main.py`, add the new extension to the trajectory domain registry:

```python
"trajectory": {
    "supported_formats": {".dcd", ".coor", ".your_ext"},
    ...
}
```

Nothing else changes. The five function templates auto-resolve to your
new functions by name.

---

## Required Functions (5)

Every trajectory parser implements **five** functions. There is no
special atom-selection function — atom selection flows through the
standard `"global_ids"` request.

### 1. `_get_trajectory_keys_reqs_{fmt}`

Returns queryable keywords and available request strings.

```python
def _get_trajectory_keys_reqs_{fmt}(filepath: str | Path
                                   ) -> tuple[set[str], set[str]]:
```

**Must always include:**

| Category | Value | Style | Purpose |
|---|---|---|---|
| keyword | `"global_ids"` | range | atom index filter from cross-domain intersection |
| keyword | `"frame_interval"` | special | frame range selector |
| request | `"positions"` | — | coordinate extraction |
| request | `"global_ids"` | — | atom selection for cross-domain intersection |

Additional keywords and requests for per-atom trajectory properties
(e.g. `"charge"`, `"occupancy"`) may be added freely for formats that
encode them per-frame.

---

### 2. `_get_trajectory_plan_shape_{fmt}` *(mandatory)*

A pure function from request string → output sizing contract. Trajectory
shapes have an additional frame axis on top of the typing/topology
shapes.

```python
def _get_trajectory_plan_shape_{fmt}(request_string: str
                                    ) -> tuple[str, tuple, int | None]:
    match request_string:
        case "positions":  return "per_atom_per_frame", (3,), 12   # 3 × float32
        case "global_ids": return "selector",           (),   None
        case _:
            raise ValueError(f"Unsupported request_string for {fmt}: {request_string!r}")
```

**Trajectory output kinds:**

| `output_kind` | Meaning | `bytes_per_match` |
|---|---|---|
| `"per_atom_per_frame"` | one entry per atom × frame | int |
| `"per_frame"` | one entry per frame | int |
| `"scalar"` | single scalar regardless of frame range | `None` |
| `"selector"` | id list used for cross-domain intersection | `None` |

The standardiser computes `estimated_bytes` as
`n_atoms * n_frames * bytes_per_match`. For `"selector"` and `"scalar"`,
the caller short-circuits and `estimated_bytes` is omitted from the plan.

---

### 3. `_plan_trajectory_query_{fmt}`

Returns an execution plan without reading frame data. Must be fast —
read the header only.

```python
def _plan_trajectory_query_{fmt}(
    filepath: str | Path,
    query_dictionary: dict,
    request_string: str,
    keywords_available: set,
    requests_available: set,
) -> dict:
```

Return raw_plan with at minimum:

```python
{
    "planner_mode": "header",
    "n_atoms":      int,    # from header
    "n_frames":     int,    # from header (post-frame_interval slicing)
    "supported":    bool,
    # plus any frame-range bookkeeping the standardiser will pass through
    # via the format_specific tier-3 dict.
}
```

The standardiser computes `estimated_bytes` from
`n_atoms * n_frames * bytes_per_match` using your plan_shape function.
Do not echo `estimated_bytes`, `estimated_mib`, or
`bytes_per_atom_per_frame` — they are dropped from raw_plan
(see `_PLAN_DROP_KEYS`).

---

### 4. `_get_trajectory_query_{fmt}`

Executes the query and returns the **bare** payload — never a tuple of
`(array, metadata)`. The framework wraps it in the response envelope.

```python
def _get_trajectory_query_{fmt}(
    filepath: str | Path,
    query_dictionary: dict,
    request_string: str,
    keywords_available: set,
    requests_available: set,
) -> np.ndarray | list | None:
```

**`"global_ids"` request — atom selection for cross-domain intersection**

This is how `fetch()` asks the trajectory domain which atoms match the
`TRAJ_Q` query. The return value drives the intersection logic:

- Return `None` if this format has no per-atom properties to query —
  meaning the trajectory imposes no atom constraint and all atoms pass
  through to the typing/topology intersection. This is the DCD hot
  path.
- Return `list[None | list[int]]` (one entry per selected frame) for
  formats that encode per-atom properties. Each entry is either `None`
  (all atoms pass at that frame) or a sorted `list[int]` of matching
  global ids for that frame. `fetch()` then collapses this to a union
  across frames before intersecting with typing/topology.

```python
case "global_ids":
    # DCD — no per-atom properties, no constraint
    return None

    # Future format with per-frame charges:
    # charge_inc, charge_exc = qh._normalise_query_pair(
    #     query_dictionary.get("charge"), range_style=True)
    # if not charge_inc and not charge_exc:
    #     return None   # no charge constraint — same hot path
    # ... iterate frames, apply predicate, return list[list[int] | None]
```

**`"positions"` request**

The framework injects `query_dictionary["global_ids"]` as
`(selected_list, exclude_set)` before this call. Extract
`global_ids = query_dictionary["global_ids"][0]` and respect
`query_dictionary.get("frame_interval", ())`.

Return `np.ndarray` shape `(n_frames, n_atoms, 3)` `float32`. **Bare
array — no metadata tuple.** The envelope and any frame-range
bookkeeping is built by `main.py` from the plan.

---

### 5. `_update_trajectory_globals_{fmt}`

Extracts global system properties from the header. Must be O(1) — never
iterate frames.

```python
def _update_trajectory_globals_{fmt}(filepath: str | Path) -> dict:
```

Return any subset of:

```python
{
    "num_atoms":  int,
    "num_frames": int,
}
```

Return `{}` on parse failure — never raise.

---

## Frame interval helper

Every trajectory parser resolves frame intervals using
`fph.resolve_frame_interval` from `file_parse_help.py`. Do not roll your
own:

```python
start, stop, step = fph.resolve_frame_interval(
    query_dictionary.get("frame_interval", ())
)
```

This converts the user-facing `(start, stop[, step])` tuple into Python
range semantics `(start_inclusive, stop_exclusive, step)`. The
user-facing stop is always **inclusive** — this is trajectory-kit's API
contract and is consistent across all formats. Conversion to
exclusive happens once, here.

If your format needs a thin local wrapper (e.g. for legacy call sites),
follow the DCD pattern:

```python
def _resolve_{fmt}_frame_interval(frame_inc):
    return fph.resolve_frame_interval(frame_inc)
```

---

## Static-file trajectories

For single-frame static files (e.g. `.coor`, `.rst`), follow the
`coor_parse.py` pattern:

- `_get_trajectory_plan_shape_{fmt}` returns the same kinds as DCD.
- `_plan_trajectory_query_{fmt}` returns `n_frames=1`.
- `_get_trajectory_query_{fmt}` returns `(1, n_atoms, 3)` float32 for
  positions, regardless of `frame_interval`.
- The plan-shape contract is identical to multi-frame formats; the
  static file just always reports `n_frames=1`.

This means `fetch()` and `positions()` work uniformly across static
and trajectory-style files with no special-casing in `main.py`.
