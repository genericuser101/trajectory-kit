# Adding a New Topology Format

A topology file provides per-atom chemical identity and bonding information — atom types, charges, masses, bond graphs, etc. Use `psf_parse.py` as the reference implementation.

---

## Registration

In `main.py`, add the new extension to the topology domain registry:

```python
"topology": {
    "supported_formats": {".psf", ".your_ext"},
    ...
}
```

---

## Required Functions (4)

### 1. `_get_topology_keys_reqs_{fmt}`

Returns queryable keywords and available request strings for this format.

```python
def _get_topology_keys_reqs_{fmt}(filepath: str | Path) -> tuple[set[str], set[str]]:
```

**Keywords** — fields users can filter on in `TOPO_Q`. Standard names to reuse:

| Keyword | Type | Description |
|---------|------|-------------|
| `"global_ids"` | range | positional atom index |
| `"local_ids"` | range | file-internal serial number |
| `"atom_name"` | set | atom name string |
| `"atom_type"` | set | force-field atom type string |
| `"residue_name"` | set | residue name string |
| `"residue_ids"` | range | residue sequence number |
| `"segment_name"` | set | segment identifier |
| `"charge"` | range | partial charge |
| `"mass"` | range | atomic mass in Da |
| `"is_virtual"` | range | 0 or 1 |
| `"bonded_with"` | special | bond graph filter — see PSF reference |
| `"bonded_with_mode"` | special | `"all"` or `"any"` |

Only include keywords the format actually provides. Drude-specific fields (`drude_alpha`, `drude_thole`) should be added conditionally when the format/file signals their presence.

**Requests** — outputs users can ask for. Must include `"global_ids"`. Standard names:

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
| `"bonds_with"` | `list` — bond adjacency (may be `NotImplementedError`) |
| `"property-system_charge"` | `float` |
| `"property-residue_charge"` | `dict` |

---

### 2. `_plan_topology_query_{fmt}`

Returns a stochastic execution plan without reading the full file.

```python
def _plan_topology_query_{fmt}(
    filepath: str | Path,
    query_dictionary: dict,
    request_string: str,
    keywords_available: set,
    requests_available: set,
) -> dict:
```

Same structure as the typing planner. For topology, bonded_with constraints cannot be stochastically estimated — return `"supported": False` with `"reason"` when `query_dictionary` contains `"bonded_with"` with non-empty include/exclude lists.

---

### 3. `_get_topology_query_{fmt}`

Executes the query and returns the requested payload.

```python
def _get_topology_query_{fmt}(
    filepath: str | Path,
    query_dictionary: dict,
    request_string: str,
    keywords_available: set,
    requests_available: set,
) -> list | float:
```

Use `fph.iter_records` in counted mode (find the atom count header, iterate exactly that many lines). Apply `_get_{fmt}_topology_predicate_state` and `_{fmt}_atom_matches_query`.

For `"global_ids"`, apply bonded_with post-filtering after the predicate pass if `query_dictionary` contains `"bonded_with"`. See `_filter_by_bonded_with` in `psf_parse.py` for the reference implementation of bond graph traversal.

---

### 4. `_update_topology_globals_{fmt}`

Extracts global system properties. Called automatically on file load.

```python
def _update_topology_globals_{fmt}(filepath: str | Path) -> dict:
```

Return a dict with any subset of:

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
    # precompute include/exclude pairs using qh._normalise_query_pair
    # include boolean need_* flags

def _{fmt}_atom_matches_query(atom: dict, predicate_state: dict) -> bool:
    # apply predicate to a single atom dict
```

Use `qh._normalise_query_pair(value, range_style=False/True)` for all include/exclude extraction.
