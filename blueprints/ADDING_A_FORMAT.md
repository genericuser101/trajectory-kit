# Adding a New Format

This is the canonical end-to-end guide for adding a new file format to
trajectory-kit. It assumes you have already read the relevant per-domain
blueprint (`TYPING_FORMAT_BLUEPRINT.md`, `TOPOLOGY_FORMAT_BLUEPRINT.md`,
or `TRAJECTORY_FORMAT_BLUEPRINT.md`).

The workflow has four steps. The cross-correlation matrix in
`tests/integration/test_cross_consistency.py` automates step four — once
you wire your format into the test harness it is automatically validated
against every existing format on every shared canonical field, with no
new assertions to write.

---

## Step 1 — Implement the parser

Each parser module exposes **five** functions following a strict naming
convention. The framework discovers them by template-formatted lookup
against the registry in `main.py`:

| Function template | Returns |
|---|---|
| `_get_{domain}_keys_reqs_{fmt}` | `(set_of_keywords, set_of_requests)` |
| `_get_{domain}_plan_shape_{fmt}` | `(output_kind, trailing_shape, bytes_per_match)` |
| `_plan_{domain}_query_{fmt}` | `dict` — the execution plan |
| `_get_{domain}_query_{fmt}` | `np.ndarray` / `list` / scalar — the payload |
| `_update_{domain}_globals_{fmt}` | `dict` of system-level properties |

`{domain}` is `type`, `topology`, or `trajectory` (note: `type`, not
`typing` — this is the historical naming used in the function templates).
`{fmt}` is the lowercase extension without the dot, e.g. `pdb`, `dcd`.

**Mandatory shared helpers** — never roll your own:

- `qh._normalise_query_pair(value, range_style=False|True)` for every
  include/exclude pair extracted from `query_dictionary`.
- `qh._match(value, inc, exc)` and `qh._match_range_scalar(value, inc, exc)`
  for predicate evaluation.
- `qh._normalise_bonded_with_pair(value)` for topology parsers that
  support `bonded_with`.
- `fph.iter_records(…)` and `fph.iter_records_sample(…)` for streaming
  text formats.
- `fph.resolve_frame_interval(frame_inc)` for trajectory parsers.

The parser must return **bare payloads** (`np.ndarray` / `list` / scalar).
The framework wraps them in the response envelope. Never return tuples
of `(array, metadata)` from the query function — that contract is gone.

---

## Step 2 — Register the format

In `main.py`, add the extension to the appropriate domain's
`supported_formats` set:

```python
"typing": {
    "supported_formats": {".pdb", ".xyz", ".mae", ".your_ext"},
    ...
}
```

Nothing else changes in `main.py`. The function-template lookups
(`keys_fn_template`, `plan_fn_template`, `plan_shape_fn_template`,
`query_fn_template`, `update_fn_template`) automatically resolve to your
new functions because they share the standard naming.

A registry self-check runs at sim construction; it raises `AttributeError`
with a clear message if any of the five functions is missing.

---

## Step 3 — Add a writer to `tests/conftest.py`

The integration test suite uses a **single-source-of-truth** master
dataset (`MASTER_ATOMS`, `MASTER_BONDS`) defined at the top of
`tests/conftest.py`. Every format is written from that master, so all
formats encode the same physical system and can be cross-correlated.

Add **one writer function** that produces a file in your format from the
master:

```python
def _write_your_ext(path: Path) -> None:
    """Write the master 54-atom system to a `.your_ext` file."""
    with path.open("w") as f:
        for atom in MASTER_ATOMS:
            f.write(...)  # whatever your format demands
```

Then add **one fixture** below the existing format fixtures:

```python
@pytest.fixture(scope="session")
def synthetic_your_ext(tmp_path_factory):
    p = tmp_path_factory.mktemp("data") / "system.your_ext"
    _write_your_ext(p)
    return p
```

And add the fixture to the `synthetic_files` mapping if you want it
discoverable by name in test loops.

That is the entire conftest change. ~15 lines.

---

## Step 4 — Wire the format into the cross-correlation matrix

Open `tests/integration/test_cross_consistency.py` and add **one entry**
to the `_build_configs()` function:

```python
def _build_configs(synthetic_pdb, synthetic_xyz, ..., synthetic_your_ext):
    return [
        # existing entries ...
        SimConfig(
            name="your_ext-as-typing",     # human-readable label
            domain="typing",                # which domain this config exposes
            sim_factory=lambda: Sim(typing=synthetic_your_ext),
        ),
    ]
```

Add the new fixture to `_build_configs`'s parameter list and to the
`@pytest.fixture` that calls it.

That is the whole wiring. The matrix harness now automatically:

- Iterates every `(typing-field, config-pair)` combination where both
  configs expose the `typing` domain — comparing your new format's
  output against every other typing format.
- Same for topology and trajectory if your format exposes those domains.
- Catches any divergence as a test failure with a precise diff
  (`format-A returned X, format-B returned Y for field Z`).

You write **zero new assertions**. If your parser is correct, the matrix
goes green. If it's not, the failure points at the exact field where the
new format disagrees with the rest.

---

## What the matrix actually tests

The matrix asserts cross-format equivalence on every canonical field
that multiple formats expose:

- `TYPING_FIELDS` — `global_ids`, `atom_names`, `x`, `y`, `z`,
  `positions`, plus `residue_names` / `residue_ids` / `segment_names`
  for formats that support them.
- `TOPOLOGY_FIELDS` — `global_ids`, `atom_names`, `residue_ids`,
  `residue_names`, `segment_names`, `charges`, `masses`, `atom_types`,
  `vdw_types`.
- `TRAJECTORY_FIELDS` — `positions` at frame 0.
- `FIELD_ALIASES` — semantic equivalences across naming conventions
  (e.g. PSF `atom_types` ≡ MAE `vdw_types`). Add entries here if your
  format uses a different name for an existing concept.
- `local_ids` is intentionally **excluded** from the matrix because it
  is format-specific by design (XYZ is 0-based, PDB/PSF/MAE are 1-based).

---

## Local checklist before opening a PR

1. ✅ Five parser functions implemented with the standard names.
2. ✅ Extension added to `supported_formats` in `main.py`.
3. ✅ Writer + fixture added to `tests/conftest.py`.
4. ✅ One config entry added to `_build_configs()` in
   `test_cross_consistency.py`.
5. ✅ `pytest tests/` passes from the project root with no failures
   and no new skips.
6. ✅ If your format introduces a new field name that is semantically
   equivalent to an existing one, add it to `FIELD_ALIASES`.
7. ✅ If your format exposes `bonded_with`, also confirm
   `tests/integration/test_bonded_with.py` continues to pass — it
   runs against any topology format wired into the conftest.

If steps 1–4 are correct, the rest is automatic.

---

## Why this works

The cross-correlation matrix replaces hand-authored per-format
assertions with a **single specification of what every format must
agree on**. Adding a format becomes a contract-fulfilment exercise
rather than a test-writing exercise:

- The master dataset defines the physical system.
- The writers translate the master into each format's encoding.
- The parsers translate each format back into canonical Python types.
- The matrix asserts the round-trip is lossless on every shared field.

A bug in your parser is detected as a divergence between your format
and at least one other format on at least one canonical field — with no
guesswork about which field, which format, or which value is wrong.

A bug in another parser shows up the same way, surfaced by the new
format. The matrix is symmetric, so adding any format strengthens the
coverage of every existing format.
