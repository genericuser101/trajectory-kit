"""
tests/test_pdb_parse.py
=======================
Unit and integration tests for trajectory_kit.pdb_parse.

Two layers covered here:

1. Pure (file-free) row parser and plan-shape classifier
   ----------------------------------------------------
   - _parse_pdb_atom_row
   - _get_type_plan_shape_pdb (typing plan shape)

2. CONECT-based topology parsing (file-backed)
   ----------------------------------------------------
   - Discovery probe (_has_conect_records_pdb)
   - CONECT presence-gated keyword/request advertisement
     (_get_topology_keys_reqs_pdb)
   - Topology globals delegate to typing globals
     (_update_topology_globals_pdb)
   - Topology plan-shape classifier (_get_topology_plan_shape_pdb)
   - Local serial -> global id map (_build_pdb_local_to_global_map)
   - bonded_with filter end-to-end (_get_topology_query_pdb)
   - All three bonded_with shorthand forms agree
   - Bond filter applies uniformly across every per-atom request type
   - Neighbour sub-queries (Level-2 bonded_with)
   - bonded_with_mode "any" / "all" semantics
   - Behaviour when CONECT records are absent

Pure-function tests at the top need no fixtures. Topology tests build
their own CONECT-enabled PDB fixtures from the master atom/bond list in
``conftest.py``, since the standard ``synthetic_pdb`` fixture writes
ATOM records only. Every assertion is derived from the master, never
hand-authored.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import pytest

from trajectory_kit import pdb_parse as pp
from trajectory_kit import _query_help as qh
from trajectory_kit.pdb_parse import (
    _parse_pdb_atom_row,
    _get_type_plan_shape_pdb,
)

from conftest import (
    MASTER_ATOMS, MASTER_BONDS, N_ATOMS, DEGREE_MAP,
    LIGAND_IDS, SOLV_IDS, IONS_IDS,
    OT_TYPE_IDS, HT_TYPE_IDS, CA_TYPE_IDS,
)


# ===========================================================================
# Pure (file-free) tests — _parse_pdb_atom_row
# ===========================================================================

class TestParsePDBAtomRow:

    def _row(self, serial=1, name="C1  ", resn="DRG", chain="L", resi=1,
             x=0.0, y=0.0, z=0.0, occ=1.0, temp=0.1, seg="LIG "):
        return (f"ATOM  {serial:5d} {name:<4s} {resn:<4s}{chain}{resi:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}{occ:6.2f}{temp:6.2f}      {seg:<4s}\n")

    def test_returns_dict(self):
        r = _parse_pdb_atom_row(self._row(), 0)
        assert isinstance(r, dict)

    def test_global_id_from_index(self):
        assert _parse_pdb_atom_row(self._row(), 7)["global_id"] == 7

    def test_atom_name_stripped(self):
        r = _parse_pdb_atom_row(self._row(name="OH2 "), 0)
        assert r["atom_name"] == "OH2"

    def test_residue_name(self):
        r = _parse_pdb_atom_row(self._row(resn="TIP"), 0)
        assert r["residue_name"] == "TIP"

    def test_segment_name(self):
        r = _parse_pdb_atom_row(self._row(seg="SOLV"), 0)
        assert r["segment_name"] == "SOLV"

    def test_coordinates(self):
        r = _parse_pdb_atom_row(self._row(x=1.5, y=-2.0, z=3.25), 0)
        assert abs(r["x"] - 1.5)   < 1e-3
        assert abs(r["y"] - -2.0)  < 1e-3
        assert abs(r["z"] - 3.25)  < 1e-3

    def test_occupancy(self):
        r = _parse_pdb_atom_row(self._row(occ=0.50), 0)
        assert abs(r["occupancy"] - 0.50) < 1e-3

    def test_temperature_coeff(self):
        r = _parse_pdb_atom_row(self._row(temp=0.75), 0)
        assert abs(r["temperature_coeff"] - 0.75) < 1e-3


# ===========================================================================
# Pure (file-free) tests — _get_type_plan_shape_pdb
# ===========================================================================

class TestPDBTypePlanShape:
    """Plan-shape classifier for the typing layer (per-atom rows from
    ATOM/HETATM). The topology plan-shape classifier is tested separately
    in TestGetTopologyPlanShapePDB further down."""

    def test_global_ids(self):
        kind, _, _ = _get_type_plan_shape_pdb("global_ids")
        assert kind == "per_atom"

    def test_positions(self):
        kind, shape, bpe = _get_type_plan_shape_pdb("positions")
        assert kind == "per_atom"
        assert shape == (3,)
        assert bpe == 12

    def test_number_of_atoms_scalar(self):
        kind, _, bpe = _get_type_plan_shape_pdb("property-number_of_atoms")
        assert kind == "scalar_property"
        assert bpe is None

    def test_box_size_scalar(self):
        kind, shape, bpe = _get_type_plan_shape_pdb("property-box_size")
        assert kind == "scalar_property"
        assert bpe is None

    def test_unknown_raises(self):
        with pytest.raises(ValueError):
            _get_type_plan_shape_pdb("bad_request")


# ===========================================================================
# Local fixtures — CONECT-enabled and CONECT-less PDB files
# ===========================================================================

def _write_pdb_atoms(lines, atoms):
    """Append ATOM rows to ``lines`` matching the conftest writer's format."""
    for a in atoms:
        serial = a["id"] + 1
        name   = a["name"]
        name_col = f" {name:<3s}" if len(name) < 4 else name
        lines.append(
            f"ATOM  {serial:5d} {name_col:4s} {a['resn']:<4s}{a['seg'][0]}{a['resi']:4d}    "
            f"{a['x']:8.3f}{a['y']:8.3f}{a['z']:8.3f}{a['occ']:6.2f}{a['temp']:6.2f}"
            f"      {a['seg']:<4s}\n"
        )


def _write_conect_records(lines, bonds, bidirectional: bool):
    """
    Append CONECT records grouped by source serial (max 4 partners per line).

    bidirectional=False: each bond appears once (loose convention; many
    real-world PDB files do this).
    bidirectional=True:  each bond appears twice (strict spec).
    """
    adj = defaultdict(list)
    for (i, j) in bonds:
        adj[i].append(j)
        if bidirectional:
            adj[j].append(i)

    for src in sorted(adj.keys()):
        partners = adj[src]
        for k in range(0, len(partners), 4):
            chunk = partners[k:k + 4]
            tokens = [f"CONECT{src:5d}"]
            for p in chunk:
                tokens.append(f"{p:5d}")
            lines.append("".join(tokens) + "\n")


@pytest.fixture(scope="session")
def pdb_with_conect(tmp_path_factory) -> Path:
    """PDB with all atoms + every bond as a CONECT record (single direction)."""
    p = tmp_path_factory.mktemp("pdb_conect") / "with_conect.pdb"
    lines = ["REMARK CONECT-enabled synthetic PDB\n"]
    _write_pdb_atoms(lines, MASTER_ATOMS)
    _write_conect_records(lines, MASTER_BONDS, bidirectional=False)
    lines.append("END\n")
    p.write_text("".join(lines))
    return p


@pytest.fixture(scope="session")
def pdb_with_conect_bidi(tmp_path_factory) -> Path:
    """PDB where every CONECT bond appears twice (strict spec layout)."""
    p = tmp_path_factory.mktemp("pdb_conect_bidi") / "with_conect_bidi.pdb"
    lines = ["REMARK CONECT-enabled (bidirectional) synthetic PDB\n"]
    _write_pdb_atoms(lines, MASTER_ATOMS)
    _write_conect_records(lines, MASTER_BONDS, bidirectional=True)
    lines.append("END\n")
    p.write_text("".join(lines))
    return p


@pytest.fixture(scope="session")
def pdb_no_conect(tmp_path_factory) -> Path:
    """PDB with ATOM records only — no CONECT section at all."""
    p = tmp_path_factory.mktemp("pdb_no_conect") / "no_conect.pdb"
    lines = ["REMARK no-CONECT synthetic PDB\n"]
    _write_pdb_atoms(lines, MASTER_ATOMS)
    lines.append("END\n")
    p.write_text("".join(lines))
    return p


# ===========================================================================
# Expected values derived from the master system
# ===========================================================================

TERMINAL_IDS = sorted(i for i, d in DEGREE_MAP.items() if d == 1)
ISOLATED_IDS = sorted(i for i, d in DEGREE_MAP.items() if d == 0)
DEGREE_2_IDS = sorted(i for i, d in DEGREE_MAP.items() if d == 2)


def _bonded_to_set_ids(target_ids):
    """Atoms bonded to at least one atom whose global_id is in target_ids."""
    target = set(target_ids)
    result = set()
    for (li, lj) in MASTER_BONDS:
        gi, gj = li - 1, lj - 1
        if gi in target:
            result.add(gj)
        if gj in target:
            result.add(gi)
    return sorted(result)


BONDED_TO_OT_IDS = _bonded_to_set_ids(OT_TYPE_IDS)
BONDED_TO_LIG_IDS = _bonded_to_set_ids(LIGAND_IDS)


# ===========================================================================
# Discovery probe — _has_conect_records_pdb
# ===========================================================================

class TestHasConectRecordsPDB:

    def test_returns_true_when_present(self, pdb_with_conect):
        assert pp._pdb_has_connect_records(pdb_with_conect) is True

    def test_returns_false_when_absent(self, pdb_no_conect):
        assert pp._pdb_has_connect_records(pdb_no_conect) is False

    def test_bidirectional_conect_also_detected(self, pdb_with_conect_bidi):
        assert pp._pdb_has_connect_records(pdb_with_conect_bidi) is True

    def test_returns_true_on_first_match(self, tmp_path):
        # File where the very first non-header line is CONECT — probe
        # should not need to read further.
        p = tmp_path / "early_conect.pdb"
        p.write_text("REMARK x\nCONECT    1    2\nEND\n")
        assert pp._pdb_has_connect_records(p) is True


# ===========================================================================
# Keyword/request advertisement gated on CONECT presence
# ===========================================================================

class TestGetTopologyKeysReqsPDB:

    def test_with_conect_advertises_bonded_with(self, pdb_with_conect):
        keys, reqs = pp._get_topology_keys_reqs_pdb(pdb_with_conect)
        assert "bonded_with" in keys
        assert "bonded_with_mode" in keys
        assert "bonds_with" in reqs

    def test_without_conect_omits_bonded_with(self, pdb_no_conect):
        keys, reqs = pp._get_topology_keys_reqs_pdb(pdb_no_conect)
        assert "bonded_with" not in keys
        assert "bonded_with_mode" not in keys
        assert "bonds_with" not in reqs

    def test_topology_keys_are_superset_of_typing_keys(self, pdb_with_conect):
        type_keys, _ = pp._get_type_keys_reqs_pdb(pdb_with_conect)
        topo_keys, _ = pp._get_topology_keys_reqs_pdb(pdb_with_conect)
        assert type_keys.issubset(topo_keys)

    def test_topology_reqs_are_superset_of_typing_reqs(self, pdb_with_conect):
        _, type_reqs = pp._get_type_keys_reqs_pdb(pdb_with_conect)
        _, topo_reqs = pp._get_topology_keys_reqs_pdb(pdb_with_conect)
        assert type_reqs.issubset(topo_reqs)

    def test_no_conect_topology_equals_typing(self, pdb_no_conect):
        # Without CONECT, the topology surface is exactly the typing
        # surface — no new keywords or requests.
        type_keys, type_reqs = pp._get_type_keys_reqs_pdb(pdb_no_conect)
        topo_keys, topo_reqs = pp._get_topology_keys_reqs_pdb(pdb_no_conect)
        assert topo_keys == type_keys
        assert topo_reqs == type_reqs


# ===========================================================================
# Topology globals delegate to typing globals
# ===========================================================================

class TestUpdateTopologyGlobalsPDB:

    def test_returns_dict(self, pdb_with_conect):
        result = pp._update_topology_globals_pdb(pdb_with_conect)
        assert isinstance(result, dict)

    def test_matches_type_globals(self, pdb_with_conect):
        topo_g = pp._update_topology_globals_pdb(pdb_with_conect)
        type_g = pp._update_type_globals_pdb(pdb_with_conect)
        assert topo_g == type_g

    def test_num_atoms_correct(self, pdb_with_conect):
        result = pp._update_topology_globals_pdb(pdb_with_conect)
        assert result["num_atoms"] == N_ATOMS

    def test_works_without_conect(self, pdb_no_conect):
        # CONECT presence is irrelevant for atom-level globals.
        result = pp._update_topology_globals_pdb(pdb_no_conect)
        assert result["num_atoms"] == N_ATOMS


# ===========================================================================
# Plan-shape classifier
# ===========================================================================

class TestGetTopologyPlanShapePDB:

    def test_global_ids_per_atom(self):
        kind, shape, bpm = pp._get_topology_plan_shape_pdb("global_ids")
        assert kind == "per_atom"

    def test_positions_per_atom_with_trailing_3(self):
        kind, shape, bpm = pp._get_topology_plan_shape_pdb("positions")
        assert kind == "per_atom"
        assert shape == (3,)

    def test_property_number_of_atoms_scalar(self):
        kind, _, _ = pp._get_topology_plan_shape_pdb("property-number_of_atoms")
        assert kind == "scalar_property"

    def test_property_box_size_scalar(self):
        kind, _, _ = pp._get_topology_plan_shape_pdb("property-box_size")
        assert kind == "scalar_property"

    def test_bonds_with_unsupported_complex(self):
        kind, _, _ = pp._get_topology_plan_shape_pdb("bonds_with")
        assert kind == "unsupported_complex"

    def test_unknown_request_raises(self):
        with pytest.raises(ValueError):
            pp._get_topology_plan_shape_pdb("nonsense_request")


# ===========================================================================
# Local serial -> global id map
# ===========================================================================

class TestBuildPDBLocalToGlobalMap:

    def test_natom_correct(self, pdb_with_conect):
        local_to_global = pp._build_pdb_local_to_global_map(pdb_with_conect)
        assert len(local_to_global) == N_ATOMS

    def test_serial_one_maps_to_global_zero(self, pdb_with_conect):
        local_to_global = pp._build_pdb_local_to_global_map(pdb_with_conect)
        assert local_to_global[1] == 0

    def test_last_serial_maps_to_last_global(self, pdb_with_conect):
        local_to_global = pp._build_pdb_local_to_global_map(pdb_with_conect)
        assert local_to_global[N_ATOMS] == N_ATOMS - 1

    def test_all_serials_present(self, pdb_with_conect):
        local_to_global = pp._build_pdb_local_to_global_map(pdb_with_conect)
        # Master atoms have serials 1..N_ATOMS
        for a in MASTER_ATOMS:
            assert local_to_global[a["id"] + 1] == a["id"]


# ===========================================================================
# CONECT line parser
# ===========================================================================

class TestParsePDBConnectLine:

    def test_parses_source_and_destinations(self):
        src, dst = pp._parse_pdb_connect_line("CONECT    1    2    3    4\n")
        assert src == 1
        assert dst == [2, 3, 4]

    def test_accepts_connect_spelling(self):
        src, dst = pp._parse_pdb_connect_line("CONNECT   10   20   30\n")
        assert src == 10
        assert dst == [20, 30]

    def test_accepts_record_with_no_destinations(self):
        src, dst = pp._parse_pdb_connect_line("CONECT    1\n")
        assert src == 1
        assert dst == []

    def test_raises_on_missing_source(self):
        with pytest.raises(ValueError, match="Malformed PDB CONECT record"):
            pp._parse_pdb_connect_line("CONECT\n")

    def test_raises_on_non_integer_source(self):
        with pytest.raises(ValueError, match="Malformed PDB CONECT record"):
            pp._parse_pdb_connect_line("CONECT    X    2\n")

    def test_raises_on_non_integer_destination(self):
        with pytest.raises(ValueError, match="Malformed PDB CONECT record"):
            pp._parse_pdb_connect_line("CONECT    1    2    X\n")


# ===========================================================================
# CONECT adjacency builder
# ===========================================================================

class TestBuildPDBConnectAdjacency:

    def test_returns_master_adjacency(self, pdb_with_conect):
        adjacency = pp._build_pdb_connect_adjacency(pdb_with_conect)

        expected: list[set[int]] = [set() for _ in range(len(MASTER_ATOMS))]
        for a_local, b_local in MASTER_BONDS:
            a_global = a_local - 1
            b_global = b_local - 1
            expected[a_global].add(b_global)
            expected[b_global].add(a_global)

        assert adjacency == expected

    def test_dedupes_bidirectional_records(self, pdb_with_conect_bidi):
        adjacency = pp._build_pdb_connect_adjacency(pdb_with_conect_bidi)

        got = {
            (min(i, j), max(i, j))
            for i, neighbours in enumerate(adjacency)
            for j in neighbours
        }
        expected = {
            (min(a - 1, b - 1), max(a - 1, b - 1))
            for a, b in MASTER_BONDS
        }

        assert got == expected
        assert len(got) == len(MASTER_BONDS)

    def test_no_conect_raises(self, pdb_no_conect):
        with pytest.raises(ValueError, match="No PDB CONECT records found"):
            pp._build_pdb_connect_adjacency(pdb_no_conect)

    def test_self_loop_skipped(self, tmp_path):
        p = tmp_path / "self_loop.pdb"
        p.write_text(
            "ATOM      1  X    LIG   1       0.000   0.000   0.000  1.00  0.00      LIG \n"
            "CONECT    1    1\n"
            "END\n"
        )

        adjacency = pp._build_pdb_connect_adjacency(p)

        assert adjacency == [set()]

    def test_nonexistent_serial_does_not_crash(self, tmp_path):
        p = tmp_path / "missing_serial.pdb"
        lines = ["REMARK missing-serial CONECT test\n"]
        _write_pdb_atoms(lines, MASTER_ATOMS[:2])
        lines.append("CONECT    1    2 9999\n")
        lines.append("END\n")
        p.write_text("".join(lines))

        adjacency = pp._build_pdb_connect_adjacency(p)

        assert adjacency == [{1}, {0}]

    def test_missing_source_serial_is_skipped(self, tmp_path):
        p = tmp_path / "missing_source_serial.pdb"
        lines = ["REMARK missing-source CONECT test\n"]
        _write_pdb_atoms(lines, MASTER_ATOMS[:2])
        lines.append("CONECT 9999    1    2\n")
        lines.append("CONECT    1    2\n")
        lines.append("END\n")
        p.write_text("".join(lines))

        adjacency = pp._build_pdb_connect_adjacency(p)

        assert adjacency == [{1}, {0}]


    def test_empty_atom_map_returns_empty_without_requiring_conect(self, tmp_path):
        p = tmp_path / "empty.pdb"
        p.write_text("REMARK no atoms\nEND\n")

        adjacency = pp._build_pdb_connect_adjacency(p)

        assert adjacency == []

# ===========================================================================
# Topology query — no bond filter (regression: typing path still works)
# ===========================================================================

class TestPDBTopologyQueryNoBondFilter:
    """Without bonded_with, every typing field must still be queryable
    through _get_topology_query_pdb — topology is a strict superset."""

    def test_global_ids_returns_all(self, pdb_with_conect):
        keys, reqs = pp._get_topology_keys_reqs_pdb(pdb_with_conect)
        ids = pp._get_topology_query_pdb(pdb_with_conect, {}, "global_ids", keys, reqs)
        assert len(ids) == N_ATOMS
        assert sorted(ids) == list(range(N_ATOMS))

    def test_atom_name_filter(self, pdb_with_conect):
        keys, reqs = pp._get_topology_keys_reqs_pdb(pdb_with_conect)
        ids = pp._get_topology_query_pdb(
            pdb_with_conect, {"atom_name": "OH2"}, "global_ids", keys, reqs
        )
        expected = [a["id"] for a in MASTER_ATOMS if a["name"] == "OH2"]
        assert sorted(ids) == sorted(expected)

    def test_segment_name_filter(self, pdb_with_conect):
        keys, reqs = pp._get_topology_keys_reqs_pdb(pdb_with_conect)
        ids = pp._get_topology_query_pdb(
            pdb_with_conect, {"segment_name": "LIG"}, "global_ids", keys, reqs
        )
        assert sorted(ids) == LIGAND_IDS

    def test_property_number_of_atoms(self, pdb_with_conect):
        keys, reqs = pp._get_topology_keys_reqs_pdb(pdb_with_conect)
        n = pp._get_topology_query_pdb(
            pdb_with_conect, {}, "property-number_of_atoms", keys, reqs
        )
        assert n == N_ATOMS

    def test_positions_shape(self, pdb_with_conect):
        keys, reqs = pp._get_topology_keys_reqs_pdb(pdb_with_conect)
        pos = pp._get_topology_query_pdb(
            pdb_with_conect, {}, "positions", keys, reqs
        )
        assert pos.shape == (1, N_ATOMS, 3)
        assert pos.dtype == np.float32

    def test_unsupported_request_raises(self, pdb_with_conect):
        keys, reqs = pp._get_topology_keys_reqs_pdb(pdb_with_conect)
        with pytest.raises(ValueError):
            pp._get_topology_query_pdb(
                pdb_with_conect, {}, "totally_made_up_request", keys, reqs
            )

    def test_bonds_with_request_raises_not_implemented(self, pdb_with_conect):
        keys, reqs = pp._get_topology_keys_reqs_pdb(pdb_with_conect)
        with pytest.raises(NotImplementedError):
            pp._get_topology_query_pdb(
                pdb_with_conect, {}, "bonds_with", keys, reqs
            )


# ===========================================================================
# bonded_with — basic correctness
# ===========================================================================

class TestPDBBondedWith:

    def test_terminal_atoms(self, pdb_with_conect):
        keys, reqs = pp._get_topology_keys_reqs_pdb(pdb_with_conect)
        ids = pp._get_topology_query_pdb(
            pdb_with_conect,
            {"bonded_with": {"total": True, "count": {"eq": 1}}},
            "global_ids", keys, reqs,
        )
        assert sorted(ids) == TERMINAL_IDS

    def test_isolated_atoms_match_ions(self, pdb_with_conect):
        # Isolated atoms in the master are exactly the four ions.
        keys, reqs = pp._get_topology_keys_reqs_pdb(pdb_with_conect)
        ids = pp._get_topology_query_pdb(
            pdb_with_conect,
            {"bonded_with": {"total": True, "count": {"eq": 0}}},
            "global_ids", keys, reqs,
        )
        assert sorted(ids) == ISOLATED_IDS
        assert sorted(ids) == sorted(IONS_IDS)

    def test_degree_two_atoms(self, pdb_with_conect):
        keys, reqs = pp._get_topology_keys_reqs_pdb(pdb_with_conect)
        ids = pp._get_topology_query_pdb(
            pdb_with_conect,
            {"bonded_with": {"total": True, "count": {"eq": 2}}},
            "global_ids", keys, reqs,
        )
        assert sorted(ids) == DEGREE_2_IDS

    def test_ge_comparator(self, pdb_with_conect):
        # Atoms with at least 1 bond = everything except the four ions.
        keys, reqs = pp._get_topology_keys_reqs_pdb(pdb_with_conect)
        ids = pp._get_topology_query_pdb(
            pdb_with_conect,
            {"bonded_with": {"total": True, "count": {"ge": 1}}},
            "global_ids", keys, reqs,
        )
        expected = sorted(i for i, d in DEGREE_MAP.items() if d >= 1)
        assert sorted(ids) == expected

    def test_explicit_none_bonded_with(self, pdb_with_conect):
        # Explicit None should be treated as no bond filter.
        keys, reqs = pp._get_topology_keys_reqs_pdb(pdb_with_conect)
        ids = pp._get_topology_query_pdb(
            pdb_with_conect,
            {"bonded_with": None},
            "global_ids", keys, reqs,
        )
        assert len(ids) == N_ATOMS


# ===========================================================================
# bonded_with — three shorthand forms must agree
# ===========================================================================

class TestPDBBondedWithShorthand:
    """Mirrors the PSF shorthand-equivalence tests."""

    BLOCK = {"total": True, "count": {"eq": 1}}

    def test_bare_dict_form(self, pdb_with_conect):
        keys, reqs = pp._get_topology_keys_reqs_pdb(pdb_with_conect)
        ids = pp._get_topology_query_pdb(
            pdb_with_conect, {"bonded_with": self.BLOCK}, "global_ids", keys, reqs
        )
        assert sorted(ids) == TERMINAL_IDS

    def test_bare_list_form(self, pdb_with_conect):
        keys, reqs = pp._get_topology_keys_reqs_pdb(pdb_with_conect)
        ids = pp._get_topology_query_pdb(
            pdb_with_conect, {"bonded_with": [self.BLOCK]}, "global_ids", keys, reqs
        )
        assert sorted(ids) == TERMINAL_IDS

    def test_full_tuple_form(self, pdb_with_conect):
        keys, reqs = pp._get_topology_keys_reqs_pdb(pdb_with_conect)
        ids = pp._get_topology_query_pdb(
            pdb_with_conect, {"bonded_with": ([self.BLOCK], [])}, "global_ids", keys, reqs
        )
        assert sorted(ids) == TERMINAL_IDS

    def test_all_three_forms_agree(self, pdb_with_conect):
        keys, reqs = pp._get_topology_keys_reqs_pdb(pdb_with_conect)
        a = pp._get_topology_query_pdb(
            pdb_with_conect, {"bonded_with": self.BLOCK}, "global_ids", keys, reqs
        )
        b = pp._get_topology_query_pdb(
            pdb_with_conect, {"bonded_with": [self.BLOCK]}, "global_ids", keys, reqs
        )
        c = pp._get_topology_query_pdb(
            pdb_with_conect, {"bonded_with": ([self.BLOCK], [])}, "global_ids", keys, reqs
        )
        assert a == b == c


# ===========================================================================
# bonded_with — applies uniformly across every per-atom request
# ===========================================================================

class TestPDBBondedWithUniversal:
    """The bond filter must apply to every per-atom topology request,
    not just global_ids — exact same property the PSF integration tests
    enforce for that parser."""

    BLOCK = {"total": True, "count": {"eq": 1}}

    @pytest.mark.parametrize("request_string", [
        "global_ids", "local_ids", "residue_ids",
        "atom_names", "residue_names", "segment_names",
        "x", "y", "z", "occupancy", "temperature_coeff",
    ])
    def test_request_has_terminal_length(self, pdb_with_conect, request_string):
        keys, reqs = pp._get_topology_keys_reqs_pdb(pdb_with_conect)
        result = pp._get_topology_query_pdb(
            pdb_with_conect,
            {"bonded_with": self.BLOCK},
            request_string, keys, reqs,
        )
        assert len(result) == len(TERMINAL_IDS)

    def test_atom_names_correspond_to_global_ids(self, pdb_with_conect):
        keys, reqs = pp._get_topology_keys_reqs_pdb(pdb_with_conect)
        ids = pp._get_topology_query_pdb(
            pdb_with_conect, {"bonded_with": self.BLOCK}, "global_ids", keys, reqs
        )
        names = pp._get_topology_query_pdb(
            pdb_with_conect, {"bonded_with": self.BLOCK}, "atom_names", keys, reqs
        )
        expected = [MASTER_ATOMS[i]["name"] for i in ids]
        assert names == expected

    def test_positions_correspond_to_global_ids(self, pdb_with_conect):
        keys, reqs = pp._get_topology_keys_reqs_pdb(pdb_with_conect)
        ids = pp._get_topology_query_pdb(
            pdb_with_conect, {"bonded_with": self.BLOCK}, "global_ids", keys, reqs
        )
        pos = pp._get_topology_query_pdb(
            pdb_with_conect, {"bonded_with": self.BLOCK}, "positions", keys, reqs
        )
        assert pos.shape == (1, len(ids), 3)


# ===========================================================================
# bonded_with — neighbour sub-queries (Level 2)
# ===========================================================================

class TestPDBBondedWithNeighbour:

    def test_bonded_to_ot(self, pdb_with_conect):
        # Atoms bonded to at least one OT (water oxygen) -> the water hydrogens.
        keys, reqs = pp._get_topology_keys_reqs_pdb(pdb_with_conect)
        ids = pp._get_topology_query_pdb(
            pdb_with_conect,
            {"bonded_with": {
                "neighbor": {"atom_name": "OH2"},
                "count": {"ge": 1},
            }},
            "global_ids", keys, reqs,
        )
        # OH2 is the OT atom name in our PDB (PDB has no atom_type column,
        # so we filter neighbours by atom_name instead).
        assert sorted(ids) == BONDED_TO_OT_IDS

    def test_bonded_to_ot_yields_ht_atoms(self, pdb_with_conect):
        # Derived expectation: neighbours of water O are H atoms (H1/H2).
        keys, reqs = pp._get_topology_keys_reqs_pdb(pdb_with_conect)
        ids = pp._get_topology_query_pdb(
            pdb_with_conect,
            {"bonded_with": {
                "neighbor": {"atom_name": "OH2"},
                "count": {"ge": 1},
            }},
            "global_ids", keys, reqs,
        )
        assert sorted(ids) == sorted(HT_TYPE_IDS)

    def test_bonded_to_ligand(self, pdb_with_conect):
        keys, reqs = pp._get_topology_keys_reqs_pdb(pdb_with_conect)
        ids = pp._get_topology_query_pdb(
            pdb_with_conect,
            {"bonded_with": {
                "neighbor": {"segment_name": "LIG"},
                "count": {"ge": 1},
            }},
            "global_ids", keys, reqs,
        )
        assert sorted(ids) == BONDED_TO_LIG_IDS

    def test_neighbour_block_without_neighbor_dict_raises(self, pdb_with_conect):
        keys, reqs = pp._get_topology_keys_reqs_pdb(pdb_with_conect)
        with pytest.raises(ValueError):
            pp._get_topology_query_pdb(
                pdb_with_conect,
                {"bonded_with": {"count": {"ge": 1}}},  # no 'neighbor', no 'total'
                "global_ids", keys, reqs,
            )


# ===========================================================================
# bonded_with — mode "any" / "all" semantics
# ===========================================================================

class TestPDBBondedWithMode:

    def test_mode_all_requires_every_block(self, pdb_with_conect):
        keys, reqs = pp._get_topology_keys_reqs_pdb(pdb_with_conect)
        # Atoms that are BOTH terminal AND bonded to at least one OH2 ->
        # exactly the water H atoms (terminal AND bonded to water O).
        ids = pp._get_topology_query_pdb(
            pdb_with_conect,
            {
                "bonded_with": [
                    {"total": True, "count": {"eq": 1}},
                    {"neighbor": {"atom_name": "OH2"}, "count": {"ge": 1}},
                ],
                "bonded_with_mode": ("all", None),
            },
            "global_ids", keys, reqs,
        )
        assert sorted(ids) == sorted(HT_TYPE_IDS)

    def test_mode_any_only_one_must_pass(self, pdb_with_conect):
        keys, reqs = pp._get_topology_keys_reqs_pdb(pdb_with_conect)
        # Mode "any": atoms that are EITHER isolated OR have degree==2.
        ids = pp._get_topology_query_pdb(
            pdb_with_conect,
            {
                "bonded_with": [
                    {"total": True, "count": {"eq": 0}},
                    {"total": True, "count": {"eq": 2}},
                ],
                "bonded_with_mode": ("any", None),
            },
            "global_ids", keys, reqs,
        )
        expected = sorted(set(ISOLATED_IDS) | set(DEGREE_2_IDS))
        assert sorted(ids) == expected

    def test_invalid_mode_raises(self, pdb_with_conect):
        keys, reqs = pp._get_topology_keys_reqs_pdb(pdb_with_conect)
        with pytest.raises(ValueError):
            pp._get_topology_query_pdb(
                pdb_with_conect,
                {
                    "bonded_with": {"total": True, "count": {"eq": 1}},
                    "bonded_with_mode": ("nonsense", None),
                },
                "global_ids", keys, reqs,
            )


# ===========================================================================
# bonded_with — exclude blocks
# ===========================================================================

class TestPDBBondedWithExclude:

    def test_exclude_terminal_yields_non_terminal(self, pdb_with_conect):
        # No include block -> all atoms pass; exclude removes terminals.
        keys, reqs = pp._get_topology_keys_reqs_pdb(pdb_with_conect)
        ids = pp._get_topology_query_pdb(
            pdb_with_conect,
            {"bonded_with": ([], [{"total": True, "count": {"eq": 1}}])},
            "global_ids", keys, reqs,
        )
        expected = sorted(set(range(N_ATOMS)) - set(TERMINAL_IDS))
        assert sorted(ids) == expected

    def test_exclusion_wins_over_inclusion(self, pdb_with_conect):
        # Include degree>=1 (everything except ions); exclude degree==1
        # (terminals). Result = atoms with degree >= 2.
        keys, reqs = pp._get_topology_keys_reqs_pdb(pdb_with_conect)
        ids = pp._get_topology_query_pdb(
            pdb_with_conect,
            {"bonded_with": (
                [{"total": True, "count": {"ge": 1}}],
                [{"total": True, "count": {"eq": 1}}],
            )},
            "global_ids", keys, reqs,
        )
        expected = sorted(i for i, d in DEGREE_MAP.items() if d >= 2)
        assert sorted(ids) == expected


# ===========================================================================
# Bidirectional CONECT files produce identical results to unidirectional
# ===========================================================================

class TestPDBBondedWithBidirectionalCONECT:
    """A PDB that lists each bond twice (strict spec) must produce the
    exact same query results as one that lists each bond once. This is
    the guarantee provided by the deduplicating CONECT iterator."""

    @pytest.mark.parametrize("query", [
        {"bonded_with": {"total": True, "count": {"eq": 1}}},
        {"bonded_with": {"total": True, "count": {"eq": 0}}},
        {"bonded_with": {"total": True, "count": {"ge": 2}}},
        {"bonded_with": {"neighbor": {"atom_name": "OH2"}, "count": {"ge": 1}}},
    ])
    def test_uni_vs_bidi_match(self, pdb_with_conect, pdb_with_conect_bidi, query):
        keys_u, reqs_u = pp._get_topology_keys_reqs_pdb(pdb_with_conect)
        keys_b, reqs_b = pp._get_topology_keys_reqs_pdb(pdb_with_conect_bidi)
        uni  = pp._get_topology_query_pdb(pdb_with_conect,      query, "global_ids", keys_u, reqs_u)
        bidi = pp._get_topology_query_pdb(pdb_with_conect_bidi, query, "global_ids", keys_b, reqs_b)
        assert sorted(uni) == sorted(bidi)


# ===========================================================================
# Recursion guard for nested bonded_with neighbour sub-queries
# ===========================================================================

class TestPDBBondedWithRecursionGuard:

    def test_deep_nesting_raises_recursion_error(self, pdb_with_conect):
        # Build a query that nests bonded_with deeper than the cap.
        keys, reqs = pp._get_topology_keys_reqs_pdb(pdb_with_conect)
        depth = qh.MAX_BONDED_WITH_DEPTH + 2
        q = {"atom_name": "OH2"}
        for _ in range(depth):
            q = {"bonded_with": {"neighbor": q, "count": {"ge": 1}}}
        with pytest.raises(RecursionError):
            pp._get_topology_query_pdb(
                pdb_with_conect, q, "global_ids", keys, reqs
            )


# ===========================================================================
# Planner — exists, returns sensible metadata, refuses bonded_with
# ===========================================================================

class TestPlanTopologyQueryPDB:

    def test_returns_dict(self, pdb_with_conect):
        keys, reqs = pp._get_topology_keys_reqs_pdb(pdb_with_conect)
        plan = pp._plan_topology_query_pdb(
            pdb_with_conect, {}, "global_ids", keys, reqs
        )
        assert isinstance(plan, dict)
        assert plan["planner_mode"] == "stochastic"

    def test_unsupported_request_raises(self, pdb_with_conect):
        keys, reqs = pp._get_topology_keys_reqs_pdb(pdb_with_conect)
        with pytest.raises(ValueError):
            pp._plan_topology_query_pdb(
                pdb_with_conect, {}, "nonsense_request", keys, reqs
            )

    def test_bonded_with_in_query_returns_unsupported(self, pdb_with_conect):
        keys, reqs = pp._get_topology_keys_reqs_pdb(pdb_with_conect)
        plan = pp._plan_topology_query_pdb(
            pdb_with_conect,
            {"bonded_with": {"total": True, "count": {"eq": 1}}},
            "global_ids", keys, reqs,
        )
        assert plan.get("supported") is False

    def test_property_request_not_estimated(self, pdb_with_conect):
        keys, reqs = pp._get_topology_keys_reqs_pdb(pdb_with_conect)
        plan = pp._plan_topology_query_pdb(
            pdb_with_conect, {}, "property-number_of_atoms", keys, reqs
        )
        # Scalar property -> planner explicitly reports unsupported with reason.
        assert plan.get("supported") is False

    def test_n_atoms_estimate_close_to_truth(self, pdb_with_conect):
        keys, reqs = pp._get_topology_keys_reqs_pdb(pdb_with_conect)
        plan = pp._plan_topology_query_pdb(
            pdb_with_conect, {}, "global_ids", keys, reqs
        )
        # 54 atoms is well below the sample budget so the estimate should
        # be exact (sample_probability == 1.0 path).
        assert plan["n_atoms"] == N_ATOMS

