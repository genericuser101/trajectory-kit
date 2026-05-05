"""
tests/test_pos_writer.py
========================
Integration tests for trajectory_kit.pos_writer.

Scope: write a new typing file (PDB or XYZ) with every atom's coordinates
replaced by the corresponding row of a specified trajectory frame.

Covers per-format:
- Frame 0 round-trip (synthetic trajectory frame 0 == typing's own coords)
- Frame k shifts x by +k per the fixture
- Non-coord columns preserved byte-for-byte on atom lines
- Non-atom records preserved byte-for-byte
- Cross-consistency: reloading the output yields the same coords

Format-agnostic:
- Input validation: non-sim, unsupported typing, no typing, no traj,
  bad frame, both output args, mismatched output extension, pre-existing
  output path
- Output naming: default, output_dir, output_filepath
- Path-wrapper convenience function
- Atom-count sanity check raises when typing and trajectory disagree
- PDB-only: %8.3f -> %8.2f overflow fallback, hard raise on unrepresentable
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from trajectory_kit import sim as Sim
from trajectory_kit import pos_writer

from conftest import N_ATOMS, N_FRAMES


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sim_pdb_dcd(synthetic_pdb, synthetic_dcd):
    return Sim(typing=synthetic_pdb, trajectory=synthetic_dcd)


@pytest.fixture
def sim_xyz_dcd(synthetic_xyz, synthetic_dcd):
    return Sim(typing=synthetic_xyz, trajectory=synthetic_dcd)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_pdb_atom_coords(path: Path) -> list[tuple[float, float, float]]:
    coords = []
    with open(path, 'rt', encoding='latin-1') as f:
        for line in f:
            if line.startswith('ATOM  ') or line.startswith('HETATM'):
                coords.append((float(line[30:38]), float(line[38:46]), float(line[46:54])))
    return coords


def _read_pdb_atom_lines(path: Path) -> list[str]:
    with open(path, 'rt', encoding='latin-1') as f:
        return [l for l in f if l.startswith('ATOM  ') or l.startswith('HETATM')]


def _read_pdb_all_lines(path: Path) -> list[str]:
    with open(path, 'rt', encoding='latin-1') as f:
        return list(f)


def _read_xyz_atom_coords(path: Path) -> list[tuple[float, float, float]]:
    """Skip the count + comment lines and parse element + xyz tokens."""
    with open(path, 'rt', encoding='utf-8', errors='replace') as f:
        lines = f.readlines()
    n = int(lines[0].strip())
    coords = []
    for line in lines[2:2 + n]:
        parts = line.split()
        coords.append((float(parts[1]), float(parts[2]), float(parts[3])))
    return coords


def _read_xyz_all_lines(path: Path) -> list[str]:
    with open(path, 'rt', encoding='utf-8', errors='replace') as f:
        return f.readlines()


# ===========================================================================
# PDB — core behaviour
# ===========================================================================

class TestPdbCore:

    def test_writes_file_at_default_location(self, sim_pdb_dcd, tmp_path):
        out = pos_writer.write_with_frame(sim_pdb_dcd, frame=0, output_dir=tmp_path)
        assert out.exists() and out.is_file() and out.suffix == ".pdb"

    def test_default_filename_pattern(self, sim_pdb_dcd, tmp_path):
        out = pos_writer.write_with_frame(sim_pdb_dcd, frame=2, output_dir=tmp_path)
        assert out.name == "synth_from_synth_f00000000002.pdb"

    def test_frame_0_matches_source_coords(self, sim_pdb_dcd, tmp_path):
        out = pos_writer.write_with_frame(sim_pdb_dcd, frame=0, output_dir=tmp_path)
        src = _read_pdb_atom_coords(Path(sim_pdb_dcd.type_file))
        dst = _read_pdb_atom_coords(out)
        assert len(src) == len(dst) == N_ATOMS
        np.testing.assert_allclose(src, dst, atol=1e-3)

    def test_frame_k_shifts_x_by_k(self, sim_pdb_dcd, tmp_path):
        out = pos_writer.write_with_frame(sim_pdb_dcd, frame=3, output_dir=tmp_path)
        src = np.array(_read_pdb_atom_coords(Path(sim_pdb_dcd.type_file)))
        dst = np.array(_read_pdb_atom_coords(out))
        np.testing.assert_allclose(dst[:, 0], src[:, 0] + 3.0, atol=1e-3)
        np.testing.assert_allclose(dst[:, 1:], src[:, 1:],     atol=1e-3)

    def test_every_frame_writes_cleanly(self, sim_pdb_dcd, tmp_path):
        src = np.array(_read_pdb_atom_coords(Path(sim_pdb_dcd.type_file)))
        for k in range(N_FRAMES):
            out = pos_writer.write_with_frame(sim_pdb_dcd, frame=k, output_dir=tmp_path)
            dst = np.array(_read_pdb_atom_coords(out))
            np.testing.assert_allclose(dst[:, 0], src[:, 0] + k, atol=1e-3)


# ===========================================================================
# PDB — byte-for-byte preservation
# ===========================================================================

class TestPdbPreservation:

    def test_non_coord_columns_preserved(self, sim_pdb_dcd, tmp_path):
        out = pos_writer.write_with_frame(sim_pdb_dcd, frame=2, output_dir=tmp_path)
        src_lines = _read_pdb_atom_lines(Path(sim_pdb_dcd.type_file))
        dst_lines = _read_pdb_atom_lines(out)
        for s_line, d_line in zip(src_lines, dst_lines):
            s_pad = s_line.ljust(80); d_pad = d_line.ljust(80)
            assert s_pad[:30] == d_pad[:30]
            assert s_pad[54:] == d_pad[54:]

    def test_non_atom_records_preserved(self, synthetic_dcd, tmp_path):
        from conftest import _write_pdb
        base    = tmp_path / "base.pdb"; _write_pdb(base)
        src_pdb = tmp_path / "with_extras.pdb"
        base_lines = _read_pdb_all_lines(base)

        extras_top = [
            "REMARK 1 SYNTHETIC PDB WITH EXTRA RECORDS\n",
            "CRYST1   50.000   50.000   50.000  90.00  90.00  90.00 P 1           1\n",
        ]
        extras_bot = ["CONECT    1    2\n", "END\n"]
        base_body  = [l for l in base_lines if not l.startswith("END")]
        src_pdb.write_text("".join(extras_top + base_body + extras_bot), encoding='latin-1')

        s   = Sim(typing=src_pdb, trajectory=synthetic_dcd)
        out = pos_writer.write_with_frame(s, frame=1, output_dir=tmp_path)

        src_lines = _read_pdb_all_lines(src_pdb)
        dst_lines = _read_pdb_all_lines(out)
        assert len(src_lines) == len(dst_lines)
        for s_line, d_line in zip(src_lines, dst_lines):
            if not (s_line.startswith('ATOM  ') or s_line.startswith('HETATM')):
                assert s_line == d_line


# ===========================================================================
# PDB — coord overflow
# ===========================================================================

class TestPdbCoordOverflow:

    def test_graceful_degradation_to_2dp(self):
        from trajectory_kit.pos_writer import _format_pdb_coord
        s = _format_pdb_coord(12345.678, label='x')
        assert len(s) == 8 and s == "12345.68"

    def test_large_negative_degrades(self):
        from trajectory_kit.pos_writer import _format_pdb_coord
        s = _format_pdb_coord(-1234.567, label='y')
        assert len(s) == 8 and s == "-1234.57"

    def test_value_fits_3dp_stays_3dp(self):
        from trajectory_kit.pos_writer import _format_pdb_coord
        s = _format_pdb_coord(42.123, label='z')
        assert len(s) == 8 and abs(float(s) - 42.123) < 1e-6

    def test_unrepresentable_raises(self):
        from trajectory_kit.pos_writer import _format_pdb_coord
        with pytest.raises(ValueError, match="cannot be represented"):
            _format_pdb_coord(1e12, label='x')

    def test_rounds_into_overflow_raises(self):
        from trajectory_kit.pos_writer import _format_pdb_coord
        with pytest.raises(ValueError, match="cannot be represented"):
            _format_pdb_coord(99999.999, label='x')

    def test_nan_raises(self):
        from trajectory_kit.pos_writer import _format_pdb_coord
        with pytest.raises(ValueError, match="non-finite"):
            _format_pdb_coord(float('nan'), label='x')

    def test_inf_raises(self):
        from trajectory_kit.pos_writer import _format_pdb_coord
        with pytest.raises(ValueError, match="non-finite"):
            _format_pdb_coord(float('inf'), label='x')


# ===========================================================================
# XYZ — core behaviour
# ===========================================================================

class TestXyzCore:

    def test_writes_file_at_default_location(self, sim_xyz_dcd, tmp_path):
        out = pos_writer.write_with_frame(sim_xyz_dcd, frame=0, output_dir=tmp_path)
        assert out.exists() and out.is_file() and out.suffix == ".xyz"

    def test_default_filename_pattern(self, sim_xyz_dcd, tmp_path):
        out = pos_writer.write_with_frame(sim_xyz_dcd, frame=2, output_dir=tmp_path)
        assert out.name == "synth_from_synth_f00000000002.xyz"

    def test_frame_0_matches_source_coords(self, sim_xyz_dcd, tmp_path):
        out = pos_writer.write_with_frame(sim_xyz_dcd, frame=0, output_dir=tmp_path)
        src = _read_xyz_atom_coords(Path(sim_xyz_dcd.type_file))
        dst = _read_xyz_atom_coords(out)
        assert len(src) == len(dst) == N_ATOMS
        np.testing.assert_allclose(src, dst, atol=1e-3)

    def test_frame_k_shifts_x_by_k(self, sim_xyz_dcd, tmp_path):
        out = pos_writer.write_with_frame(sim_xyz_dcd, frame=3, output_dir=tmp_path)
        src = np.array(_read_xyz_atom_coords(Path(sim_xyz_dcd.type_file)))
        dst = np.array(_read_xyz_atom_coords(out))
        np.testing.assert_allclose(dst[:, 0], src[:, 0] + 3.0, atol=1e-3)
        np.testing.assert_allclose(dst[:, 1:], src[:, 1:],     atol=1e-3)

    def test_every_frame_writes_cleanly(self, sim_xyz_dcd, tmp_path):
        src = np.array(_read_xyz_atom_coords(Path(sim_xyz_dcd.type_file)))
        for k in range(N_FRAMES):
            out = pos_writer.write_with_frame(sim_xyz_dcd, frame=k, output_dir=tmp_path)
            dst = np.array(_read_xyz_atom_coords(out))
            np.testing.assert_allclose(dst[:, 0], src[:, 0] + k, atol=1e-3)


# ===========================================================================
# XYZ — preservation
# ===========================================================================

class TestXyzPreservation:

    def test_count_and_comment_lines_preserved(self, sim_xyz_dcd, tmp_path):
        out = pos_writer.write_with_frame(sim_xyz_dcd, frame=2, output_dir=tmp_path)
        src_lines = _read_xyz_all_lines(Path(sim_xyz_dcd.type_file))
        dst_lines = _read_xyz_all_lines(out)
        assert src_lines[0] == dst_lines[0]   # atom count
        assert src_lines[1] == dst_lines[1]   # comment

    def test_element_token_preserved(self, sim_xyz_dcd, tmp_path):
        out = pos_writer.write_with_frame(sim_xyz_dcd, frame=2, output_dir=tmp_path)
        src_lines = _read_xyz_all_lines(Path(sim_xyz_dcd.type_file))
        dst_lines = _read_xyz_all_lines(out)
        for s_line, d_line in zip(src_lines[2:2 + N_ATOMS], dst_lines[2:2 + N_ATOMS]):
            assert s_line.split()[0] == d_line.split()[0]

    def test_extra_columns_preserved(self, synthetic_dcd, tmp_path):
        """If the XYZ has extra trailing columns (e.g. velocities), those
        survive untouched while x/y/z are rewritten."""
        from conftest import MASTER_ATOMS

        src_xyz = tmp_path / "with_velocities.xyz"
        lines = [f"{N_ATOMS}\n", "synthetic XYZ with velocities\n"]
        for a in MASTER_ATOMS:
            # element x y z vx vy vz
            lines.append(
                f"{a['name']:<6s} "
                f"{a['x']:10.4f} {a['y']:10.4f} {a['z']:10.4f} "
                f"{0.123:10.6f} {0.456:10.6f} {0.789:10.6f}\n"
            )
        src_xyz.write_text("".join(lines))

        s   = Sim(typing=src_xyz, trajectory=synthetic_dcd)
        out = pos_writer.write_with_frame(s, frame=1, output_dir=tmp_path)

        src_lines = _read_xyz_all_lines(src_xyz)
        dst_lines = _read_xyz_all_lines(out)
        for s_line, d_line in zip(src_lines[2:2 + N_ATOMS], dst_lines[2:2 + N_ATOMS]):
            s_parts = s_line.split()
            d_parts = d_line.split()
            assert s_parts[0]  == d_parts[0]      # element
            assert s_parts[4:] == d_parts[4:]     # vx, vy, vz preserved

    def test_whitespace_layout_preserved_at_frame_0(self, sim_xyz_dcd, tmp_path):
        """Frame-0 round-trip on a well-formatted XYZ should be byte-identical
        on the atom lines (the synthetic fixture uses %10.4f → 4 decimals
        round-trip cleanly through float32 for its small values)."""
        out = pos_writer.write_with_frame(sim_xyz_dcd, frame=0, output_dir=tmp_path)
        src_lines = _read_xyz_all_lines(Path(sim_xyz_dcd.type_file))
        dst_lines = _read_xyz_all_lines(out)
        for s_line, d_line in zip(src_lines[2:2 + N_ATOMS], dst_lines[2:2 + N_ATOMS]):
            assert s_line == d_line, (
                f"frame-0 atom line not preserved:\n  src={s_line!r}\n  dst={d_line!r}"
            )


# ===========================================================================
# XYZ — number-formatting helpers
# ===========================================================================

class TestXyzFormatting:

    def test_precision_inferred(self):
        from trajectory_kit.pos_writer import _infer_precision
        assert _infer_precision("12.3456")     == 4
        assert _infer_precision("  12.34  ")   == 2
        assert _infer_precision("-0.1")        == 1
        assert _infer_precision("100")         is None    # no decimal
        assert _infer_precision("1e3")         is None    # scientific

    def test_format_like_preserves_width_and_precision(self):
        from trajectory_kit.pos_writer import _format_xyz_coord_like
        # Original token: 10 chars wide, 4 dp
        out = _format_xyz_coord_like("    1.2345", 9.8765, label='x')
        assert len(out) == 10
        assert out.strip() == "9.8765"

    def test_format_widens_when_value_overflows_original_width(self):
        from trajectory_kit.pos_writer import _format_xyz_coord_like
        # Original 7 chars, value won't fit
        out = _format_xyz_coord_like("12.3456", -1234567.890123, label='x')
        # 4 dp inferred from original; widens as needed
        assert out.endswith(".8901")
        assert len(out) >= 7


# ===========================================================================
# Format-agnostic — input validation
# ===========================================================================

class TestInputValidation:

    def test_non_sim_first_arg_raises(self, tmp_path):
        with pytest.raises(TypeError, match="sim instance"):
            pos_writer.write_with_frame("not a sim", frame=0, output_dir=tmp_path)

    def test_no_typing_file_raises(self, tmp_path):
        s = Sim()
        with pytest.raises(ValueError, match="typing file"):
            pos_writer.write_with_frame(s, frame=0, output_dir=tmp_path)

    def test_unsupported_typing_raises(self, synthetic_mae, synthetic_dcd, tmp_path):
        """MAE is a supported typing format on sim but pos_writer doesn't
        handle it (yet) — should raise a clear error."""
        s = Sim(typing=synthetic_mae, trajectory=synthetic_dcd)
        with pytest.raises(ValueError, match="supports typing formats"):
            pos_writer.write_with_frame(s, frame=0, output_dir=tmp_path)

    def test_no_trajectory_loaded_raises(self, synthetic_pdb, tmp_path):
        s = Sim(typing=synthetic_pdb)
        with pytest.raises(ValueError, match="trajectory file"):
            pos_writer.write_with_frame(s, frame=0, output_dir=tmp_path)

    def test_negative_frame_raises(self, sim_pdb_dcd, tmp_path):
        with pytest.raises(ValueError, match="non-negative"):
            pos_writer.write_with_frame(sim_pdb_dcd, frame=-1, output_dir=tmp_path)

    def test_bool_rejected_as_frame(self, sim_pdb_dcd, tmp_path):
        with pytest.raises(ValueError, match="non-negative int"):
            pos_writer.write_with_frame(sim_pdb_dcd, frame=True, output_dir=tmp_path)

    def test_both_output_args_raises(self, sim_pdb_dcd, tmp_path):
        with pytest.raises(ValueError, match="not both"):
            pos_writer.write_with_frame(
                sim_pdb_dcd, frame=0,
                output_dir=tmp_path,
                output_filepath=tmp_path / "x.pdb",
            )

    def test_mismatched_extension_raises_pdb_to_xyz(self, sim_pdb_dcd, tmp_path):
        with pytest.raises(ValueError, match="does not match the source"):
            pos_writer.write_with_frame(
                sim_pdb_dcd, frame=0,
                output_filepath=tmp_path / "wrong.xyz",
            )

    def test_mismatched_extension_raises_xyz_to_pdb(self, sim_xyz_dcd, tmp_path):
        with pytest.raises(ValueError, match="does not match the source"):
            pos_writer.write_with_frame(
                sim_xyz_dcd, frame=0,
                output_filepath=tmp_path / "wrong.pdb",
            )

    def test_existing_output_path_raises(self, sim_pdb_dcd, tmp_path):
        out = pos_writer.write_with_frame(sim_pdb_dcd, frame=0, output_dir=tmp_path)
        with pytest.raises(ValueError, match="already exists"):
            pos_writer.write_with_frame(sim_pdb_dcd, frame=0, output_dir=tmp_path)
        assert out.exists()


# ===========================================================================
# Format-agnostic — output naming
# ===========================================================================

class TestOutputNaming:

    def test_pdb_default_name(self, sim_pdb_dcd, tmp_path):
        out = pos_writer.write_with_frame(sim_pdb_dcd, frame=1, output_dir=tmp_path)
        assert out.name == "synth_from_synth_f00000000001.pdb"

    def test_xyz_default_name(self, sim_xyz_dcd, tmp_path):
        out = pos_writer.write_with_frame(sim_xyz_dcd, frame=1, output_dir=tmp_path)
        assert out.name == "synth_from_synth_f00000000001.xyz"

    def test_explicit_filepath_pdb(self, sim_pdb_dcd, tmp_path):
        target = tmp_path / "custom.pdb"
        out = pos_writer.write_with_frame(sim_pdb_dcd, frame=0, output_filepath=target)
        assert out == target and target.exists()

    def test_explicit_filepath_xyz(self, sim_xyz_dcd, tmp_path):
        target = tmp_path / "custom.xyz"
        out = pos_writer.write_with_frame(sim_xyz_dcd, frame=0, output_filepath=target)
        assert out == target and target.exists()

    def test_default_name_next_to_typing_when_no_dir(self, synthetic_dcd, tmp_path):
        from conftest import _write_pdb, _write_dcd
        local_pdb = tmp_path / "local.pdb"; _write_pdb(local_pdb)
        local_dcd = tmp_path / "local.dcd"; _write_dcd(local_dcd)
        s   = Sim(typing=local_pdb, trajectory=local_dcd)
        out = pos_writer.write_with_frame(s, frame=1)
        assert out.parent == tmp_path
        assert out.name   == "local_from_local_f00000000001.pdb"


# ===========================================================================
# Cross-consistency with sim
# ===========================================================================

class TestSimCrossConsistency:

    def test_pdb_output_reloaded_matches_frame(self, sim_pdb_dcd, tmp_path):
        out = pos_writer.write_with_frame(sim_pdb_dcd, frame=2, output_dir=tmp_path)
        original = sim_pdb_dcd.positions(TRAJ_Q={'frame_interval': (2, 2)})
        s2       = Sim(typing=out)
        reloaded = s2.positions()
        np.testing.assert_allclose(reloaded, original, atol=1e-3)

    def test_xyz_output_reloaded_matches_frame(self, sim_xyz_dcd, tmp_path):
        out = pos_writer.write_with_frame(sim_xyz_dcd, frame=2, output_dir=tmp_path)
        original = sim_xyz_dcd.positions(TRAJ_Q={'frame_interval': (2, 2)})
        s2       = Sim(typing=out)
        reloaded = s2.positions()
        np.testing.assert_allclose(reloaded, original, atol=1e-3)


# ===========================================================================
# Path-wrapper convenience
# ===========================================================================

class TestPathWrapper:

    def test_pdb_via_paths(self, synthetic_pdb, synthetic_dcd, tmp_path):
        out = pos_writer.write_with_frame_from_paths(
            type_filepath       = synthetic_pdb,
            trajectory_filepath = synthetic_dcd,
            frame               = 1,
            output_dir          = tmp_path,
        )
        assert out.exists()
        src = np.array(_read_pdb_atom_coords(synthetic_pdb))
        dst = np.array(_read_pdb_atom_coords(out))
        np.testing.assert_allclose(dst[:, 0], src[:, 0] + 1.0, atol=1e-3)

    def test_xyz_via_paths(self, synthetic_xyz, synthetic_dcd, tmp_path):
        out = pos_writer.write_with_frame_from_paths(
            type_filepath       = synthetic_xyz,
            trajectory_filepath = synthetic_dcd,
            frame               = 2,
            output_dir          = tmp_path,
        )
        assert out.exists()
        src = np.array(_read_xyz_atom_coords(synthetic_xyz))
        dst = np.array(_read_xyz_atom_coords(out))
        np.testing.assert_allclose(dst[:, 0], src[:, 0] + 2.0, atol=1e-3)


# ===========================================================================
# Backwards-compat aliases
# ===========================================================================

class TestBackwardsCompat:

    def test_old_pdb_name_still_works(self, sim_pdb_dcd, tmp_path):
        """write_pdb_with_frame is kept as an alias for write_with_frame."""
        out = pos_writer.write_pdb_with_frame(sim_pdb_dcd, frame=0, output_dir=tmp_path)
        assert out.exists()

    def test_old_pdb_paths_name_still_works(self, synthetic_pdb, synthetic_dcd, tmp_path):
        out = pos_writer.write_pdb_with_frame_from_paths(
            type_filepath       = synthetic_pdb,
            trajectory_filepath = synthetic_dcd,
            frame               = 0,
            output_dir          = tmp_path,
        )
        assert out.exists()


# ===========================================================================
# Atom-count sanity — low-level streamers
# ===========================================================================

class TestAtomCountSanity:

    def test_pdb_with_more_atoms_than_traj_raises(self, tmp_path):
        from conftest import _write_pdb
        from trajectory_kit.pos_writer import _stream_rewrite_pdb

        base  = tmp_path / "base.pdb"; _write_pdb(base)
        lines = _read_pdb_all_lines(base)
        last_atom = [l for l in lines if l.startswith('ATOM  ') or l.startswith('HETATM')][-1]
        bad_pdb = tmp_path / "oversized.pdb"
        bad_pdb.write_text("".join(lines + [last_atom]), encoding='latin-1')

        coords = np.zeros((N_ATOMS, 3), dtype=np.float32)
        with pytest.raises(ValueError, match="more ATOM/HETATM rows"):
            _stream_rewrite_pdb(
                src_file = bad_pdb,
                dst_file = tmp_path / "never.pdb",
                coords   = coords,
            )

    def test_xyz_with_more_atoms_than_traj_raises(self, tmp_path):
        from conftest import _write_xyz
        from trajectory_kit.pos_writer import _stream_rewrite_xyz

        base  = tmp_path / "base.xyz"; _write_xyz(base)
        lines = _read_xyz_all_lines(base)

        # Append one extra atom-shaped line. (We don't bother updating the
        # count line — we're testing the streamer's safety check directly,
        # not the public API which would catch this earlier via sim.)
        bad_xyz = tmp_path / "oversized.xyz"
        bad_xyz.write_text("".join(lines + ["X1     0.0000     0.0000     0.0000\n"]),
                           encoding='utf-8')

        coords = np.zeros((N_ATOMS, 3), dtype=np.float32)
        with pytest.raises(ValueError, match="more atom rows"):
            _stream_rewrite_xyz(
                src_file = bad_xyz,
                dst_file = tmp_path / "never.xyz",
                coords   = coords,
            )
