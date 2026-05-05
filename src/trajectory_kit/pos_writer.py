# Preloaded imports
from __future__ import annotations
from pathlib import Path
import re

# Third-party
import numpy as np

# Local
from trajectory_kit.main import sim


# -------------------------------------------------------------------------
# PUBLIC API
# -------------------------------------------------------------------------

# Format -> dispatch table for the streamer.
# Populated below once the streamers are defined.
_STREAMERS: dict = {}


def write_with_frame(
    s:               sim,
    frame:           int,
    *,
    output_dir:      str | Path | None = None,
    output_filepath: str | Path | None = None,
) -> Path:

    '''
    Write a new typing file (PDB or XYZ) that is identical to the one
    loaded on the given sim, except that every atom's coordinates are
    replaced with those of the specified trajectory frame.

    Everything else — atom ordering, atom names, segment ids, occupancy,
    B-factor (PDB), comment line (XYZ), CRYST1, CONECT, REMARKs, END,
    extra columns past z (XYZ velocities/forces) — is preserved exactly.

    Parameters
    ----------
    s : sim
        A sim with a .pdb or .xyz typing file loaded and a trajectory
        file loaded.
    frame : int
        Frame index to pull coordinates from. 0-based.
    output_dir : str | Path, optional
        Directory the output file is written to. Defaults to the directory
        containing the source typing file. Mutually exclusive with
        output_filepath.
    output_filepath : str | Path, optional
        Exact output path. Overrides automatic naming. Mutually exclusive
        with output_dir. Must have the same extension as the source typing
        file.

    Returns
    -------
    Path
        The path of the newly written file.
    '''

    # ------------------------------------------------------------------ #
    # Entry validation                                                   #
    # ------------------------------------------------------------------ #

    if not isinstance(s, sim):
        raise TypeError(
            f"First argument must be a sim instance. Received: {type(s).__name__}"
        )

    if s.type_file is None:
        raise ValueError(
            "pos_writer requires the sim to have a typing file loaded. "
            "Load a typing file with sim.load_typing(...) first."
        )

    if s.type_type not in _STREAMERS:
        supported = sorted(_STREAMERS.keys())
        raise ValueError(
            f"pos_writer supports typing formats {supported}. "
            f"The loaded typing file is {s.type_type!r}."
        )

    if s.traj_file is None:
        raise ValueError(
            "pos_writer requires the sim to have a trajectory file loaded. "
            "Without a trajectory there is nothing to write. "
            "Load a trajectory with sim.load_trajectory(...) first."
        )

    if not isinstance(frame, (int, np.integer)) or isinstance(frame, bool):
        raise ValueError(
            f"'frame' must be a non-negative int. Received: {frame!r}"
        )
    if frame < 0:
        raise ValueError(
            f"'frame' must be a non-negative int. Received: {frame}"
        )

    if output_dir is not None and output_filepath is not None:
        raise ValueError(
            "Pass either output_dir or output_filepath, not both."
        )

    if output_filepath is not None:
        out_ext = Path(output_filepath).suffix.lower()
        if out_ext != s.type_type:
            raise ValueError(
                f"output_filepath extension {out_ext!r} does not match the "
                f"source typing file extension {s.type_type!r}. The writer "
                f"preserves the input format — it does not convert."
            )

    # ------------------------------------------------------------------ #
    # Pull all coords for the requested frame.                            #
    # sim handles frame-range validation and parser dispatch. The result  #
    # is shape (1, num_atoms, 3) in global_id order.                      #
    # ------------------------------------------------------------------ #

    coords_frame = s.positions(TRAJ_Q={'frame_interval': (frame, frame)})[0]

    # ------------------------------------------------------------------ #
    # Resolve the output path                                             #
    # ------------------------------------------------------------------ #

    out_path = _resolve_output_path(
        type_filepath   = Path(s.type_file),
        traj_filepath   = Path(s.traj_file),
        frame           = frame,
        output_dir      = output_dir,
        output_filepath = output_filepath,
    )

    if out_path.exists():
        raise ValueError(
            f"Output path already exists: {out_path}. "
            f"Refusing to overwrite — choose a different output_dir / "
            f"output_filepath, or delete the existing file first."
        )

    # ------------------------------------------------------------------ #
    # Dispatch to the format-specific streamer.                           #
    # Both streamers walk the source file top-to-bottom and pop coords    #
    # off the array sequentially — sim returns coords in global_id order  #
    # and atom records appear in the file in global_id order.             #
    # ------------------------------------------------------------------ #

    streamer = _STREAMERS[s.type_type]
    streamer(
        src_file = Path(s.type_file),
        dst_file = out_path,
        coords   = coords_frame,
    )

    return out_path


# Backwards-compatible alias for the original public name.
write_pdb_with_frame = write_with_frame


# -------------------------------------------------------------------------
# CONVENIENCE WRAPPER
# -------------------------------------------------------------------------

def write_with_frame_from_paths(
    type_filepath:       str | Path,
    trajectory_filepath: str | Path,
    frame:               int,
    *,
    output_dir:          str | Path | None = None,
    output_filepath:     str | Path | None = None,
) -> Path:

    '''
    Convenience wrapper around write_with_frame that builds a sim
    internally from the supplied file paths.
    '''

    s = sim(typing=type_filepath, trajectory=trajectory_filepath)

    return write_with_frame(
        s,
        frame,
        output_dir      = output_dir,
        output_filepath = output_filepath,
    )


write_pdb_with_frame_from_paths = write_with_frame_from_paths


# -------------------------------------------------------------------------
# PATH RESOLUTION
# -------------------------------------------------------------------------

def _resolve_output_path(
    *,
    type_filepath:   Path,
    traj_filepath:   Path,
    frame:           int,
    output_dir:      str | Path | None,
    output_filepath: str | Path | None,
) -> Path:

    '''
    Compute the output path:
      - output_filepath given -> exact path used.
      - output_dir given      -> auto-name in that directory.
      - neither given         -> auto-name next to the source typing file.

    The auto-name uses the source file's extension so the writer preserves
    the input format.
    '''

    if output_filepath is not None:
        return Path(output_filepath)

    ext = type_filepath.suffix
    default_name = (
        f"{type_filepath.stem}_from_{traj_filepath.stem}_f{frame:011d}{ext}" # adjust zero pad for frames
    )

    if output_dir is not None:
        return Path(output_dir) / default_name

    return type_filepath.parent / default_name


# -------------------------------------------------------------------------
# PDB STREAMER
# -------------------------------------------------------------------------

def _stream_rewrite_pdb(
    *,
    src_file: Path,
    dst_file: Path,
    coords:   np.ndarray,   # shape (num_atoms, 3)
) -> None:

    '''
    Stream the source PDB to the destination, replacing every ATOM/HETATM
    line's coordinates (cols 30..54) with the corresponding row of
    ``coords``. Non-atom lines pass through byte-for-byte.

    Raises if the PDB contains more ATOM/HETATM rows than the trajectory
    provides coords for. Asserts at the end that the counts matched.
    '''

    dst_file.parent.mkdir(parents=True, exist_ok=True)

    n_expected = coords.shape[0]
    coord_idx  = 0

    # latin-1: every byte round-trips, including non-ascii junk that may
    # appear in REMARKs. We only touch columns 30..54 of ATOM/HETATM lines.
    with open(src_file, 'rt', encoding='latin-1', newline='') as src, \
         open(dst_file, 'wt', encoding='latin-1', newline='') as dst:

        for line in src:

            if line.startswith('ATOM  ') or line.startswith('HETATM'):

                if coord_idx >= n_expected:
                    raise ValueError(
                        f"PDB contains more ATOM/HETATM rows than the trajectory "
                        f"has atoms ({n_expected}). The two files disagree — "
                        f"this output would be corrupt; aborting."
                    )

                x, y, z = coords[coord_idx]
                dst.write(_splice_pdb_coords(line, x, y, z))
                coord_idx += 1

            else:
                dst.write(line)

    assert coord_idx == n_expected, (
        f"pos_writer sanity check failed: spliced {coord_idx} ATOM/HETATM "
        f"rows but the trajectory frame has {n_expected} atoms. The PDB "
        f"and trajectory disagree on atom count."
    )


def _splice_pdb_coords(line: str, x: float, y: float, z: float) -> str:

    '''
    Replace columns [30:38], [38:46], [46:54] of a PDB ATOM/HETATM line
    with formatted x, y, z. Preserve everything else byte-for-byte,
    including the trailing newline and any trailing whitespace past col 54.
    '''

    # Strip trailing newline for clean column arithmetic, re-add after splice.
    newline_suffix = ''
    body = line
    if body.endswith('\r\n'):
        newline_suffix = '\r\n'
        body = body[:-2]
    elif body.endswith('\n'):
        newline_suffix = '\n'
        body = body[:-1]

    if len(body) < 54:
        body = body.ljust(54)

    prefix = body[:30]
    suffix = body[54:]

    x_str = _format_pdb_coord(x, label='x')
    y_str = _format_pdb_coord(y, label='y')
    z_str = _format_pdb_coord(z, label='z')

    return prefix + x_str + y_str + z_str + suffix + newline_suffix


def _format_pdb_coord(value: float, *, label: str) -> str:

    '''
    Format a coordinate into exactly 8 ASCII characters.

    Primary format is %8.3f (PDB standard). If the value does not fit,
    fall back to %8.2f. If even that does not fit, raise ValueError.
    '''

    if not np.isfinite(value):
        raise ValueError(
            f"Cannot write non-finite coordinate {label}={value!r} to PDB."
        )

    s = f"{value:8.3f}"
    if len(s) == 8:
        return s

    s = f"{value:8.2f}"
    if len(s) == 8:
        return s

    raise ValueError(
        f"Coordinate {label}={value!r} cannot be represented in the "
        f"8-column PDB coordinate field, even at 2-decimal precision. "
        f"Consider a format with wider coordinate columns (e.g. mmCIF)."
    )


# -------------------------------------------------------------------------
# XYZ STREAMER
# -------------------------------------------------------------------------

# Predicate for "this line looks like an atom record" — same heuristic as
# the parser's _is_xyz_atom_line, kept local so we don't reach across modules.
def _is_xyz_atom_line(line: str) -> bool:
    parts = line.split()
    if len(parts) < 4:
        return False
    if parts[0].lstrip('-').replace('.', '', 1).isdigit():
        return False
    try:
        float(parts[1]); float(parts[2]); float(parts[3])
        return True
    except ValueError:
        return False


# Tokeniser that captures (whitespace, token) pairs across a line, so we
# can rebuild the line preserving original spacing exactly.
_XYZ_TOKEN_RE = re.compile(r'(\s+)|(\S+)')


def _stream_rewrite_xyz(
    *,
    src_file: Path,
    dst_file: Path,
    coords:   np.ndarray,
) -> None:

    '''
    Stream the source XYZ to the destination, replacing every atom line's
    x/y/z tokens with the corresponding row of ``coords``. The element
    token, any trailing tokens (e.g. velocities), and all original
    whitespace are preserved.

    The atom-count line and comment line pass through byte-for-byte.
    '''

    dst_file.parent.mkdir(parents=True, exist_ok=True)

    n_expected = coords.shape[0]
    coord_idx  = 0

    with open(src_file, 'rt', encoding='utf-8', errors='replace', newline='') as src, \
         open(dst_file, 'wt', encoding='utf-8',                   newline='') as dst:

        for line in src:

            if _is_xyz_atom_line(line):

                if coord_idx >= n_expected:
                    raise ValueError(
                        f"XYZ contains more atom rows than the trajectory has "
                        f"atoms ({n_expected}). The two files disagree — "
                        f"this output would be corrupt; aborting."
                    )

                x, y, z = coords[coord_idx]
                dst.write(_splice_xyz_coords(line, x, y, z))
                coord_idx += 1

            else:
                dst.write(line)

    assert coord_idx == n_expected, (
        f"pos_writer sanity check failed: spliced {coord_idx} XYZ atom "
        f"rows but the trajectory frame has {n_expected} atoms. The XYZ "
        f"and trajectory disagree on atom count."
    )


def _splice_xyz_coords(line: str, x: float, y: float, z: float) -> str:

    '''
    Replace the 2nd, 3rd, and 4th non-whitespace tokens of an XYZ atom
    line with the supplied x, y, z values. The element token (1st token),
    any trailing tokens past z, and all whitespace are preserved.

    Numeric formatting reproduces the original tokens' (width, precision)
    so a round-trip on a well-formed file is bit-identical to within
    float32 precision. If the new value does not fit in the original
    width, the field widens by as many chars as needed.
    '''

    # Newline suffix preserved exactly.
    newline_suffix = ''
    body = line
    if body.endswith('\r\n'):
        newline_suffix = '\r\n'; body = body[:-2]
    elif body.endswith('\n'):
        newline_suffix = '\n';   body = body[:-1]

    # Walk the line capturing alternating whitespace/token chunks.
    chunks: list[str] = []      # raw segments (whitespace OR token)
    is_token: list[bool] = []   # parallel flag

    for m in _XYZ_TOKEN_RE.finditer(body):
        ws, tok = m.group(1), m.group(2)
        if ws is not None:
            chunks.append(ws);  is_token.append(False)
        else:
            chunks.append(tok); is_token.append(True)

    # Find the indices of the 2nd, 3rd, 4th tokens — those are x, y, z.
    token_positions = [i for i, t in enumerate(is_token) if t]
    if len(token_positions) < 4:
        # Defensive: predicate said it's an atom line, so this should never
        # trigger. Treat as a hard bug if it does.
        raise ValueError(
            f"XYZ atom line has fewer than 4 tokens after splitting: {line!r}"
        )

    for axis, (val, label) in zip(token_positions[1:4],
                                  [(x, 'x'), (y, 'y'), (z, 'z')]):
        original_token  = chunks[axis]
        chunks[axis]    = _format_xyz_coord_like(original_token, val, label=label)

    return ''.join(chunks) + newline_suffix


def _format_xyz_coord_like(original: str, value: float, *, label: str) -> str:

    '''
    Format ``value`` using the (width, precision) inferred from the
    ``original`` token. If the formatted result is wider than the
    original, the field widens; precision is preserved.

    Falls back to a default of width=12, precision=6 if the original is
    not a recognisable decimal (e.g. scientific notation in the source).
    '''

    if not np.isfinite(value):
        raise ValueError(
            f"Cannot write non-finite coordinate {label}={value!r} to XYZ."
        )

    width     = len(original)
    precision = _infer_precision(original)

    if precision is None:
        # Source used integer or scientific or otherwise odd; fall back.
        precision = 6
        width     = max(width, precision + 6)   # sign + leading + dot + precision + slack

    # Format with the inferred precision; right-justify within the
    # original width. If the formatted number is wider, just emit it as-is.
    s = f"{value:.{precision}f}"
    if len(s) <= width:
        s = s.rjust(width)

    return s


# Recognise tokens of the form [optional sign][digits].[digits], possibly
# leading whitespace already stripped. Returns the digit count after the
# decimal point, or None for scientific notation / non-decimal tokens.
_XYZ_DECIMAL_RE = re.compile(r'^[+-]?\d*\.(\d+)$')


def _infer_precision(token: str) -> int | None:
    m = _XYZ_DECIMAL_RE.match(token.strip())
    if m is None:
        return None
    return len(m.group(1))


# -------------------------------------------------------------------------
# DISPATCH TABLE
# -------------------------------------------------------------------------

_STREAMERS[".pdb"] = _stream_rewrite_pdb
_STREAMERS[".xyz"] = _stream_rewrite_xyz
