# preloaded imports
from pathlib import Path

# mandatory imports
import numpy as np

# local imports
from trajectory_kit import _file_parse_help as fph
from trajectory_kit import _query_help as qh


# =========================================================================
# COOR FORMAT NOTES  (NAMD binary coordinate / "namdbin" format)
# =========================================================================
#
# A .coor file produced by NAMD with binaryoutput yes (the default) is a
# **raw, headerless binary stream** — there are NO Fortran record markers
# (unlike DCD), and NO text header.  The layout on disk is exactly:
#
#   Bytes 0..3          int32           — atom count  N
#   Bytes 4..4+N*24-1   N * (float64, float64, float64)  — XYZ per atom
#
# All values are written in the **native endianness of the machine that ran
# NAMD**.  Little-endian (x86/x86-64) is the overwhelmingly common case;
# NAMD's own docs note that the flipbinpdb utility can swap bytes if the
# file was produced on a big-endian machine (e.g. old IBM Blue Gene).
#
# Coordinates are in **Angstroms**, double precision (float64).
# Velocities (.vel) share the same binary layout; velocities are in NAMD
# internal units — multiply by 20.45482706 to obtain Å/ps.
#
# Because there is only ONE set of coordinates (no time axis), the
# "trajectory" domain for .coor is a single-frame trajectory.
# frame_interval is NOT a supported keyword — .coor has no concept of
# frames.  If the caller passes frame_interval in the query dictionary it
# is silently ignored and a user-facing warning is printed.
#
# Endianness detection
# --------------------
# We read the 8-byte atom-count record and try both endians.  The correct
# one produces a positive integer that is consistent with the file size:
#
#   expected_file_size = 8  +  N * 3 * 8
#
# If neither endian passes the size check we fall back to little-endian and
# issue a warning — the file may be corrupt or from a non-standard NAMD build.
#
# References
# ----------
# NAMD User Guide §6 "Input/Output Files"
#   https://www.ks.uiuc.edu/Research/namd/current/ug/node12.html
# MDAnalysis NAMDBIN reader (independent open-source implementation)
#   https://docs.mdanalysis.org/stable/documentation_pages/coordinates/NAMDBIN.html
# NAMD mailing list thread on binary format internals
#   https://www.ks.uiuc.edu/Research/namd/mailing_list/namd-l.2009-2010/4104.html
#


# =========================================================================
# INTERNAL HELPERS
# =========================================================================


def _detect_endian_and_natom(coor_filepath: str | Path) -> tuple[str, int]:
    """
    Detect the byte order of a NAMD binary .coor file and return the atom count.

    Strategy
    --------
    Read the first 4 bytes as an int32 in little-endian and big-endian.
    Cross-check each candidate atom count against the actual file size:

        expected_bytes = 4 + N * 3 * 8

    Whichever candidate is consistent with the file size wins.  On a tie
    (e.g. N == 0, which should never happen for a valid file) little-endian
    is preferred.

    Parameters
    ----------
    coor_filepath : str | Path

    Returns
    -------
    (endian, natom) where endian is '<' or '>'
    """

    coor_filepath = Path(coor_filepath)
    file_size = coor_filepath.stat().st_size

    if file_size < 8:
        raise ValueError(
            f"File is too small to be a valid NAMD binary .coor file: {coor_filepath}"
        )

    with open(coor_filepath, "rb") as f:
        header_bytes = f.read(4)

    natom_le = int(np.frombuffer(header_bytes, dtype="<i4")[0])
    natom_be = int(np.frombuffer(header_bytes, dtype=">i4")[0])

    expected_le = 4 + natom_le * 3 * 8
    expected_be = 4 + natom_be * 3 * 8

    le_ok = (natom_le > 0) and (expected_le == file_size)
    be_ok = (natom_be > 0) and (expected_be == file_size)

    if le_ok and not be_ok:
        return "<", natom_le
    if be_ok and not le_ok:
        return ">", natom_be
    if le_ok and be_ok:
        # Extremely degenerate — both pass; prefer little-endian (most common)
        return "<", natom_le

    # Neither endian matched cleanly — warn and fall back
    import warnings
    warnings.warn(
        f"Could not determine endianness of {coor_filepath} from file size "
        f"(size={file_size}, N_le={natom_le}, N_be={natom_be}). "
        "Falling back to little-endian; coordinates may be wrong.",
        RuntimeWarning,
        stacklevel=3,
    )
    return "<", max(natom_le, 1)


def _read_coor_header_metadata(coor_filepath: str | Path) -> tuple[str, int]:
    """
    Return (endian, natom) for a NAMD binary .coor file without reading
    the coordinate payload.

    Parameters
    ----------
    coor_filepath : str | Path

    Returns
    -------
    (endian, natom)
    """
    return _detect_endian_and_natom(coor_filepath)


def _read_coor_positions(
    coor_filepath: str | Path,
    global_ids: list[int] | np.ndarray,
    ) -> np.ndarray:
    
    """
    Read all atom positions from a NAMD binary .coor file and return the
    subset selected by *global_ids*.

    Parameters
    ----------
    coor_filepath : str | Path
    global_ids : array-like of int
        Zero-based atom indices to extract (same convention as the rest of
        trajectory_kit).

    Returns
    -------
    positions : np.ndarray, shape (1, len(global_ids), 3), dtype float32
        Positions in Angstroms wrapped in a single-frame axis so the output
        is drop-in compatible with the (n_frames, n_atoms, 3) convention
        used by _read_dcd_positions_timeline.
    """

    coor_filepath = Path(coor_filepath)
    gids = np.asarray(global_ids, dtype=np.int64)

    if gids.ndim != 1:
        raise ValueError("global_ids must be a 1-D array.")

    endian, natom = _read_coor_header_metadata(coor_filepath)

    if gids.size > 0:
        if gids.min() < 0 or gids.max() >= natom:
            raise IndexError(
                f"global_ids out of bounds: file has {natom} atoms, "
                f"but requested indices span [{gids.min()}, {gids.max()}]."
            )

    # Read the full coordinate block (natom * 3 float64 values)
    # Layout: 8 bytes atom-count header, then natom * 3 * 8 bytes of XYZ
    with open(coor_filepath, "rb") as f:
        f.seek(4)  # skip the atom-count header
        raw = f.read(natom * 3 * 8)

    if len(raw) != natom * 3 * 8:
        raise EOFError(
            f"Unexpected EOF reading coordinates from {coor_filepath}: "
            f"expected {natom * 3 * 8} bytes, got {len(raw)}."
        )

    # Interpret as (natom, 3) double array, then cast to float32 for
    # consistency with the rest of trajectory_kit (DCD also uses float32)
    all_xyz = np.frombuffer(raw, dtype=endian + "f8").reshape(natom, 3)
    selected_xyz = all_xyz[gids].astype(np.float32)

    # Wrap in a (1, n_atoms_selected, 3) array — single-frame trajectory
    return selected_xyz[np.newaxis, :, :]


# =========================================================================
# MANDATORY TRAJECTORY INTERFACE  (required by trajectory_kit)
# =========================================================================


def _get_trajectory_keys_reqs_coor(coor_filepath: str | Path):
    """
    Return the available keywords and requests for a NAMD .coor file.

    frame_interval is intentionally absent — .coor is a single-frame
    format with no notion of time.  If the caller supplies frame_interval
    in the query dictionary it will be caught at query time and a warning
    will be printed; the single frame is always returned regardless.
    """

    set_of_keywords = {
        "global_ids",
    }

    set_of_requests = {
        "positions",
        "global_ids",
    }

    return set_of_keywords, set_of_requests


def _plan_trajectory_query_coor(
    coor_filepath: str | Path,
    query_dictionary: dict,
    request_string: str,
    keywords_available: set,
    requests_available: set,
):
    """
    Return a lightweight plan dict describing what _get_trajectory_query_coor
    would produce, without actually reading the coordinate payload.

    If frame_interval is present in the query dictionary a warning is printed
    and it is ignored — .coor always contains exactly one frame.
    """

    if request_string not in requests_available:
        raise ValueError(
            f"Unsupported request_string {request_string!r}. "
            f"Available requests: {sorted(requests_available)}"
        )

    if "frame_interval" in query_dictionary:
        print(
            "Warning [coor]: frame_interval was passed but .coor is a "
            "single-frame format — it will be ignored and the single "
            "available frame will be returned."
        )

    _, natom = _read_coor_header_metadata(coor_filepath)

    # .coor always contains exactly 1 frame
    return {
        "planner_mode": "header",
        "n_atoms":      natom,
        "n_frames":     1,
    }


def _get_trajectory_query_coor(
    coor_filepath: str | Path,
    query_dictionary: dict,
    request_string: str,
    keywords_available: set,
    requests_available: set,
):
    """
    Execute a trajectory query on a NAMD binary .coor file.

    Supported request strings
    -------------------------
    "positions"
        Returns positions as an ndarray of shape
        (1, n_atoms_selected, 3) in Angstroms (float32).

    frame_interval handling
    -----------------------
    If frame_interval is present in the query dictionary a warning is printed
    and the key is ignored.  The single available frame is always returned.
    """

    if "frame_interval" in query_dictionary:
        print(
            "Warning [coor]: frame_interval was passed but .coor is a "
            "single-frame format — it will be ignored and the single "
            "available frame will be returned."
        )

    match request_string:

        case "global_ids":
            # COOR has no per-atom properties to filter on.
            # Return None to signal "no constraint from this domain" —
            # fetch() treats None as a hotpath meaning atom selection is
            # fully delegated to the typing / topology domain.
            return None

        case "positions":
            return _read_coor_positions(
                coor_filepath=coor_filepath,
                global_ids=query_dictionary["global_ids"][0],
            )

        case _:
            raise ValueError(
                f"Unsupported request_string for COOR: {request_string!r}"
            )


def _update_trajectory_globals_coor(coor_filepath: str | Path) -> dict:
    """
    Extract global system properties from a NAMD binary .coor file.

    Reads only the 8-byte atom-count header — O(1) regardless of system size.

    Returns
    -------
    dict
        Keys: "num_atoms", "num_frames"  (always 1 frame for a .coor file).
    """

    try:
        _, natom = _read_coor_header_metadata(coor_filepath)
    except Exception:
        return {}

    return {
        "num_atoms":  natom,
        "num_frames": 1,       # .coor is a single-snapshot file
    }


# =========================================================================
# ADDITIONAL / OPTIONAL INTERFACE  (mirrors dcd_parse extras)
# =========================================================================


def _get_trajectory_plan_shape_coor(
    request_string: str,
) -> tuple[str, tuple, int | None]:
    """
    Return (output_kind, trailing_shape, bytes_per_entry) for planner use.

    Matches the same contract as _get_trajectory_plan_shape_dcd.
    """

    match request_string:
        case "positions":
            return "per_atom_per_frame", (3,), 12   # 3 × float32
        case "global_ids":
            return "selector", (), None
        case _:
            raise ValueError(
                f"Unsupported request_string for coor planner: {request_string!r}"
            )