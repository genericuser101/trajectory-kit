# preloaded imports
from pathlib import Path

# mandatory imports
import numpy as np

# local imports
from trajectory_kit import _file_parse_help as fph
from trajectory_kit import _query_help as qh


# -------------------------------------------------------------------------
# TRAJECTORY PARSING
# -------------------------------------------------------------------------


def _update_trajectory_globals_dcd(dcd_filepath: str | Path) -> dict:
 
    '''
    Extract global system properties from a DCD trajectory file.
 
    Reads only the fixed-size DCD header — O(1) regardless of trajectory length.
 
    Parameters:
    ----------
    dcd_filepath: str | Path
        The file path to the DCD file.
 
    Returns:
    -------
    dict
        A dictionary of global system properties with keys matching
        ``sim.global_system_properties``.
    '''
 
    try:
        nset, natom, _ = _read_dcd_header_metadata(dcd_filepath)
    except Exception:
        return {}
 
    return {
        "num_atoms":  natom,
        "num_frames": nset,
    }


def _get_trajectory_keys_reqs_dcd(dcd_filepath: str | Path):

    set_of_keywords = {
        "global_ids",
        "frame_interval",
    }

    set_of_requests = {
        "positions",
        "global_ids",
    }

    return set_of_keywords, set_of_requests


def _plan_trajectory_query_dcd(
    dcd_filepath: str | Path,
    query_dictionary: dict,
    request_string: str,
    keywords_available: set,
    requests_available: set,
    ):

    if request_string not in requests_available:
        raise ValueError(
            f"Unsupported request_string {request_string!r}. "
            f"Available requests: {sorted(requests_available)}"
        )

    start, stop, step = _resolve_dcd_frame_interval(
        query_dictionary.get("frame_interval", ())
    )

    nset, natom, has_unitcell = _read_dcd_header_metadata(dcd_filepath)

    if stop is None or stop > nset:
        stop = nset

    if start < 0:
        raise ValueError("start must be >= 0.")
    if start >= nset:
        raise ValueError("start beyond total frames.")
    if stop <= start:
        raise ValueError("stop must be > start.")

    n_frames_selected = (stop - start + step - 1) // step

    return {
        "planner_mode": "header",
        "n_atoms":      natom,
        "n_frames":     n_frames_selected,
    }


def _get_trajectory_query_dcd(
    dcd_filepath: str | Path,
    query_dictionary: dict,
    request_string: str,
    keywords_available: set,
    requests_available: set,
    ):

    match request_string:

        case "global_ids":
            # DCD has no per-frame per-atom properties to filter on.
            # Return None to signal "no constraint from this domain" —
            # fetch() treats None as a hotpath meaning atom selection is
            # fully delegated to the typing / topology domain.
            return None

        case "positions":

            start, stop, step = _resolve_dcd_frame_interval(
                query_dictionary.get("frame_interval", ())
            )

            return _read_dcd_positions_timeline(
                dcd_filepath=dcd_filepath,
                global_ids=query_dictionary["global_ids"][0],
                frame_interval=(start, stop, step),
                return_box=False,
            )

        case _:
            raise ValueError(f"Unsupported request_string for DCD: {request_string!r}")


# -------------------------------------------------------------------------
# ADDITIONAL FUNCTIONALITY
# -------------------------------------------------------------------------


def _get_trajectory_plan_shape_dcd(request_string: str) -> tuple[str, tuple, int | None]:

    '''
    Return output_kind, trailing_shape, and bytes_per_entry for planner use.

    Returns
    -------
    tuple[str, tuple, int | None]
        output_kind: "per_atom_per_frame" or "per_frame" or "scalar"
        trailing_shape: shape beyond (n_frames, n_atoms)
        bytes_per_entry: bytes per atom per frame, None for non-numeric
    '''

    match request_string:
        case "positions":
            return "per_atom_per_frame", (3,), 12   # 3 x float32
        case "global_ids":
            return "selector", (), None
        case _:
            raise ValueError(f"Unsupported request_string for dcd planner: {request_string!r}")


def _resolve_dcd_frame_interval(frame_inc) -> tuple[int, int | None, int]:
    """Thin wrapper — delegates to the shared fph.resolve_frame_interval."""
    return fph.resolve_frame_interval(frame_inc)


def _read_dcd_header_metadata(dcd_filepath: str | Path) -> tuple[int, int, bool]:

    dcd_filepath = Path(dcd_filepath)

    def _read_i32(f, endian):
        return int(np.frombuffer(f.read(4), dtype=endian + "i4")[0])

    def _read_record(f, endian):
        n = _read_i32(f, endian)
        payload = f.read(n)
        if len(payload) != n:
            raise EOFError("Unexpected EOF in DCD.")
        if _read_i32(f, endian) != n:
            raise ValueError("DCD record length mismatch.")
        return payload

    with open(dcd_filepath, "rb") as f:
        first4 = f.read(4)
        if len(first4) != 4:
            raise ValueError("Invalid DCD file.")
        f.seek(0)

        if int(np.frombuffer(first4, dtype="<i4")[0]) == 84:
            endian = "<"
        elif int(np.frombuffer(first4, dtype=">i4")[0]) == 84:
            endian = ">"
        else:
            try:
                _ = _read_record(f, "<")
                endian = "<"
            except Exception:
                f.seek(0)
                _ = _read_record(f, ">")
                endian = ">"

        hdr = _read_record(f, endian)
        icntrl = np.frombuffer(hdr[4:4 + 20 * 4], dtype=endian + "i4")
        nset = int(icntrl[0])
        has_unitcell = bool(icntrl[10])

        _ = _read_record(f, endian)
        natom_rec = _read_record(f, endian)
        natom = int(np.frombuffer(natom_rec[:4], dtype=endian + "i4")[0])

    return nset, natom, has_unitcell


def _read_dcd_positions_timeline(
    dcd_filepath: str | Path,
    global_ids: list[int] | np.ndarray,
    frame_interval: tuple[int | None, int | None, int] = (None, None, 1),
    *,
    return_box: bool = False,
    ):

    dcd_filepath = Path(dcd_filepath)

    gids = np.asarray(global_ids, dtype=np.int32)
    if gids.ndim != 1:
        raise ValueError("global_ids must be 1D.")

    start, stop, step = frame_interval

    if step is None:
        step = 1
    if step < 1:
        raise ValueError("step must be >= 1.")
    if start is None:
        start = 0

    def _read_i32(f, endian):
        return int(np.frombuffer(f.read(4), dtype=endian + "i4")[0])

    def _read_record(f, endian):
        n = _read_i32(f, endian)
        payload = f.read(n)
        if len(payload) != n:
            raise EOFError("Unexpected EOF in DCD.")
        if _read_i32(f, endian) != n:
            raise ValueError("DCD record length mismatch.")
        return payload

    with open(dcd_filepath, "rb") as f:
        first4 = f.read(4)
        if len(first4) != 4:
            raise ValueError("Invalid DCD file.")
        f.seek(0)

        if int(np.frombuffer(first4, dtype="<i4")[0]) == 84:
            endian = "<"
        elif int(np.frombuffer(first4, dtype=">i4")[0]) == 84:
            endian = ">"
        else:
            try:
                _ = _read_record(f, "<")
                endian = "<"
            except Exception:
                f.seek(0)
                _ = _read_record(f, ">")
                endian = ">"

        hdr = _read_record(f, endian)
        icntrl = np.frombuffer(hdr[4:4 + 20 * 4], dtype=endian + "i4")
        nset = int(icntrl[0])
        namnf = int(icntrl[8])
        has_unitcell = bool(icntrl[10])

        if namnf != 0:
            raise NotImplementedError("Fixed atoms not supported.")

        _ = _read_record(f, endian)
        natom_rec = _read_record(f, endian)
        natom = int(np.frombuffer(natom_rec[:4], dtype=endian + "i4")[0])

        if gids.min(initial=0) < 0 or gids.max(initial=-1) >= natom:
            raise IndexError("global_ids out of bounds.")

        if stop is None or stop > nset:
            stop = nset

        if start < 0:
            raise ValueError("start must be >= 0.")
        if start >= nset:
            raise ValueError("start beyond total frames.")
        if stop <= start:
            raise ValueError("stop must be > start.")

        n_keep = (stop - start + step - 1) // step
        pos_out = np.empty((n_keep, len(gids), 3), dtype=np.float32)

        out_i = 0

        for frame_i in range(nset):
            if frame_i < start:
                skip = True
            elif frame_i >= stop:
                break
            elif (frame_i - start) % step != 0:
                skip = True
            else:
                skip = False

            if has_unitcell:
                _ = _read_record(f, endian)

            xrec = _read_record(f, endian)
            yrec = _read_record(f, endian)
            zrec = _read_record(f, endian)

            if skip:
                continue

            x = np.frombuffer(xrec, dtype=endian + "f4", count=natom)
            y = np.frombuffer(yrec, dtype=endian + "f4", count=natom)
            z = np.frombuffer(zrec, dtype=endian + "f4", count=natom)

            pos_out[out_i, :, 0] = x[gids]
            pos_out[out_i, :, 1] = y[gids]
            pos_out[out_i, :, 2] = z[gids]

            out_i += 1

        return pos_out[:out_i]