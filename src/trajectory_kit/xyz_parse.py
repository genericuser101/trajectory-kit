# preloaded imports
from __future__ import annotations
from pathlib import Path

# mandatory imports
import numpy as np

# local imports
from trajectory_kit import file_parse_help as fph
from trajectory_kit import query_help as qh


# -------------------------------------------------------------------------
# TYPE PARSING
# -------------------------------------------------------------------------


def _update_type_globals_xyz(xyz_filepath: str | Path) -> dict:
 
    '''
    Extract global system properties from an XYZ file.
 
    Makes a single pass: reads the atom count from the first line, then
    iterates all atom records to compute the bounding box for
    ``start_box_size``.
 
    Parameters:
    ----------
    xyz_filepath: str | Path
        The file path to the XYZ file.
 
    Returns:
    -------
    dict
        A dictionary of global system properties with keys matching
        ``sim.global_system_properties``.
    '''
 
    with open(xyz_filepath, "r", encoding="utf-8", errors="replace") as f:
        first_line = f.readline().strip()
 
    if not first_line.isdigit():
        return {}
 
    num_atoms = int(first_line)
 
    xs, ys, zs = [], [], []
    for atom in fph.iter_records(
        xyz_filepath,
        mode="predicate",
        record_pred=_is_xyz_atom_line,
        parse_row=_parse_xyz_atom_row,
        start_index=0,
        encoding="utf-8",
        errors="replace",
    ):
        xs.append(atom["x"])
        ys.append(atom["y"])
        zs.append(atom["z"])
 
    result = {"num_atoms": num_atoms}
 
    if xs:
        result["start_box_size"] = (
            min(xs), max(xs),
            min(ys), max(ys),
            min(zs), max(zs),
        )
 
    return result


def _get_type_keys_reqs_xyz(xyz_filepath: str | Path):

    '''
    This function returns available keywords which can be called on in the xyz file.

    Parameters:
    ----------
    xyz_filepath: str | Path
        The file path to the xyz file. This is required and should be a .xyz

    Returns:
    -------
    tuple[set, set]
        A tuple containing two sets: the first is a set of keywords that can be called on in the xyz file, and the second is a set of requests that can be made on the xyz file.
    
    Raises:
    ------
    None

    '''

    # Queryable keywords
    set_of_keywords = {
        "global_ids",
        "local_ids",
        "atom_name",
        "x",
        "y",
        "z",
    }

    # Requestable keywords
    set_of_requests = {
        "global_ids",
        "local_ids",
        "atom_names",
        "x",
        "y",
        "z",
        "positions",
        "property-number_of_atoms",
        "property-box_size",
    }

    return set_of_keywords, set_of_requests


def _plan_type_query_xyz(
    xyz_filepath: str | Path,
    query_dictionary: dict,
    request_string: str,
    keywords_available: set,
    requests_available: set,
    ):

    '''
    Create a stochastic execution plan for an XYZ type query.

    The planner reads the atom count from the header, samples atom records,
    evaluates the predicate on sampled rows, and returns approximate payload
    metadata.

    Parameters:
    ----------
    xyz_filepath: str | Path
        The file path to the xyz file. This is required and should be a .xyz
    query_dictionary: dict
        A dictionary querying the XYZ file.
    request_string: str
        A string specifying the requested output, e.g. "global_ids", "positions", or "property-number_of_atoms".
    keywords_available: set
        A set of keywords that can be used in the query_dictionary for this file type.
    requests_available: set
        A set of supported request strings that can be used in request_string for this file type.
    
    Returns:
    -------
    dict
        Planner metadata with estimated selected atoms and estimated payload size.

    Raises:
    ------
    ValueError
        If the request_string is not supported or if the XYZ file format is invalid.
    NotImplementedError
        If the request_string corresponds to a request that is recognized but not yet implemented.
    
    '''

    if request_string not in requests_available:
        raise ValueError(
            f"Unsupported request_string {request_string!r}. "
            f"Available requests: {sorted(requests_available)}"
        )

    output_kind, trailing_shape, bytes_per_match = _get_xyz_request_plan_shape(request_string)

    if output_kind == "scalar_property":
        return {
            "planner_mode": "stochastic",
            "file_type": "xyz",
            "request": request_string,
            "supported": False,
            "reason": (
                f"Stochastic payload estimation is only implemented for per-atom "
                f"requests. Request {request_string!r} is a scalar/system property."
            ),
            "query_dictionary": query_dictionary,
        }

    predicate_state = _get_xyz_type_predicate_state(query_dictionary)

    sample_info = fph.iter_records_sample(
        xyz_filepath,
        record_pred=_is_xyz_atom_line,
        parse_row=_parse_xyz_atom_row,
        start_index=0,
        encoding="utf-8",
        errors="replace",
        target_sample_size=3000,
        rng_seed=42,
    )

    sampled_records = sample_info["sampled_records"]
    n_total_lines = sample_info["number_of_lines"]
    n_sampled_lines = sample_info["number_of_sampled_lines"]
    n_sampled_eligible = sample_info["number_of_sampled_eligible_records"]

    n_matching_sampled = sum(
        1 for atom in sampled_records
        if _xyz_atom_matches_query(atom, predicate_state)
    )

    eligible_line_fraction = (
        n_sampled_eligible / n_sampled_lines
        if n_sampled_lines > 0 else 0.0
    )

    matching_fraction_given_eligible = (
        n_matching_sampled / n_sampled_eligible
        if n_sampled_eligible > 0 else 0.0
    )

    estimated_eligible_records = n_total_lines * eligible_line_fraction
    estimated_matches = estimated_eligible_records * matching_fraction_given_eligible

    estimated_matches_int = int(round(estimated_matches))
    estimated_payload_bytes = int(round(estimated_matches * bytes_per_match))

    if trailing_shape == ():
        estimated_output_shape = (estimated_matches_int,)
    else:
        estimated_output_shape = (estimated_matches_int, *trailing_shape)

    if n_sampled_eligible == 0:
        confidence = "none"
    elif n_matching_sampled < 10:
        confidence = "low"
    elif n_matching_sampled < 100:
        confidence = "medium"
    else:
        confidence = "high"

    return {
        "planner_mode":     "stochastic",
        "n_lines_sampled":  n_sampled_lines,
        "n_lines_eligible": n_sampled_eligible,
        "n_lines_matching": n_matching_sampled,
        "n_atoms":          estimated_matches_int,
        "estimated_bytes":  estimated_payload_bytes,
        "estimated_mib":    estimated_payload_bytes / (1024 ** 2),
        "confidence":       confidence,
    }


def _get_type_query_xyz(
    xyz_filepath,
    query_dictionary,
    request_string,
    keywords_available,
    requests_available,
    ):

    '''
    Execute a query against an XYZ typing file and return the requested output.
 
    Standard XYZ format:
        Line 0:   <number of atoms>
        Line 1:   <comment string>
        Line 2+:  <element> <x> <y> <z>
 
    Parameters:
    ----------
    xyz_filepath: str | Path
        The file path to the xyz file.
    query_dictionary: dict
        A dictionary of query filters using the include/exclude schema.
    request_string: str
        The type of output requested.
    keywords_available: set
        The set of valid query keys for this file.
    requests_available: set
        The set of valid request strings for this file.
 
    Returns:
    -------
    list | int | tuple | np.ndarray
        Depends on request_string. Per-atom requests return list, property
        requests return a scalar or tuple. "positions" returns
        np.ndarray of shape (1, n_matched, 3) float32.
 
    Raises:
    ------
    NotImplementedError
        If the request_string is recognised but not yet implemented.
    ValueError
        If the request_string is not supported or the file format is invalid.
    '''
 
    def _atoms_iter():
        return fph.iter_records(
            xyz_filepath,
            mode="predicate",
            record_pred=_is_xyz_atom_line,
            parse_row=_parse_xyz_atom_row,
            start_index=0,
            encoding="utf-8",
            errors="replace",
        )
 
    match request_string:
 
        case "global_ids":
            predicate_state = _get_xyz_type_predicate_state(query_dictionary)
            return [
                atom["global_id"]
                for atom in _atoms_iter()
                if _xyz_atom_matches_query(atom, predicate_state)
            ]
 
        case "local_ids":
            predicate_state = _get_xyz_type_predicate_state(query_dictionary)
            return [
                atom["local_id"]
                for atom in _atoms_iter()
                if _xyz_atom_matches_query(atom, predicate_state)
            ]
 
        case "atom_names":
            predicate_state = _get_xyz_type_predicate_state(query_dictionary)
            return [
                atom["atom_name"]
                for atom in _atoms_iter()
                if _xyz_atom_matches_query(atom, predicate_state)
            ]
 
        case "x":
            predicate_state = _get_xyz_type_predicate_state(query_dictionary)
            return [
                atom["x"]
                for atom in _atoms_iter()
                if _xyz_atom_matches_query(atom, predicate_state)
            ]
 
        case "y":
            predicate_state = _get_xyz_type_predicate_state(query_dictionary)
            return [
                atom["y"]
                for atom in _atoms_iter()
                if _xyz_atom_matches_query(atom, predicate_state)
            ]
 
        case "z":
            predicate_state = _get_xyz_type_predicate_state(query_dictionary)
            return [
                atom["z"]
                for atom in _atoms_iter()
                if _xyz_atom_matches_query(atom, predicate_state)
            ]

        case "positions":
            predicate_state = _get_xyz_type_predicate_state(query_dictionary)
            rows = [
                (atom["x"], atom["y"], atom["z"])
                for atom in _atoms_iter()
                if _xyz_atom_matches_query(atom, predicate_state)
            ]
            arr = np.array(rows, dtype=np.float32).reshape(-1, 3)  # (n, 3) safe when empty
            return arr[np.newaxis, :, :]                             # (1, n, 3)
 
        case "property-number_of_atoms":
            with open(xyz_filepath, "r", encoding="utf-8", errors="replace") as f:
                first_line = f.readline().strip()
            if not first_line.isdigit():
                raise ValueError(
                    f"XYZ file does not have a valid atom count on line 1: {first_line!r}"
                )
            return int(first_line)
 
        case "property-box_size":
            xs, ys, zs = [], [], []
            for atom in _atoms_iter():
                xs.append(atom["x"])
                ys.append(atom["y"])
                zs.append(atom["z"])
            if not xs:
                raise ValueError("No atom records found in XYZ file.")
            return (
                min(xs), max(xs),
                min(ys), max(ys),
                min(zs), max(zs),
            )
 
        case _:
            raise ValueError(f"Unsupported request_string for XYZ: {request_string!r}")


# -------------------------------------------------------------------------
# ADDITIONAL FUNCTIONALITY
# -------------------------------------------------------------------------


def _get_xyz_request_plan_shape(request_string: str) -> tuple[str, tuple | None, int | None]:

    '''
    Return output_kind, trailing_shape, and bytes_per_match for planner use.
 
    Parameters:
    ----------
    request_string: str
        The request string to look up.
 
    Returns:
    -------
    tuple[str, tuple | None, int | None]
        output_kind: "per_atom" or "scalar_property"
        trailing_shape: shape of each element beyond the atom count axis
        bytes_per_match: bytes per matched atom, None for scalar properties
 
    Raises:
    ------
    ValueError
        If request_string is not a known request for XYZ files.
    '''

    match request_string:
        case "global_ids":
            return "per_atom", (), 8
        case "local_ids":
            return "per_atom", (), 8
        case "atom_names":
            return "per_atom", (), 16
        case "x":
            return "per_atom", (), 8
        case "y":
            return "per_atom", (), 8
        case "z":
            return "per_atom", (), 8
        case "positions":
            return "per_atom", (3,), 12
        case "property-number_of_atoms":
            return "scalar_property", (), None
        case "property-box_size":
            return "scalar_property", (6,), None
        case _:
            raise ValueError(f"Unsupported request_string for xyz planner: {request_string!r}")


def _is_xyz_count_line(line: str) -> bool:

    '''
    Returns True for the first line of an XYZ file, which contains only the
    integer atom count. iter_records "counted" mode uses this as header_pred.
    The comment line immediately following is consumed automatically as the
    first next(f) call inside iter_records before the atom loop begins.

    Note: iter_records counted mode reads exactly count_from_header(header)
    lines after the header. XYZ has one comment line between the count and the
    atoms, so we skip it inside count_from_header by reading it as a side
    effect via a wrapper — see _xyz_count_skip_comment below.
    '''

    stripped = line.strip()
    return stripped.isdigit() and int(stripped) > 0


def _xyz_count_skip_comment(header_line: str, f) -> int:

    '''
    Helper used when manually iterating: reads the comment line as a side
    effect and returns the atom count.
    Not used by iter_records directly — see note in _get_type_query_xyz where
    counted mode is configured with a lambda that handles this.
    '''

    _ = f.readline()   # consume comment
    return int(header_line.strip())


def _is_xyz_atom_line(line: str) -> bool:

    '''
    Predicate for iter_records_sample: returns True for lines that look like
    XYZ atom records (element x y z), used for stochastic sampling where we
    cannot use the counted header approach.
    '''

    parts = line.split()
    if len(parts) < 4:
        return False
    # first token must be a non-numeric element symbol
    if parts[0].lstrip('-').replace('.', '', 1).isdigit():
        return False
    try:
        float(parts[1])
        float(parts[2])
        float(parts[3])
        return True
    except ValueError:
        return False


def _parse_xyz_atom_row(line: str, global_id: int) -> dict:

    '''
    Parse a single XYZ atom line into the standard atom dictionary.

    XYZ column layout:
        element   x   y   z   [optional extra columns ignored]

    local_id mirrors global_id since XYZ files have no serial number field.
    '''

    parts = line.split()
    return {
        "global_id": global_id,
        "local_id": global_id,         # XYZ has no serial; use positional index
        "atom_name": parts[0].strip(),
        "x": float(parts[1]),
        "y": float(parts[2]),
        "z": float(parts[3]),
    }


def _get_xyz_type_predicate_state(query_dictionary: dict) -> dict:

    '''
    Precompute query components and boolean need-flags once so both the planner
    and the exact execution path use identical predicate semantics.
    '''
    
    atom_inc, atom_exc = qh._normalise_query_pair(query_dictionary.get("atom_name"))

    x_inc, x_exc = qh._normalise_query_pair(query_dictionary.get("x"), range_style=True)
    y_inc, y_exc = qh._normalise_query_pair(query_dictionary.get("y"), range_style=True)
    z_inc, z_exc = qh._normalise_query_pair(query_dictionary.get("z"), range_style=True)

    return {
        "atom_inc": atom_inc,
        "atom_exc": atom_exc,
        "x_inc": x_inc,
        "x_exc": x_exc,
        "y_inc": y_inc,
        "y_exc": y_exc,
        "z_inc": z_inc,
        "z_exc": z_exc,
        "need_atom": bool(atom_inc or atom_exc),
        "need_x": x_inc != (None, None) or x_exc != (None, None),
        "need_y": y_inc != (None, None) or y_exc != (None, None),
        "need_z": z_inc != (None, None) or z_exc != (None, None),
    }


def _xyz_atom_matches_query(atom: dict, predicate_state: dict) -> bool:
    
    '''
    Shared XYZ atom-selection predicate used by both exact query execution
    and stochastic planning.
    '''

    match_ = qh._match
    match_range = qh._match_range_scalar

    ok = True

    if predicate_state["need_atom"]:
        ok = match_(atom["atom_name"],
                    predicate_state["atom_inc"],
                    predicate_state["atom_exc"])

    if ok and predicate_state["need_x"]:
        ok = match_range(atom["x"],
                             predicate_state["x_inc"],
                             predicate_state["x_exc"])

    if ok and predicate_state["need_y"]:
        ok = match_range(atom["y"],
                             predicate_state["y_inc"],
                             predicate_state["y_exc"])

    if ok and predicate_state["need_z"]:
        ok = match_range(atom["z"],
                             predicate_state["z_inc"],
                             predicate_state["z_exc"])

    return ok
