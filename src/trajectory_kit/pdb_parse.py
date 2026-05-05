# preloaded imports
from pathlib import Path

# mandatory imports
import numpy as np

# local imports
from trajectory_kit import _file_parse_help as fph
from trajectory_kit import _query_help as qh


# -------------------------------------------------------------------------
# TYPE PARSING
# -------------------------------------------------------------------------

def _update_type_globals_pdb(pdb_filepath: str | Path) -> dict:
 
    '''
    Extract global system properties from a PDB typing file.
 
    Makes a single pass over the file to count atoms, collect unique residue
    IDs, and compute the bounding box for ``start_box_size``.
 
    Parameters:
    ----------
    pdb_filepath: str | Path
        The file path to the PDB file.
 
    Returns:
    -------
    dict
        A dictionary of global system properties with keys matching
        ``sim.global_system_properties``.
    '''
 
    num_atoms   = 0
    residue_ids = set()
    xs, ys, zs  = [], [], []
 
    with open(pdb_filepath, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            record = line[:6].strip()
            if record in ('ATOM', 'HETATM'):
                num_atoms += 1
                try:
                    residue_ids.add(int(line[22:26]))
                except ValueError:
                    pass
                try:
                    xs.append(float(line[30:38]))
                    ys.append(float(line[38:46]))
                    zs.append(float(line[46:54]))
                except ValueError:
                    pass
 
    result = {}
    if num_atoms:
        result['num_atoms']    = num_atoms
        result['num_residues'] = len(residue_ids)
 
    if xs:
        result['start_box_size'] = (
            min(xs), max(xs),
            min(ys), max(ys),
            min(zs), max(zs),
        )
 
    return result


def _get_type_keys_reqs_pdb(pdb_filepath: str | Path):

    '''
    This function returns available keywords which can be called on in the pdb file.

    Parameters:
    ----------
    pdb_filepath: str | Path
        The file path to the pdb file. This is required and should be a .pdb
    
    Returns:
    -------
    set_of_keywords: set         
        A set of keywords that can be called on in the pdb file.
    set_of_requests : set         
        A set of requests that can be made on the pdb file.

    Raises:
    ------
    None
    '''

    # Queryable keywords
    set_of_keywords = {
        'global_ids',
        'local_ids',
        'atom_name',
        'residue_name',
        'residue_ids',
        'segment_name',
        'x',
        'y',
        'z',
        'temperature_coeff',
        'occupancy',
    }

    # Requestable keywords
    set_of_requests = {
        'global_ids',
        'local_ids',
        'residue_ids',
        'atom_names',
        'residue_names',
        'segment_names',
        'x',
        'y',
        'z',
        'temperature_coeff',
        'occupancy',
        'positions',
        'property-box_size',
        'property-number_of_atoms', 
        'property-number_of_residues', 
        'property-number_of_segments', 
    }
    
    return set_of_keywords, set_of_requests


def _plan_type_query_pdb(
    pdb_filepath: str | Path,
    query_dictionary: dict,
    request_string: str,
    keywords_available: set,
    requests_available: set, 
    ):

    '''
    Create a stochastic execution plan for a PDB type query.

    The planner samples physical file lines, filters ATOM/HETATM records,
    evaluates the normal PDB atom-selection predicate on sampled eligible rows,
    and returns approximate payload metadata.

    Parameters:
    ----------
    pdb_filepath: str | Path
        The file path to the PDB file. This is required and should be a .pdb
    query_dictionary: dict
        A dictionary querying the PDB file.
    request_string: str
        A string specifying the requested output, e.g. 'global_ids', 'positions', or 'property-number_of_atoms'.
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
        On an unsupported request_string or if required query keywords are missing from query_dictionary.
    '''

    if request_string not in requests_available:
        raise ValueError(
            f'Unsupported request_string {request_string!r}. '
            f'Available requests: {sorted(requests_available)}'
        )

    output_kind, trailing_shape, bytes_per_match = _get_type_plan_shape_pdb(request_string)

    # Scalar property requests are short-circuited centrally by
    # _plan_domain_request via the plan_shape function, so they should never
    # reach this planner. The check below is defensive only.
    if output_kind == 'scalar_property':
        return {
            'planner_mode': 'stochastic',
            'supported':    False,
            'reason': (
                f'Stochastic payload estimation is only implemented for per-atom '
                f'requests. Request {request_string!r} is a scalar/system property.'
            ),
        }

    predicate_state = _get_pdb_type_predicate_state(query_dictionary)

    sample_info = fph.iter_records_sample(
        pdb_filepath,
        record_pred=lambda line: line[0:6] in ('ATOM  ', 'HETATM'),
        parse_row=_parse_pdb_atom_row,
        start_index=0,
        encoding='ascii',
        errors='replace',
        target_sample_size=3000,
        rng_seed=42,
    )

    sampled_records = sample_info['sampled_records']
    n_total_lines = sample_info['number_of_lines']
    n_sampled_lines = sample_info['number_of_sampled_lines']
    n_sampled_eligible = sample_info['number_of_sampled_eligible_records']

    n_matching_sampled = sum(
        1 for atom in sampled_records
        if _pdb_atom_matches_query(atom, predicate_state)
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

    if n_sampled_eligible == 0:
        confidence = 'none'
    elif n_matching_sampled < 10:
        confidence = 'low'
    elif n_matching_sampled < 100:
        confidence = 'medium'
    else:
        confidence = 'high'

    return {
        "planner_mode":     "stochastic",
        "n_lines_sampled":  n_sampled_lines,
        "n_lines_eligible": n_sampled_eligible,
        "n_lines_matching": n_matching_sampled,
        "n_atoms":          estimated_matches_int,
        "n_frames":         1,
        "confidence":       confidence,
    }


def _get_type_query_pdb(
    pdb_filepath,
    query_dictionary,
    request_string,
    keywords_available,
    requests_available,
    ):

    '''
    Execute a query against a PDB typing file and return the requested output.
 
    Parameters:
    ----------
    pdb_filepath: str | Path
        The file path to the pdb file.
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
        requests return a scalar or tuple. 'positions' returns
        np.ndarray of shape (1, n_matched, 3) float32.
 
    Raises:
    ------
    NotImplementedError
        If the request_string is recognised but not yet implemented.
    ValueError
        If the request_string is not supported.
    '''
 
    def _atoms_iter():
        return fph.iter_records(
            pdb_filepath,
            mode='predicate',
            record_pred=lambda line: line[0:6] in ('ATOM  ', 'HETATM'),
            parse_row=_parse_pdb_atom_row,
            start_index=0,
            encoding='ascii',
            errors='replace',
        )
 
    match request_string:
 
        case 'global_ids':
            predicate_state = _get_pdb_type_predicate_state(query_dictionary)
            return [
                atom['global_id']
                for atom in _atoms_iter()
                if _pdb_atom_matches_query(atom, predicate_state)
            ]
 
        case 'local_ids':
            predicate_state = _get_pdb_type_predicate_state(query_dictionary)
            return [
                atom['local_id']
                for atom in _atoms_iter()
                if _pdb_atom_matches_query(atom, predicate_state)
            ]
 
        case 'residue_ids':
            predicate_state = _get_pdb_type_predicate_state(query_dictionary)
            return [
                atom['residue_id']
                for atom in _atoms_iter()
                if _pdb_atom_matches_query(atom, predicate_state)
            ]
 
        case 'atom_names':
            predicate_state = _get_pdb_type_predicate_state(query_dictionary)
            return [
                atom['atom_name']
                for atom in _atoms_iter()
                if _pdb_atom_matches_query(atom, predicate_state)
            ]
 
        case 'residue_names':
            predicate_state = _get_pdb_type_predicate_state(query_dictionary)
            return [
                atom['residue_name']
                for atom in _atoms_iter()
                if _pdb_atom_matches_query(atom, predicate_state)
            ]
 
        case 'segment_names':
            predicate_state = _get_pdb_type_predicate_state(query_dictionary)
            return [
                atom['segment_name']
                for atom in _atoms_iter()
                if _pdb_atom_matches_query(atom, predicate_state)
            ]
 
        case 'x':
            predicate_state = _get_pdb_type_predicate_state(query_dictionary)
            return [
                atom['x']
                for atom in _atoms_iter()
                if _pdb_atom_matches_query(atom, predicate_state)
            ]
 
        case 'y':
            predicate_state = _get_pdb_type_predicate_state(query_dictionary)
            return [
                atom['y']
                for atom in _atoms_iter()
                if _pdb_atom_matches_query(atom, predicate_state)
            ]
 
        case 'z':
            predicate_state = _get_pdb_type_predicate_state(query_dictionary)
            return [
                atom['z']
                for atom in _atoms_iter()
                if _pdb_atom_matches_query(atom, predicate_state)
            ]
 
        case 'occupancy':
            predicate_state = _get_pdb_type_predicate_state(query_dictionary)
            return [
                atom['occupancy']
                for atom in _atoms_iter()
                if _pdb_atom_matches_query(atom, predicate_state)
            ]
 
        case 'temperature_coeff':
            predicate_state = _get_pdb_type_predicate_state(query_dictionary)
            return [
                atom['temperature_coeff']
                for atom in _atoms_iter()
                if _pdb_atom_matches_query(atom, predicate_state)
            ]

        case 'positions':
            predicate_state = _get_pdb_type_predicate_state(query_dictionary)
            rows = [
                (atom['x'], atom['y'], atom['z'])
                for atom in _atoms_iter()
                if _pdb_atom_matches_query(atom, predicate_state)
            ]
            arr = np.array(rows, dtype=np.float32).reshape(-1, 3)  # (n, 3) safe when empty
            return arr[np.newaxis, :, :]                             # (1, n, 3)
 
        case 'property-number_of_atoms':
            return sum(1 for _ in _atoms_iter())
 
        case 'property-number_of_residues':
            return len({atom['residue_id'] for atom in _atoms_iter()})
 
        case 'property-number_of_segments':
            return len({atom['segment_name'] for atom in _atoms_iter()})
 
        case 'property-box_size':
            xs, ys, zs = [], [], []
            for atom in _atoms_iter():
                xs.append(atom['x'])
                ys.append(atom['y'])
                zs.append(atom['z'])
            if not xs:
                raise ValueError('No ATOM/HETATM records found in PDB file.')
            return (
                min(xs), max(xs),
                min(ys), max(ys),
                min(zs), max(zs),
            )
 
        case _:
            raise ValueError(f'Unsupported request_string for PDB: {request_string!r}')



# -------------------------------------------------------------------------
# TOPOLOGY PARSING
# -------------------------------------------------------------------------

def _update_topology_globals_pdb(pdb_filepath: str | Path) -> dict:
    """
    Extract global system properties from a PDB topology file.

    PDB topology data is carried by ATOM/HETATM rows plus optional CONECT
    records. Atom/residue counts are always reported; bond availability is
    surfaced as format-specific metadata by the standardiser.
    """

    return _update_type_globals_pdb(pdb_filepath)


def _get_topology_keys_reqs_pdb(pdb_filepath: str | Path):
    """
    Return available topology query keywords and requests for a PDB file.

    ``bonded_with`` and ``bonded_with_mode`` are exposed only when the file
    contains PDB CONECT records. Without CONECT data there is no reliable bond
    graph to query, so the normal query validator should reject bond filters.
    """

    set_of_keywords = {
        "global_ids",
        "local_ids",
        "atom_name",
        "residue_name",
        "residue_ids",
        "segment_name",
        "x",
        "y",
        "z",
        "temperature_coeff",
        "occupancy",
    }

    has_connect = _pdb_has_connect_records(pdb_filepath)
    if has_connect:
        set_of_keywords.add("bonded_with")
        set_of_keywords.add("bonded_with_mode")

    set_of_requests = {
        "global_ids",
        "local_ids",
        "residue_ids",
        "atom_names",
        "residue_names",
        "segment_names",
        "x",
        "y",
        "z",
        "temperature_coeff",
        "positions",
        "occupancy",
        "property-box_size",
        "property-number_of_atoms",
        "property-number_of_residues",
        "property-number_of_segments",
    }

    if has_connect:
        set_of_requests.add("bonds_with")

    return set_of_keywords, set_of_requests


def _plan_topology_query_pdb(
    pdb_filepath: str | Path,
    query_dictionary: dict,
    request_string: str,
    keywords_available: set,
    requests_available: set,
    ):
    """
    Stochastic planner for PDB topology atom-selection prevalence.

    As in the PSF planner, bonded_with constraints are not estimated because
    they require resolving the full bond graph. Direct atom predicates are
    estimated by sampling ATOM/HETATM records.
    """

    if request_string not in requests_available:
        raise ValueError(
            f"Unsupported request_string {request_string!r}. "
            f"Available requests: {sorted(requests_available)}"
        )

    output_kind, trailing_shape, bytes_per_match = _get_topology_plan_shape_pdb(request_string)

    if output_kind != "per_atom":
        return {
            "planner_mode": "stochastic",
            "supported": False,
            "reason": (
                f"Stochastic payload estimation is only implemented for per-atom "
                f"requests. Request {request_string!r} is not a per-atom payload."
            ),
        }

    bond_inc, bond_exc = qh._normalise_bonded_with_pair(query_dictionary.get("bonded_with"))
    if bond_inc or bond_exc:
        return {
            "planner_mode": "stochastic",
            "supported": False,
            "reason": (
                "PDB stochastic planning does not estimate bonded_with constraints. "
                "Only direct atom-selection prevalence is estimated."
            ),
        }

    predicate_state = _get_pdb_type_predicate_state(query_dictionary)

    sample_info = fph.iter_records_sample(
        pdb_filepath,
        record_pred=lambda line: line[0:6] in ("ATOM  ", "HETATM"),
        parse_row=_parse_pdb_atom_row,
        start_index=0,
        encoding="ascii",
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
        if _pdb_atom_matches_query(atom, predicate_state)
    )

    eligible_line_fraction = (
        n_sampled_eligible / n_sampled_lines
        if n_sampled_lines > 0 else 0.0
    )
    matching_fraction_given_eligible = (
        n_matching_sampled / n_sampled_eligible
        if n_sampled_eligible > 0 else 0.0
    )

    estimated_matches = n_total_lines * eligible_line_fraction * matching_fraction_given_eligible
    estimated_matches_int = int(round(estimated_matches))

    if n_sampled_eligible == 0:
        confidence = "none"
    elif n_matching_sampled < 10:
        confidence = "low"
    elif n_matching_sampled < 100:
        confidence = "medium"
    else:
        confidence = "high"

    return {
        "planner_mode": "stochastic",
        "n_lines_sampled": n_sampled_lines,
        "n_lines_eligible": n_sampled_eligible,
        "n_lines_matching": n_matching_sampled,
        "n_atoms": estimated_matches_int,
        "n_frames": 1,
        "confidence": confidence,
    }


def _get_topology_query_pdb(
    pdb_filepath: str | Path,
    query_dictionary: dict,
    request_string: str,
    keywords_available: set,
    requests_available: set,
    *,
    _bonded_with_depth: int = 0,
    _neighbor_cache: dict | None = None,
    ):
    """
    Execute a query against a PDB topology file.

    The recursive ``bonded_with`` implementation mirrors the PSF topology
    parser, but the bond graph is built from PDB CONECT records.
    """

    if _neighbor_cache is None:
        _neighbor_cache = {}

    def _atoms_iter():
        return fph.iter_records(
            pdb_filepath,
            mode="predicate",
            record_pred=lambda line: line[0:6] in ("ATOM  ", "HETATM"),
            parse_row=_parse_pdb_atom_row,
            start_index=0,
            encoding="ascii",
            errors="replace",
        )

    def _resolve_global_ids_with_bonds(base_ids: list[int]) -> list[int]:
        bond_inc, bond_exc = qh._normalise_bonded_with_pair(query_dictionary.get("bonded_with"))
        bond_mode, _ = query_dictionary.get("bonded_with_mode", ("all", None))

        if not bond_inc and not bond_exc:
            return base_ids

        if "bonded_with" not in keywords_available:
            raise ValueError(
                "This PDB file has no CONECT records, so bonded_with is not available."
            )

        def _resolve_neighbor_set(block: dict) -> set[int] | None:
            if block.get("total", False):
                return None
            neighbor_q = block.get("neighbor")
            if not isinstance(neighbor_q, dict):
                raise ValueError(
                    "bonded_with block must include 'neighbor' dict unless total=True."
                )

            next_depth = _bonded_with_depth + 1
            if next_depth > qh.MAX_BONDED_WITH_DEPTH:
                raise RecursionError(
                    f"bonded_with neighbour recursion exceeded max depth of "
                    f"{qh.MAX_BONDED_WITH_DEPTH}. Most realistic chemistry "
                    f"queries require depth <= 5; check your query for "
                    f"unintended deep nesting."
                )

            neighbor_q = dict(neighbor_q)
            neighbor_q.pop("bonded_with_mode", None)
            key = qh._freeze_query(neighbor_q)
            if key in _neighbor_cache:
                return _neighbor_cache[key]

            ids = _get_topology_query_pdb(
                pdb_filepath=pdb_filepath,
                query_dictionary=neighbor_q,
                request_string="global_ids",
                keywords_available=keywords_available,
                requests_available=requests_available,
                _bonded_with_depth=next_depth,
                _neighbor_cache=_neighbor_cache,
            )
            s = set(ids)
            _neighbor_cache[key] = s
            return s

        neighbor_sets_inc = [_resolve_neighbor_set(b) for b in bond_inc]
        neighbor_sets_exc = [_resolve_neighbor_set(b) for b in bond_exc]

        bond_pass_mask = _filter_pdb_by_bonded_with(
            pdb_filepath=pdb_filepath,
            candidate_globals=base_ids,
            include_blocks=bond_inc,
            exclude_blocks=bond_exc,
            mode=bond_mode,
            neighbor_sets_inc=neighbor_sets_inc,
            neighbor_sets_exc=neighbor_sets_exc,
        )

        return [g for g, ok in zip(base_ids, bond_pass_mask) if ok]

    def _matched_atoms():
        predicate_state = _get_pdb_type_predicate_state(query_dictionary)

        bond_inc, bond_exc = qh._normalise_bonded_with_pair(query_dictionary.get("bonded_with"))
        has_bond_filter = bool(bond_inc or bond_exc)

        if not has_bond_filter:
            for atom in _atoms_iter():
                if _pdb_atom_matches_query(atom, predicate_state):
                    yield atom
            return

        base_atoms = [
            atom for atom in _atoms_iter()
            if _pdb_atom_matches_query(atom, predicate_state)
        ]
        base_ids = [atom["global_id"] for atom in base_atoms]
        surviving = set(_resolve_global_ids_with_bonds(base_ids))

        for atom in base_atoms:
            if atom["global_id"] in surviving:
                yield atom

    match request_string:
        case "global_ids":
            return [atom["global_id"] for atom in _matched_atoms()]
        case "local_ids":
            return [atom["local_id"] for atom in _matched_atoms()]
        case "residue_ids":
            return [atom["residue_id"] for atom in _matched_atoms()]
        case "atom_names":
            return [atom["atom_name"] for atom in _matched_atoms()]
        case "residue_names":
            return [atom["residue_name"] for atom in _matched_atoms()]
        case "segment_names":
            return [atom["segment_name"] for atom in _matched_atoms()]
        case "x":
            return [atom["x"] for atom in _matched_atoms()]
        case "y":
            return [atom["y"] for atom in _matched_atoms()]
        case "z":
            return [atom["z"] for atom in _matched_atoms()]
        case "occupancy":
            return [atom["occupancy"] for atom in _matched_atoms()]
        case "temperature_coeff":
            return [atom["temperature_coeff"] for atom in _matched_atoms()]
        case "positions":
            rows = [
                (atom["x"], atom["y"], atom["z"])
                for atom in _matched_atoms()
            ]
            arr = np.array(rows, dtype=np.float32).reshape(-1, 3)  # (n, 3) safe when empty
            return arr[np.newaxis, :, :]                             # (1, n, 3)
        case "property-box_size":
            xs, ys, zs = [], [], []
            for atom in _atoms_iter():
                xs.append(atom['x'])
                ys.append(atom['y'])
                zs.append(atom['z'])
            if not xs:
                raise ValueError('No ATOM/HETATM records found in PDB file.')
            return (
                min(xs), max(xs),
                min(ys), max(ys),
                min(zs), max(zs),
            )
        case "property-number_of_atoms":
            return sum(1 for _ in _atoms_iter())
        case "property-number_of_residues":
            return len({atom["residue_id"] for atom in _atoms_iter()})
        case "property-number_of_segments":
            return len({atom["segment_name"] for atom in _atoms_iter()})
        case "bonds_with":
            raise NotImplementedError(
                "bonds_with payload return is not yet implemented. "
                "Use bonded_with in the query_dictionary to filter atoms by bond graph."
            )
        case _:
            raise ValueError(f"Unsupported request_string for PDB topology: {request_string!r}")


# -------------------------------------------------------------------------
# ADDITIONAL FUNCTIONALITY
# -------------------------------------------------------------------------


def _get_topology_plan_shape_pdb(request_string: str) -> tuple[str, tuple | None, int | None]:
    """
    Return output_kind, trailing_shape, and bytes_per_match for PDB topology.
    """

    match request_string:
        case "global_ids":
            return "per_atom", (), 8
        case "local_ids":
            return "per_atom", (), 8
        case "residue_ids":
            return "per_atom", (), 8
        case "atom_names":
            return "per_atom", (), 16
        case "residue_names":
            return "per_atom", (), 8
        case "segment_names":
            return "per_atom", (), 8
        case "x":
            return "per_atom", (), 8
        case "y":
            return "per_atom", (), 8
        case "z":
            return "per_atom", (), 8
        case "occupancy":
            return "per_atom", (), 8
        case "temperature_coeff":
            return "per_atom", (), 8
        case "positions":
            return "per_atom", (3,), 12
        case "bonds_with":
            return "unsupported_complex", None, None
        case 'property-box_size':
            return 'scalar_property', (6,), None
        case "property-number_of_atoms":
            return "scalar_property", (), None
        case "property-number_of_residues":
            return "scalar_property", (), None
        case "property-number_of_segments":
            return "scalar_property", (), None
        case _:
            raise ValueError(f"Unsupported request_string for pdb topology planner: {request_string!r}")


def _pdb_has_connect_records(pdb_filepath: str | Path) -> bool:
    with open(pdb_filepath, "r", encoding="ascii", errors="replace") as f:
        for line in f:
            record = line[:6].strip().upper()
            if record in ("CONECT"): # we dont support CONNECT ALIAS
                return True
    return False


def _build_pdb_local_to_global_map(pdb_filepath: str | Path) -> dict[int, int]:
    local_to_global: dict[int, int] = {}
    for atom in fph.iter_records(
        pdb_filepath,
        mode="predicate",
        record_pred=lambda line: line[0:6] in ("ATOM  ", "HETATM"),
        parse_row=_parse_pdb_atom_row,
        start_index=0,
        encoding="ascii",
        errors="replace",
    ):
        local_to_global[atom["local_id"]] = atom["global_id"]
    return local_to_global


def _parse_pdb_connect_line(line: str) -> tuple[int, list[int]]:
    parts = line.split()
    if len(parts) < 2:
        raise ValueError(f"Malformed PDB CONECT record: {line!r}")
    try:
        src = int(parts[1])
        dst = [int(tok) for tok in parts[2:]]
    except ValueError as exc:
        raise ValueError(f"Malformed PDB CONECT record: {line!r}") from exc
    return src, dst


def _build_pdb_connect_adjacency(pdb_filepath: str | Path) -> list[set[int]]:
    local_to_global = _build_pdb_local_to_global_map(pdb_filepath)
    if not local_to_global:
        return []

    n_atoms = max(local_to_global.values()) + 1
    adjacency: list[set[int]] = [set() for _ in range(n_atoms)]
    seen_connect = False

    with open(pdb_filepath, "r", encoding="ascii", errors="replace") as f:
        for line in f:
            record = line[:6].strip().upper()
            if record not in ("CONECT", "CONNECT"):
                continue
            seen_connect = True
            src_local, dst_locals = _parse_pdb_connect_line(line)
            if src_local not in local_to_global:
                continue
            src_global = local_to_global[src_local]
            for dst_local in dst_locals:
                if dst_local not in local_to_global:
                    continue
                dst_global = local_to_global[dst_local]
                if dst_global == src_global:
                    continue
                adjacency[src_global].add(dst_global)
                adjacency[dst_global].add(src_global)

    if not seen_connect:
        raise ValueError("No PDB CONECT records found.")

    return adjacency


def _filter_pdb_by_bonded_with(
    pdb_filepath: str | Path,
    candidate_globals: list[int],
    include_blocks: list[dict],
    exclude_blocks: list[dict],
    mode: str,
    neighbor_sets_inc: list[set[int] | None],
    neighbor_sets_exc: list[set[int] | None],
    ) -> np.ndarray:

    if mode not in ("all", "any"):
        raise ValueError("bonded_with_mode must be 'all' or 'any'.")

    n_cand = len(candidate_globals)
    if n_cand == 0:
        return np.zeros(0, dtype=bool)

    adjacency = _build_pdb_connect_adjacency(pdb_filepath)

    def _parse_cmp(count_dict: dict) -> tuple[str, int]:
        if not isinstance(count_dict, dict) or len(count_dict) != 1:
            raise ValueError("bonded_with block 'count' must be a dict with exactly one comparator.")
        (op, v), = count_dict.items()
        if op not in ("eq", "ne", "ge", "le", "gt", "lt"):
            raise ValueError(f"Unsupported comparator: {op!r}")
        if not isinstance(v, int) or v < 0:
            raise ValueError("Comparator value must be a non-negative int.")
        return op, v

    def _cmp(x: int, op: str, v: int) -> bool:
        if op == "eq": return x == v
        if op == "ne": return x != v
        if op == "ge": return x >= v
        if op == "le": return x <= v
        if op == "gt": return x > v
        if op == "lt": return x < v
        raise ValueError(op)

    set_id_to_col: dict[int, int] = {}
    unique_sets: list[set[int]] = []

    def _col_for_set(s: set[int]) -> int:
        sid = id(s)
        col = set_id_to_col.get(sid)
        if col is None:
            col = len(unique_sets)
            set_id_to_col[sid] = col
            unique_sets.append(s)
        return col

    inc_norm: list[tuple] = []
    exc_norm: list[tuple] = []

    for block, s in zip(include_blocks, neighbor_sets_inc):
        op, val = _parse_cmp(block["count"])
        if block.get("total", False):
            inc_norm.append(("total", op, val))
        else:
            if s is None:
                raise ValueError("Internal error: include block missing neighbor set and total=False.")
            inc_norm.append(("set", _col_for_set(s), op, val))

    for block, s in zip(exclude_blocks, neighbor_sets_exc):
        op, val = _parse_cmp(block["count"])
        if block.get("total", False):
            exc_norm.append(("total", op, val))
        else:
            if s is None:
                raise ValueError("Internal error: exclude block missing neighbor set and total=False.")
            exc_norm.append(("set", _col_for_set(s), op, val))

    if not inc_norm and not exc_norm:
        return np.ones(n_cand, dtype=bool)

    counts_total = np.zeros(n_cand, dtype=np.int32)
    counts_by_set = np.zeros((n_cand, len(unique_sets)), dtype=np.int32) if unique_sets else None

    for i, g in enumerate(candidate_globals):
        if g >= len(adjacency):
            continue
        neigh = adjacency[g]
        counts_total[i] = len(neigh)
        if counts_by_set is not None:
            for col, s in enumerate(unique_sets):
                counts_by_set[i, col] = len(neigh & s)

    def _eval_one(i: int, c: tuple) -> bool:
        if c[0] == "total":
            _, op, v = c
            return _cmp(int(counts_total[i]), op, v)
        _, col, op, v = c
        return _cmp(int(counts_by_set[i, col]), op, v)

    mask = np.ones(n_cand, dtype=bool)

    if inc_norm:
        if mode == "all":
            for i in range(n_cand):
                for c in inc_norm:
                    if not _eval_one(i, c):
                        mask[i] = False
                        break
        else:
            for i in range(n_cand):
                mask[i] = any(_eval_one(i, c) for c in inc_norm)

    if exc_norm:
        for i in range(n_cand):
            if not mask[i]:
                continue
            for c in exc_norm:
                if _eval_one(i, c):
                    mask[i] = False
                    break

    return mask


def _get_type_plan_shape_pdb(request_string: str) -> tuple[str, tuple | None, int | None]:

    '''
    Return output_kind, trailing_shape, and bytes_per_match for planner use.
 
    Parameters:
    ----------
    request_string: str
        The request string to look up.
 
    Returns:
    -------
    tuple[str, tuple | None, int | None]
        output_kind: 'per_atom' or 'scalar_property'
        trailing_shape: shape of each element beyond the atom count axis
        bytes_per_match: bytes per matched atom, None for scalar properties
 
    Raises:
    ------
    ValueError
        If request_string is not a known request for PDB files.
    '''

    match request_string:
        case 'global_ids':
            return 'per_atom', (), 8
        case 'local_ids':
            return 'per_atom', (), 8
        case 'residue_ids':
            return 'per_atom', (), 8
        case 'atom_names':
            return 'per_atom', (), 16
        case 'residue_names':
            return 'per_atom', (), 8
        case 'segment_names':
            return 'per_atom', (), 8
        case 'x':
            return 'per_atom', (), 8
        case 'y':
            return 'per_atom', (), 8
        case 'z':
            return 'per_atom', (), 8
        case 'occupancy':
            return 'per_atom', (), 8
        case 'temperature_coeff':
            return 'per_atom', (), 8
        case 'positions':
            return 'per_atom', (3,), 12
        case 'property-box_size':
            return 'scalar_property', (6,), None
        case 'property-number_of_atoms':
            return 'scalar_property', (), None
        case 'property-number_of_residues':
            return 'scalar_property', (), None
        case 'property-number_of_segments':
            return 'scalar_property', (), None
        case _:
            raise ValueError(f'Unsupported request_string for pdb planner: {request_string!r}')
 

def _parse_pdb_atom_row(line: str, global_id: int) -> dict:

    '''
    Parse a PDB ATOM/HETATM line into the standard atom dictionary used by both
    exact execution and stochastic planning.

    Parameters:
    ----------
    line: str
        A line from a PDB file that starts with 'ATOM  ' or 'HETATM'.
    global_id: int
        A unique global id for this atom, assigned sequentially by the line number in the file (starting from 0).
    
    Returns:
    -------
    dict
        A dictionary with keys: global_id, local_id, atom_name, residue_name, residue_id, segment_name, x, y, z, occupancy, temperature_coeff.
    
    Raises:
    ------
    ValueError
        If the line is not long enough to parse the required fields or if the line does not start with 'ATOM  ' or 'HETATM'.
    '''

    return {
        'global_id': global_id,
        'local_id': int(line[6:11]),
        'atom_name': line[12:16].strip(),
        'residue_name': line[17:21].strip(),   # 4-char for CHARMM-like PSF/PDB
        'residue_id': int(line[22:26]),
        'segment_name': line[72:76].strip(),
        'x': float(line[30:38]),
        'y': float(line[38:46]),
        'z': float(line[46:54]),
        'occupancy': float(line[54:60]),
        'temperature_coeff': float(line[60:66]),
    }


def _get_pdb_type_predicate_state(query_dictionary: dict) -> dict:

    '''
    Precompute query components and boolean need-flags once so both the planner
    and the exact execution path use identical predicate semantics.

    Parameters:
    ----------
    query_dictionary: dict
        The query dictionary containing include/exclude sets for various atom properties.
    
    Returns:
    -------
    dict
        A dictionary containing preprocessed query components and boolean flags indicating which properties are needed for the query
    
    Raises:
    ------
    None
    '''

    atom_inc, atom_exc = qh._normalise_query_pair(query_dictionary.get('atom_name'))
    resn_inc, resn_exc = qh._normalise_query_pair(query_dictionary.get('residue_name'))
    seg_inc,  seg_exc  = qh._normalise_query_pair(query_dictionary.get('segment_name'))

    gid_inc,  gid_exc  = qh._normalise_query_pair(query_dictionary.get('global_ids'),        range_style=True)
    li_inc,   li_exc   = qh._normalise_query_pair(query_dictionary.get('local_ids'),         range_style=True)
    ri_inc,   ri_exc   = qh._normalise_query_pair(query_dictionary.get('residue_ids'),       range_style=True)
    occ_inc,  occ_exc  = qh._normalise_query_pair(query_dictionary.get('occupancy'),         range_style=True)
    temp_inc, temp_exc = qh._normalise_query_pair(query_dictionary.get('temperature_coeff'), range_style=True)

    x_inc, x_exc = qh._normalise_query_pair(query_dictionary.get('x'), range_style=True)
    y_inc, y_exc = qh._normalise_query_pair(query_dictionary.get('y'), range_style=True)
    z_inc, z_exc = qh._normalise_query_pair(query_dictionary.get('z'), range_style=True)

    return {
        'atom_inc': atom_inc,
        'atom_exc': atom_exc,
        'resn_inc': resn_inc,
        'resn_exc': resn_exc,
        'seg_inc': seg_inc,
        'seg_exc': seg_exc,
        'gid_inc': gid_inc,
        'gid_exc': gid_exc,
        'li_inc': li_inc,
        'li_exc': li_exc,
        'ri_inc': ri_inc,
        'ri_exc': ri_exc,
        'occ_inc': occ_inc,
        'occ_exc': occ_exc,
        'temp_inc': temp_inc,
        'temp_exc': temp_exc,
        'x_inc': x_inc,
        'x_exc': x_exc,
        'y_inc': y_inc,
        'y_exc': y_exc,
        'z_inc': z_inc,
        'z_exc': z_exc,
        'need_atom': bool(atom_inc or atom_exc),
        'need_resn': bool(resn_inc or resn_exc),
        'need_seg': bool(seg_inc or seg_exc),
        'need_gid': bool(gid_inc or gid_exc),
        'need_li': bool(li_inc or li_exc),
        'need_ri': bool(ri_inc or ri_exc),
        'need_occ': bool(occ_inc or occ_exc),
        'need_temp': bool(temp_inc or temp_exc),
        'need_x': bool(x_inc or x_exc),
        'need_y': bool(y_inc or y_exc),
        'need_z': bool(z_inc or z_exc),
    }


def _pdb_atom_matches_query(atom: dict, predicate_state: dict) -> bool:

    '''
    Shared PDB atom-selection predicate used by both exact query execution and stochastic planning.
    For each property, the predicate checks if the atom's property value matches the include and exclude sets/ranges specified in the query. 
    If a property is not needed for the query (as indicated by the boolean flags), it is not checked.
    
    Parameters:
    ----------
    atom: dict
        A dictionary representing an atom, with keys like 'atom_name', 'residue_name', 'x', 'occupancy', etc.
    predicate_state: dict
        A dictionary containing preprocessed query components and boolean flags indicating which properties are needed for the query.

    Returns:
    -------
    ok: bool
        A true or false value indicating whether the atom matches the query predicates.

    Raises:
    ------
    None    
    '''

    match_ = qh._match
    match_range = qh._match_range_scalar

    ok = True

    # global_ids first: most selective, exact integer match by id
    if predicate_state['need_gid']:
        ok = match_range(atom['global_id'],
                             predicate_state['gid_inc'],
                             predicate_state['gid_exc'])

    if ok and predicate_state['need_li']:
        ok = match_range(atom['local_id'],
                             predicate_state['li_inc'],
                             predicate_state['li_exc'])
    if ok and predicate_state['need_atom']:
        ok = match_(atom['atom_name'],
                    predicate_state['atom_inc'],
                    predicate_state['atom_exc'])
    if ok and predicate_state['need_resn']:
        ok = match_(atom['residue_name'],
                    predicate_state['resn_inc'],
                    predicate_state['resn_exc'])
    if ok and predicate_state['need_ri']:
        ok = match_range(atom['residue_id'],
                             predicate_state['ri_inc'],
                             predicate_state['ri_exc'])
    if ok and predicate_state['need_seg']:
        ok = match_(atom['segment_name'],
                    predicate_state['seg_inc'],
                    predicate_state['seg_exc'])

    if ok and predicate_state['need_x']:
        ok = match_range(atom['x'],
                             predicate_state['x_inc'],
                             predicate_state['x_exc'])
    if ok and predicate_state['need_y']:
        ok = match_range(atom['y'],
                             predicate_state['y_inc'],
                             predicate_state['y_exc'])
    if ok and predicate_state['need_z']:
        ok = match_range(atom['z'],
                             predicate_state['z_inc'],
                             predicate_state['z_exc'])

    if ok and predicate_state['need_occ']:
        ok = match_range(atom['occupancy'],
                             predicate_state['occ_inc'],
                             predicate_state['occ_exc'])
    if ok and predicate_state['need_temp']:
        ok = match_range(atom['temperature_coeff'],
                             predicate_state['temp_inc'],
                             predicate_state['temp_exc'])

    return ok
