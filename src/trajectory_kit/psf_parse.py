# preloaded imports
from __future__ import annotations
from pathlib import Path

# mandatory imports
import numpy as np

# local imports
from trajectory_kit import file_parse_help as fph
from trajectory_kit import query_help as qh

# -------------------------------------------------------------------------
# TOPOLOGY PARSING
# -------------------------------------------------------------------------


def _update_topology_globals_psf(psf_filepath: str | Path) -> dict:
 
    '''
    Extract global system properties from a PSF topology file.
 
    Reads the !NATOM section header to get the atom count, and makes one
    additional pass to count unique residue IDs.
 
    Parameters:
    ----------
    psf_filepath: str | Path
        The file path to the PSF file.
 
    Returns:
    -------
    dict
        A dictionary of global system properties with keys matching
        ``sim.global_system_properties``.
    '''
 
    num_atoms   = None
    residue_ids = set()
 
    with open(psf_filepath, "r", encoding="ascii", errors="replace") as f:
        for line in f:
            if "!NATOM" in line:
                try:
                    num_atoms = int(line.split()[0])
                except (ValueError, IndexError):
                    pass
                break
 
        if num_atoms is not None:
            for _ in range(num_atoms):
                atom_line = f.readline()
                parts = atom_line.split()
                if len(parts) >= 4:
                    # PSF column 3 (0-indexed) is the residue sequence number
                    try:
                        residue_ids.add(int(parts[2]))
                    except ValueError:
                        pass
 
    result = {}
    if num_atoms is not None:
        result["num_atoms"]    = num_atoms
        result["num_residues"] = len(residue_ids) if residue_ids else None
 
    return result


def _get_topology_keys_reqs_psf(psf_filepath: str | Path):

    '''
    This function retrieves the available keywords and requests for a PSF file. 
    It checks for the presence of DRUDE columns to determine if Drude-related keywords should be included.

    Parameters:
    ----------
    psf_filepath: str | Path
        The file path to the PSF file. This is required and should be a .psf file.
    
    Returns:
    -------
    tuple[set[str], set[str]]
        A tuple containing two sets: the first set includes the available keywords for querying the PSF file, 
        and the second set includes the available requests that can be made on the PSF file.

    Raises:
    ------
    None
    '''

    DRUDE_FLAG = False
    with open(psf_filepath, "r", encoding="ascii", errors="replace") as f:
        first_line = f.readline()

    if "DRUDE" in first_line:
        DRUDE_FLAG = True

    set_of_keywords = {
        "global_ids",
        "mass",
        "charge",
        "atom_name",
        "atom_type",
        "residue_ids",
        "segment_name",
        "residue_name",
        "local_ids",
        "is_virtual",
        "bonded_with",
        "bonded_with_mode",
    }

    if DRUDE_FLAG:
        set_of_keywords.add("drude_alpha")
        set_of_keywords.add("drude_thole")

    set_of_requests = {
        "global_ids",
        "masses",
        "charges",
        "atom_names",
        "atom_types",
        "residue_ids",
        "segment_names",
        "residue_names",
        "bonds_with",
        "local_ids",
        "property-system_charge",
    }

    if DRUDE_FLAG:
        set_of_requests.add("drude_alphas")
        set_of_requests.add("drude_tholes")

    return set_of_keywords, set_of_requests


def _plan_topology_query_psf(
    psf_filepath: str | Path,
    query_dictionary: dict,
    request_string: str,
    keywords_available: set,
    requests_available: set,
    ):

    '''
    Stochastic planner for PSF atom-selection prevalence.

    This planner intentionally does NOT estimate bonded_with constraints.
    It only estimates per-atom requests from sampled NATOM rows.
    '''

    if request_string not in requests_available:
        raise ValueError(
            f"Unsupported request_string {request_string!r}. "
            f"Available requests: {sorted(requests_available)}"
        )

    output_kind, trailing_shape, bytes_per_match = _get_psf_request_plan_shape(request_string)

    if output_kind != "per_atom":
        return {
            "planner_mode": "stochastic",
            "file_type": "psf",
            "request": request_string,
            "supported": False,
            "reason": (
                f"Stochastic payload estimation is only implemented for per-atom "
                f"requests. Request {request_string!r} is not a per-atom payload."
            ),
            "query_dictionary": query_dictionary,
        }

    bond_inc, bond_exc = query_dictionary.get("bonded_with", ([], []))
    need_bonds = bool(bond_inc or bond_exc)

    if need_bonds:
        return {
            "planner_mode": "stochastic",
            "file_type": "psf",
            "request": request_string,
            "supported": False,
            "reason": (
                "PSF stochastic planning does not estimate bonded_with constraints. "
                "Only direct atom-selection prevalence is estimated."
            ),
            "query_dictionary": query_dictionary,
        }

    predicate_state = _get_psf_topology_predicate_state(
        query_dictionary=query_dictionary,
        keywords_available=keywords_available,
    )

    sample_info = fph.iter_records_sample(
        psf_filepath,
        record_pred=_is_psf_natom_record_line,
        parse_row=_parse_psf_atom_row,
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
        if _psf_atom_matches_query(atom, predicate_state)
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


def _get_topology_query_psf(
    psf_filepath,
    query_dictionary,
    request_string,
    keywords_available,
    requests_available,
    ):
    
    '''
    Execute a query against a PSF topology file and return the requested output.
 
    Parameters:
    ----------
    psf_filepath: str | Path
        The file path to the psf file.
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
    list | float
        Depends on request_string. Per-atom requests return list, property
        requests return a scalar.
 
    Raises:
    ------
    NotImplementedError
        If the request_string is recognised but not yet implemented.
    ValueError
        If the request_string is not supported.
    '''
 
    def _atoms_iter():

        return fph.iter_records(
            psf_filepath,
            mode="counted",
            header_pred=lambda line: "!NATOM" in line,
            count_from_header=lambda header: int(header.split()[0]),
            parse_row=_parse_psf_atom_row,
            start_index=0,
            encoding="ascii",
            errors="replace",
        )
 
    def _resolve_global_ids_with_bonds(base_ids):

        '''
        Applies bonded_with constraints to a pre-filtered list of global ids.
 
        Parameters:
        ----------
        base_ids: list[int]
            Pre-filtered global atom indices to apply bond constraints to.
 
        Returns:
        -------
        list[int]
            Filtered global atom indices satisfying bond constraints.
 
        Raises:
        ------
        ValueError
            If bond constraint blocks are malformed.
        '''
        bond_inc, bond_exc = query_dictionary.get("bonded_with", ([], []))
        bond_mode, _ = query_dictionary.get("bonded_with_mode", ("all", None))
 
        if not bond_inc and not bond_exc:
            return base_ids
 
        def _freeze(obj):
            if isinstance(obj, dict):
                return tuple(sorted((k, _freeze(v)) for k, v in obj.items()))
            if isinstance(obj, (list, tuple)):
                return tuple(_freeze(x) for x in obj)
            if isinstance(obj, set):
                return tuple(sorted(obj))
            return obj
 
        neighbor_cache: dict[tuple, set[int]] = {}
 
        def _resolve_neighbor_set(block: dict) -> set[int] | None:
            if block.get("total", False):
                return None
            neighbor_q = block.get("neighbor")
            if not isinstance(neighbor_q, dict):
                raise ValueError(
                    "bonded_with block must include 'neighbor' dict unless total=True."
                )
            neighbor_q = dict(neighbor_q)
            neighbor_q.pop("bonded_with", None)
            neighbor_q.pop("bonded_with_mode", None)
            key = _freeze(neighbor_q)
            if key in neighbor_cache:
                return neighbor_cache[key]
            ids = _get_topology_query_psf(
                psf_filepath=psf_filepath,
                query_dictionary=neighbor_q,
                request_string="global_ids",
                keywords_available=keywords_available,
                requests_available=requests_available,
            )
            s = set(ids)
            neighbor_cache[key] = s
            return s
 
        neighbor_sets_inc = [_resolve_neighbor_set(b) for b in bond_inc]
        neighbor_sets_exc = [_resolve_neighbor_set(b) for b in bond_exc]
 
        bond_pass_mask = _filter_by_bonded_with(
            psf_filepath=psf_filepath,
            candidate_globals=base_ids,
            include_blocks=bond_inc,
            exclude_blocks=bond_exc,
            mode=bond_mode,
            neighbor_sets_inc=neighbor_sets_inc,
            neighbor_sets_exc=neighbor_sets_exc,
        )
 
        return [g for g, ok in zip(base_ids, bond_pass_mask) if ok]
 
    match request_string:
 
        case "global_ids":
            predicate_state = _get_psf_topology_predicate_state(
                query_dictionary=query_dictionary,
                keywords_available=keywords_available,
            )
            base_ids = [
                atom["global_id"]
                for atom in _atoms_iter()
                if _psf_atom_matches_query(atom, predicate_state)
            ]
            return _resolve_global_ids_with_bonds(base_ids)
 
        case "local_ids":
            predicate_state = _get_psf_topology_predicate_state(
                query_dictionary=query_dictionary,
                keywords_available=keywords_available,
            )
            return [
                atom["local_id"]
                for atom in _atoms_iter()
                if _psf_atom_matches_query(atom, predicate_state)
            ]
 
        case "residue_ids":
            predicate_state = _get_psf_topology_predicate_state(
                query_dictionary=query_dictionary,
                keywords_available=keywords_available,
            )
            return [
                atom["residue_id"]
                for atom in _atoms_iter()
                if _psf_atom_matches_query(atom, predicate_state)
            ]
 
        case "atom_names":
            predicate_state = _get_psf_topology_predicate_state(
                query_dictionary=query_dictionary,
                keywords_available=keywords_available,
            )
            return [
                atom["atom_name"]
                for atom in _atoms_iter()
                if _psf_atom_matches_query(atom, predicate_state)
            ]
 
        case "atom_types":
            predicate_state = _get_psf_topology_predicate_state(
                query_dictionary=query_dictionary,
                keywords_available=keywords_available,
            )
            return [
                atom["atom_type"]
                for atom in _atoms_iter()
                if _psf_atom_matches_query(atom, predicate_state)
            ]
 
        case "residue_names":
            predicate_state = _get_psf_topology_predicate_state(
                query_dictionary=query_dictionary,
                keywords_available=keywords_available,
            )
            return [
                atom["residue_name"]
                for atom in _atoms_iter()
                if _psf_atom_matches_query(atom, predicate_state)
            ]
 
        case "segment_names":
            predicate_state = _get_psf_topology_predicate_state(
                query_dictionary=query_dictionary,
                keywords_available=keywords_available,
            )
            return [
                atom["segment_name"]
                for atom in _atoms_iter()
                if _psf_atom_matches_query(atom, predicate_state)
            ]
 
        case "charges":
            predicate_state = _get_psf_topology_predicate_state(
                query_dictionary=query_dictionary,
                keywords_available=keywords_available,
            )
            return [
                atom["charge"]
                for atom in _atoms_iter()
                if _psf_atom_matches_query(atom, predicate_state)
            ]
 
        case "masses":
            predicate_state = _get_psf_topology_predicate_state(
                query_dictionary=query_dictionary,
                keywords_available=keywords_available,
            )
            return [
                atom["mass"]
                for atom in _atoms_iter()
                if _psf_atom_matches_query(atom, predicate_state)
            ]
 
        case "drude_alphas":
            predicate_state = _get_psf_topology_predicate_state(
                query_dictionary=query_dictionary,
                keywords_available=keywords_available,
            )
            return [
                atom["drude_alpha"]
                for atom in _atoms_iter()
                if _psf_atom_matches_query(atom, predicate_state)
            ]
 
        case "drude_tholes":
            predicate_state = _get_psf_topology_predicate_state(
                query_dictionary=query_dictionary,
                keywords_available=keywords_available,
            )
            return [
                atom["drude_thole"]
                for atom in _atoms_iter()
                if _psf_atom_matches_query(atom, predicate_state)
            ]
 
        case "property-system_charge":
            return sum(atom["charge"] for atom in _atoms_iter())
 
        case "bonds_with":
            raise NotImplementedError(
                "bonds_with payload return is not yet implemented. "
                "Use bonded_with in the query_dictionary to filter atoms by bond graph."
            )
 
        case _:
            raise ValueError(f"Unsupported request_string for PSF: {request_string!r}")
        

# -------------------------------------------------------------------------
# ADDITIONAL FUNCTIONALITY
# -------------------------------------------------------------------------


def _get_psf_request_plan_shape(request_string: str) -> tuple[str, tuple | None, int | None]:

    '''
    Return output_kind, trailing_shape, and bytes_per_match for planner use.
 
    Parameters:
    ----------
    request_string: str
        The request string to look up.
 
    Returns:
    -------
    tuple[str, tuple | None, int | None]
        output_kind: "per_atom", "scalar_property", or "unsupported_complex"
        trailing_shape: shape of each element beyond the atom count axis
        bytes_per_match: bytes per matched atom, None for scalar/complex
 
    Raises:
    ------
    ValueError
        If request_string is not a known request for PSF files.
    '''

    match request_string:
        case "global_ids":
            return "per_atom", (), 8
        case "local_ids":
            return "per_atom", (), 8
        case "residue_ids":
            return "per_atom", (), 8
        case "atom_names":
            return "per_atom", (), 16
        case "atom_types":
            return "per_atom", (), 16
        case "residue_names":
            return "per_atom", (), 8
        case "segment_names":
            return "per_atom", (), 8
        case "charges":
            return "per_atom", (), 8
        case "masses":
            return "per_atom", (), 8
        case "drude_alphas":
            return "per_atom", (), 8
        case "drude_tholes":
            return "per_atom", (), 8
        case "bonds_with":
            return "unsupported_complex", None, None
        case "property-system_charge":
            return "scalar_property", (), None
        case _:
            raise ValueError(f"Unsupported request_string for psf planner: {request_string!r}")
        

def _parse_psf_atom_row(row: str, global_id: int) -> dict:
    p = row.split()
    return {
        "global_id": global_id,
        "local_id": int(p[0]),
        "segment_name": p[1],
        "residue_id": int(p[2]),
        "residue_name": p[3],
        "atom_name": p[4],
        "atom_type": p[5],
        "charge": float(p[6]),
        "mass": float(p[7]),
        "is_virtual": int(p[8]) if len(p) > 8 else 0,
        "drude_alpha": float(p[9]) if len(p) > 9 else None,
        "drude_thole": float(p[10]) if len(p) > 10 else None,
    }


def _is_psf_natom_record_line(line: str) -> bool:

    parts = line.split()
    if len(parts) < 8:
        return False
    # section headers like "3 !NBOND" have a ! in the second token
    if parts[1].startswith('!'):
        return False
    return parts[0].isdigit()


def _get_psf_topology_predicate_state(query_dictionary: dict, keywords_available: set) -> dict:

    li_inc,         li_exc         = qh._normalise_query_pair(query_dictionary.get("local_ids"),    range_style=True)
    seg_inc,        seg_exc        = qh._normalise_query_pair(query_dictionary.get("segment_name"))
    ri_inc,         ri_exc         = qh._normalise_query_pair(query_dictionary.get("residue_ids"),  range_style=True)
    resn_inc,       resn_exc       = qh._normalise_query_pair(query_dictionary.get("residue_name"))
    atom_inc,       atom_exc       = qh._normalise_query_pair(query_dictionary.get("atom_name"))
    atomt_inc,      atomt_exc      = qh._normalise_query_pair(query_dictionary.get("atom_type"))
    charge_inc,     charge_exc     = qh._normalise_query_pair(query_dictionary.get("charge"),       range_style=True)
    mass_inc,       mass_exc       = qh._normalise_query_pair(query_dictionary.get("mass"),         range_style=True)
    virt_inc,       virt_exc       = qh._normalise_query_pair(query_dictionary.get("is_virtual"))

    has_alpha = "drude_alpha" in keywords_available
    has_thole = "drude_thole" in keywords_available

    if has_alpha ^ has_thole:
        raise ValueError("PSF files must have both drude_alpha and drude_thole columns or neither.")

    drude_alpha_inc, drude_alpha_exc = qh._normalise_query_pair(query_dictionary.get("drude_alpha"), range_style=True)
    drude_thole_inc, drude_thole_exc = qh._normalise_query_pair(query_dictionary.get("drude_thole"), range_style=True)

    return {
        "li_inc": li_inc,
        "li_exc": li_exc,
        "seg_inc": seg_inc,
        "seg_exc": seg_exc,
        "ri_inc": ri_inc,
        "ri_exc": ri_exc,
        "resn_inc": resn_inc,
        "resn_exc": resn_exc,
        "atom_inc": atom_inc,
        "atom_exc": atom_exc,
        "atomt_inc": atomt_inc,
        "atomt_exc": atomt_exc,
        "charge_inc": charge_inc,
        "charge_exc": charge_exc,
        "mass_inc": mass_inc,
        "mass_exc": mass_exc,
        "virt_inc": virt_inc,
        "virt_exc": virt_exc,
        "drude_alpha_inc": drude_alpha_inc,
        "drude_alpha_exc": drude_alpha_exc,
        "drude_thole_inc": drude_thole_inc,
        "drude_thole_exc": drude_thole_exc,
        "need_li": li_inc != (None, None) or li_exc != (None, None),
        "need_seg": bool(seg_inc or seg_exc),
        "need_ri": ri_inc != (None, None) or ri_exc != (None, None),
        "need_resn": bool(resn_inc or resn_exc),
        "need_atom": bool(atom_inc or atom_exc),
        "need_atomt": bool(atomt_inc or atomt_exc),
        "need_charge": charge_inc != (None, None) or charge_exc != (None, None),
        "need_mass": mass_inc != (None, None) or mass_exc != (None, None),
        "need_virt": bool(virt_inc or virt_exc),
        "need_drude_alpha": has_alpha and (drude_alpha_inc != (None, None) or drude_alpha_exc != (None, None)),
        "need_drude_thole": has_thole and (drude_thole_inc != (None, None) or drude_thole_exc != (None, None)),
    }


def _psf_atom_matches_query(atom: dict, predicate_state: dict) -> bool:
    match_ = qh._match
    match_range = qh._match_range_scalar

    ok = True

    if predicate_state["need_li"]:
        ok = match_range(atom["local_id"], predicate_state["li_inc"], predicate_state["li_exc"])
    if ok and predicate_state["need_atom"]:
        ok = match_(atom["atom_name"], predicate_state["atom_inc"], predicate_state["atom_exc"])
    if ok and predicate_state["need_atomt"]:
        ok = match_(atom["atom_type"], predicate_state["atomt_inc"], predicate_state["atomt_exc"])
    if ok and predicate_state["need_resn"]:
        ok = match_(atom["residue_name"], predicate_state["resn_inc"], predicate_state["resn_exc"])
    if ok and predicate_state["need_seg"]:
        ok = match_(atom["segment_name"], predicate_state["seg_inc"], predicate_state["seg_exc"])
    if ok and predicate_state["need_ri"]:
        ok = match_range(atom["residue_id"], predicate_state["ri_inc"], predicate_state["ri_exc"])
    if ok and predicate_state["need_charge"]:
        ok = match_range(atom["charge"], predicate_state["charge_inc"], predicate_state["charge_exc"])
    if ok and predicate_state["need_mass"]:
        ok = match_range(atom["mass"], predicate_state["mass_inc"], predicate_state["mass_exc"])
    if ok and predicate_state["need_virt"]:
        ok = match_(atom["is_virtual"], predicate_state["virt_inc"], predicate_state["virt_exc"])
    if ok and predicate_state["need_drude_alpha"]:
        ok = match_range(atom["drude_alpha"], predicate_state["drude_alpha_inc"], predicate_state["drude_alpha_exc"])
    if ok and predicate_state["need_drude_thole"]:
        ok = match_range(atom["drude_thole"], predicate_state["drude_thole_inc"], predicate_state["drude_thole_exc"])

    return ok


def _filter_by_bonded_with(
    psf_filepath: str | Path,
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

    local_to_glob_ntype, _ = _build_local_to_global_to_type_map(psf_filepath)

    cand_set = set(candidate_globals)
    cand_index = {g: i for i, g in enumerate(candidate_globals)}
    natom = max(v[0] for v in local_to_glob_ntype.values()) + 1

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

    neighbor_mask = np.zeros(natom, dtype=object)
    for col, s in enumerate(unique_sets):
        bit = 1 << col
        for g in s:
            neighbor_mask[g] = int(neighbor_mask[g]) | bit

    counts_total = np.zeros(n_cand, dtype=np.int32)
    counts_by_set = np.zeros((n_cand, len(unique_sets)), dtype=np.int32) if unique_sets else None

    def _int_stream_after_nbonds(f):
        for line in f:
            if "!NBOND" in line:
                nbond = int(line.split()[0])
                need = 2 * nbond
                got = 0
                while got < need:
                    row = f.readline()
                    if not row:
                        break
                    for tok in row.split():
                        yield int(tok)
                        got += 1
                        if got >= need:
                            return
                return
        raise ValueError("!NBOND section not found in PSF.")

    def _bump_counts(i: int, other_global: int):
        counts_total[i] += 1
        if counts_by_set is None:
            return
        m = int(neighbor_mask[other_global])
        while m:
            lsb = m & -m
            k = (lsb.bit_length() - 1)
            counts_by_set[i, k] += 1
            m ^= lsb

    with open(psf_filepath, "r", encoding="ascii", errors="replace") as f:
        ints = _int_stream_after_nbonds(f)
        for a_local, b_local in zip(ints, ints):
            a_g = local_to_glob_ntype[a_local][0]
            b_g = local_to_glob_ntype[b_local][0]

            if a_g in cand_set:
                _bump_counts(cand_index[a_g], b_g)
            if b_g in cand_set:
                _bump_counts(cand_index[b_g], a_g)

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
        elif mode == "any":
            for i in range(n_cand):
                ok_any = False
                for c in inc_norm:
                    if _eval_one(i, c):
                        ok_any = True
                        break
                mask[i] = ok_any
        else:
            raise ValueError(f"Internal Error: mode '{mode}' not supported.")

    if exc_norm:
        for i in range(n_cand):
            if not mask[i]:
                continue
            for c in exc_norm:
                if _eval_one(i, c):
                    mask[i] = False
                    break

    return mask


def _build_local_to_global_to_type_map(psf_filepath: str | Path):

    '''
    Build a local id to global and atom type mapping from the PSF file. This is used for resolving bonded_with constraints.
    In reality I should change this because we don actullally need the to type map.

    Parameters:
    ----------
    psf_filepath: str | Path
        The file path to the PSF file. This is required and should be a .psf file.
    
    Returns:
    -------
    tuple[dict[int, tuple[int, int]], dict[int, str]]
        A tuple containing:
        - A dictionary mapping local atom IDs to a tuple of (global atom ID, atom type integer).
        - A dictionary mapping atom type integers to their corresponding atom type strings.
    
    Raises:
    ------
    None
    '''


    local_global_ntype: dict[int, tuple[int, int]] = {}
    ntype_to_type: dict[int, str] = {}
    type_to_ntype: dict[str, int] = {}

    next_ntype = 0

    with open(psf_filepath, "r", encoding="ascii", errors="replace") as f:
        for line in f:
            if "!NATOM" in line:
                natom = int(line.split()[0])
                break

        for global_id in range(natom):
            line = f.readline()
            parts = line.split()

            local_id = int(parts[0])
            atom_type = parts[5]

            if atom_type not in type_to_ntype:
                type_to_ntype[atom_type] = next_ntype
                ntype_to_type[next_ntype] = atom_type
                atom_type_int = next_ntype
                next_ntype += 1
            else:
                atom_type_int = type_to_ntype[atom_type]

            local_global_ntype[local_id] = (global_id, atom_type_int)

    return local_global_ntype, ntype_to_type


# -------------------------------------------------------------------------
# ARCHIVED
# -------------------------------------------------------------------------

'''

def ARCHIVE_filter_by_bonded_with(psf_filepath: str | Path,
                           candidate_globals: list[int],
                           bonded_with_inc: list[dict],
                           bonded_with_exc: list[dict],
                           mode: str,) -> set[int]:
    
    
    #############################################################################################
    #  THIS IS VOID AND A GENERAL NEIGHBOUR FINDER HAS BEEN IMPLEMENTED USING RECURSIVE CHECKS  #
    #############################################################################################

    This function filters a list of candidate global atom indices based on their bonded neighbors and the specified bonded_with constraints.

    Parameters:
    ----------
    psf_filepath: str | Path
        The file path to the psf file. This is required and should be a .psf file.
    candidate_globals: list[int]
        A list of global atom indices to filter based on their bonded neighbors.
    bonded_with_inc: list[dict]
        A list of bonded_with constraints for inclusion. Each constraint is a dict that specifies the required number of bonded neighbors and their atom types. See the README for the format of these dicts.
    bonded_with_exc: list[dict]
        A list of bonded_with constraints for exclusion. Each constraint is a dict that specifies the required number of bonded neighbors and their atom types. See the README for the format of these dicts.
    mode: str
        The mode for evaluating the bonded_with constraints. Must be either "all" (the atom must satisfy all inclusion constraints and none of the exclusion constraints) 
        or "any" (the atom must satisfy at least one inclusion constraint and none of the exclusion constraints).

    Returns:
    -------
    set[int]
        A set of global atom indices from the candidate_globals that satisfy the bonded_with constraints according to the specified mode.
    

    if mode not in ("all", "any"):
        raise ValueError("bonded_with_mode must be 'all' or 'any'.")

    if not candidate_globals:
        return set()

    # local_id -> (global_id, atom_type_int); atom_type_int -> atom_type_str
    local_to_glob_ntype, ntype_to_type = _build_local_to_global_to_type_map(psf_filepath)
    type_to_ntype = {v: k for k, v in ntype_to_type.items()}

    cand_set = set(candidate_globals)
    cand_index = {g: i for i, g in enumerate(candidate_globals)}
    n_cand = len(candidate_globals)

    def _parse_count_cmp(count_dict: dict) -> tuple[str, int]:
        if not isinstance(count_dict, dict) or len(count_dict) != 1:
            raise ValueError("count must be a dict with exactly one comparator key.")
        (k, v), = count_dict.items()
        if k not in ("eq", "ne", "ge", "le", "gt", "lt"):
            raise ValueError(f"Unsupported comparator: {k!r}")
        if not isinstance(v, int) or v < 0:
            raise ValueError("count comparator value must be a non-negative int.")
        return k, v

    def _cmp(x: int, op: str, v: int) -> bool:
        if op == "eq": return x == v
        if op == "ne": return x != v
        if op == "ge": return x >= v
        if op == "le": return x <= v
        if op == "gt": return x > v
        if op == "lt": return x < v
        raise ValueError(f"Unknown op: {op!r}")

    # Normalize constraints to either:
    #   ("total", op, val)
    #   ("type", ntype_int, op, val)
    def _norm_one(c: dict) -> tuple:
        if not isinstance(c, dict):
            raise ValueError("bonded_with constraints must be dicts (PSF canonical format).")
        if "count" not in c:
            raise ValueError("bonded_with constraint missing 'count'.")

        op, val = _parse_count_cmp(c["count"])

        if c.get("total", False):
            return ("total", op, val)

        neighbor = c.get("neighbor")
        if not isinstance(neighbor, dict):
            raise ValueError("bonded_with constraint missing 'neighbor' dict (or set total=True).")

        if "atom_type" not in neighbor:
            raise ValueError("bonded_with.neighbor must include 'atom_type' selector.")

        atom_type_sel = neighbor["atom_type"]
        if not (isinstance(atom_type_sel, tuple) and len(atom_type_sel) == 2):
            raise ValueError("neighbor['atom_type'] must be (include_set, exclude_set).")

        inc_set, exc_set = atom_type_sel
        if not isinstance(inc_set, set) or not isinstance(exc_set, set):
            raise ValueError("neighbor['atom_type'] must be (set, set).")

        if exc_set:
            raise ValueError("neighbor['atom_type'] exclude set not supported in bonded_with yet.")
        if len(inc_set) != 1:
            raise ValueError("neighbor['atom_type'] include set must have exactly one atom type.")

        (type_str,) = tuple(inc_set)
        if type_str not in type_to_ntype:
            raise ValueError(f"Unknown atom_type in bonded_with: {type_str!r}")

        return ("type", type_to_ntype[type_str], op, val)

    inc = [_norm_one(c) for c in (bonded_with_inc or [])]
    exc = [_norm_one(c) for c in (bonded_with_exc or [])]

    if not inc and not exc:
        return set(candidate_globals)

    needed_ntypes = sorted({c[1] for c in (inc + exc) if c[0] == "type"})
    ntype_to_col = {nt: j for j, nt in enumerate(needed_ntypes)}

    counts_total = np.zeros(n_cand, dtype=np.int32)
    counts_by_type = np.zeros((n_cand, len(needed_ntypes)), dtype=np.int32) if needed_ntypes else None

    def _int_stream_after_nbonds(f):
        for line in f:
            if "!NBOND" in line:
                nbond = int(line.split()[0])
                need = 2 * nbond
                got = 0
                while got < need:
                    row = f.readline()
                    if not row:
                        break
                    for tok in row.split():
                        yield int(tok)
                        got += 1
                        if got >= need:
                            return
                return
        raise ValueError("!NBOND section not found in PSF.")

    with open(psf_filepath, "r", encoding="ascii", errors="replace") as f:
        ints = _int_stream_after_nbonds(f)
        for a_local, b_local in zip(ints, ints):
            a_g, a_nt = local_to_glob_ntype[a_local]
            b_g, b_nt = local_to_glob_ntype[b_local]

            if a_g in cand_set:
                i = cand_index[a_g]
                counts_total[i] += 1
                if counts_by_type is not None:
                    col = ntype_to_col.get(b_nt)
                    if col is not None:
                        counts_by_type[i, col] += 1

            if b_g in cand_set:
                i = cand_index[b_g]
                counts_total[i] += 1
                if counts_by_type is not None:
                    col = ntype_to_col.get(a_nt)
                    if col is not None:
                        counts_by_type[i, col] += 1

    def _eval(i: int, c: tuple) -> bool:
        if c[0] == "total":
            _, op, v = c
            return _cmp(int(counts_total[i]), op, v)
        _, nt, op, v = c
        return _cmp(int(counts_by_type[i, ntype_to_col[nt]]), op, v)

    out: set[int] = set()

    for i, g in enumerate(candidate_globals):
        # include
        if inc:
            if mode == "all":
                inc_ok = all(_eval(i, c) for c in inc)
            else:
                inc_ok = any(_eval(i, c) for c in inc)
        else:
            inc_ok = True

        if not inc_ok:
            continue

        # exclude: any match rejects
        if any(_eval(i, c) for c in exc):
            continue

        out.add(g)

    return out

'''