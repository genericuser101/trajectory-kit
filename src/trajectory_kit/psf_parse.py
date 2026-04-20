# preloaded imports
from __future__ import annotations
from pathlib import Path

# mandatory imports
import numpy as np

# local imports
from trajectory_kit import _file_parse_help as fph
from trajectory_kit import _query_help as qh

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

    output_kind, trailing_shape, bytes_per_match = _get_topology_plan_shape_psf(request_string)

    # Scalar property requests are short-circuited centrally by
    # _plan_domain_request via the plan_shape function. Defensive only.
    if output_kind != "per_atom":
        return {
            "planner_mode": "stochastic",
            "supported":    False,
            "reason": (
                f"Stochastic payload estimation is only implemented for per-atom "
                f"requests. Request {request_string!r} is not a per-atom payload."
            ),
        }

    bond_inc, bond_exc = qh._normalise_bonded_with_pair(query_dictionary.get("bonded_with"))
    need_bonds = bool(bond_inc or bond_exc)

    if need_bonds:
        return {
            "planner_mode": "stochastic",
            "supported":    False,
            "reason": (
                "PSF stochastic planning does not estimate bonded_with constraints. "
                "Only direct atom-selection prevalence is estimated."
            ),
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
        "n_frames":         1,
        "confidence":       confidence,
    }


def _get_topology_query_psf(
    psf_filepath,
    query_dictionary,
    request_string,
    keywords_available,
    requests_available,
    *,
    _bonded_with_depth: int = 0,
    _neighbor_cache: dict | None = None,
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
    _bonded_with_depth: int, default 0
        Internal recursion counter for nested ``bonded_with`` neighbour
        sub-queries. Defaults to 0 for user-facing calls; the parser
        increments it when recursing through ``_resolve_neighbor_set``.
        Capped at ``qh.MAX_BONDED_WITH_DEPTH`` (16); deeper queries raise
        ``RecursionError``.
    _neighbor_cache: dict | None, default None
        Internal call-spanning cache for resolved neighbour sets. The
        top-level call (``_bonded_with_depth == 0``) allocates a fresh dict;
        recursive calls reuse the parent's cache so identical neighbour
        sub-queries at any depth are resolved exactly once per user call.
        The cache is per-call — a fresh one is allocated for every
        user-facing entry point (``positions()``, ``fetch()``, etc.).
 
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
    RecursionError
        If nested bonded_with neighbour resolution exceeds
        ``qh.MAX_BONDED_WITH_DEPTH``.
    '''

    # Allocate a fresh neighbour cache at the top-level entry point so that
    # all recursion from this user call shares one cache. Recursive calls
    # must pass the same cache through; never reuse across user calls.
    if _neighbor_cache is None:
        _neighbor_cache = {}
 
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
        bond_inc, bond_exc = qh._normalise_bonded_with_pair(query_dictionary.get("bonded_with"))
        bond_mode, _ = query_dictionary.get("bonded_with_mode", ("all", None))
 
        if not bond_inc and not bond_exc:
            return base_ids

        def _resolve_neighbor_set(block: dict) -> set[int] | None:
            if block.get("total", False):
                return None
            neighbor_q = block.get("neighbor")
            if not isinstance(neighbor_q, dict):
                raise ValueError(
                    "bonded_with block must include 'neighbor' dict unless total=True."
                )
            # Recursion depth check — increment for the recursive call below.
            next_depth = _bonded_with_depth + 1
            if next_depth > qh.MAX_BONDED_WITH_DEPTH:
                raise RecursionError(
                    f"bonded_with neighbour recursion exceeded max depth of "
                    f"{qh.MAX_BONDED_WITH_DEPTH}. Most realistic chemistry "
                    f"queries require depth <= 5; check your query for "
                    f"unintended deep nesting."
                )
            neighbor_q = dict(neighbor_q)
            # Strip bonded_with_mode (parent-level orchestration concern).
            # bonded_with itself is intentionally KEPT — neighbour sub-queries
            # may carry their own bonded_with for graph-pattern matching.
            neighbor_q.pop("bonded_with_mode", None)
            key = qh._freeze_query(neighbor_q)
            if key in _neighbor_cache:
                return _neighbor_cache[key]
            ids = _get_topology_query_psf(
                psf_filepath=psf_filepath,
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

    def _matched_atoms():

        '''
        Yield full atom dicts for atoms passing BOTH the per-atom predicate
        AND the bond filter (when bonded_with is in the query).

        When no bonded_with is present, this streams atoms one by one with
        no materialisation cost. When bonded_with is present, it materialises
        the predicate-passing candidates once, resolves the bond filter
        against them, and re-yields the survivors in original file order.

        Every per-atom request branch consumes from this helper, so
        bonded_with is uniformly enforced across all topology requests
        rather than only for the global_ids branch.
        '''
        predicate_state = _get_psf_topology_predicate_state(
            query_dictionary=query_dictionary,
            keywords_available=keywords_available,
        )

        bond_inc, bond_exc = qh._normalise_bonded_with_pair(query_dictionary.get("bonded_with"))
        has_bond_filter = bool(bond_inc or bond_exc)

        if not has_bond_filter:
            for atom in _atoms_iter():
                if _psf_atom_matches_query(atom, predicate_state):
                    yield atom
            return

        # Bond-filter path: materialise candidates so we can apply the
        # bond constraint, then re-yield survivors in file order.
        base_atoms = [
            atom for atom in _atoms_iter()
            if _psf_atom_matches_query(atom, predicate_state)
        ]
        base_ids = [atom["global_id"] for atom in base_atoms]
        surviving = set(_resolve_global_ids_with_bonds(base_ids))

        for atom in base_atoms:
            if atom["global_id"] in surviving:
                yield atom
 
    match request_string:

        case "global_ids":
            return [atom["global_id"]    for atom in _matched_atoms()]

        case "local_ids":
            return [atom["local_id"]     for atom in _matched_atoms()]

        case "residue_ids":
            return [atom["residue_id"]   for atom in _matched_atoms()]

        case "atom_names":
            return [atom["atom_name"]    for atom in _matched_atoms()]

        case "atom_types":
            return [atom["atom_type"]    for atom in _matched_atoms()]

        case "residue_names":
            return [atom["residue_name"] for atom in _matched_atoms()]

        case "segment_names":
            return [atom["segment_name"] for atom in _matched_atoms()]

        case "charges":
            return [atom["charge"]       for atom in _matched_atoms()]

        case "masses":
            return [atom["mass"]         for atom in _matched_atoms()]

        case "drude_alphas":
            return [atom["drude_alpha"]  for atom in _matched_atoms()]

        case "drude_tholes":
            return [atom["drude_thole"]  for atom in _matched_atoms()]

        case "property-system_charge":
            # Scalar system property — ignores per-atom predicate and bond
            # filter by design; reports the total charge of the whole file.
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


def _get_topology_plan_shape_psf(request_string: str) -> tuple[str, tuple | None, int | None]:

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

    gid_inc,        gid_exc        = qh._normalise_query_pair(query_dictionary.get("global_ids"),   range_style=True)
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
        "gid_inc": gid_inc,
        "gid_exc": gid_exc,
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
        "need_gid": bool(gid_inc or gid_exc),
        "need_li": bool(li_inc or li_exc),
        "need_seg": bool(seg_inc or seg_exc),
        "need_ri": bool(ri_inc or ri_exc),
        "need_resn": bool(resn_inc or resn_exc),
        "need_atom": bool(atom_inc or atom_exc),
        "need_atomt": bool(atomt_inc or atomt_exc),
        "need_charge": bool(charge_inc or charge_exc),
        "need_mass": bool(mass_inc or mass_exc),
        "need_virt": bool(virt_inc or virt_exc),
        "need_drude_alpha": has_alpha and bool(drude_alpha_inc or drude_alpha_exc),
        "need_drude_thole": has_thole and bool(drude_thole_inc or drude_thole_exc),
    }


def _psf_atom_matches_query(atom: dict, predicate_state: dict) -> bool:
    match_ = qh._match
    match_range = qh._match_range_scalar

    ok = True

    # global_ids first: most selective integer-id match
    if predicate_state["need_gid"]:
        ok = match_range(atom["global_id"], predicate_state["gid_inc"], predicate_state["gid_exc"])

    if ok and predicate_state["need_li"]:
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


