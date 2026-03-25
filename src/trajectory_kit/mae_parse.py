 # preloaded imports
from __future__ import annotations
from pathlib import Path

# mandatory imports
import numpy as np

# local imports
from trajectory_kit import file_parse_help as fph
from trajectory_kit import query_help as qh


# =========================================================================
# MAE FORMAT NOTES
# =========================================================================
#
# IF YOU WANT TO ADD SUPPORT FOR A NEW MAE FIELD YOU MUST READ THIS
#
# A .mae file contains one or more named blocks.  The outer block is a
# version header block, followed by one or more ``f_m_ct`` (chemical
# topology) blocks.  Each f_m_ct block may carry:
#
#   m_atom[N]   — per-atom record table (coordinates, PDB names, …)
#   m_bond[N]   — bond table (i_m_from, i_m_to, i_m_order)
#   ffio_ff     — force-field sub-block, itself containing:
#       ffio_sites[N]  — per-atom FF data (charge, mass, vdwtype, …)
#       ffio_bonds[N]  — bonded term table
#
# Block syntax
# ------------
# <block_name> {
#   <key1>
#   <key2>
#   ...
#   :::
#   <row_index> <val1> <val2> ...     (data rows)
#   :::                                (end of data; optional)
# }
#
# Values may be unquoted tokens or double-quoted strings.  Quoted strings
# can span a single line but may contain spaces; they cannot span multiple
# lines.
#
# Multi-CT files (system.mae)
# ---------------------------
# The first f_m_ct (ct_type == "solute") carries the protein/solute atoms.
# Subsequent CTs are water, ions, etc.  We parse only the FIRST f_m_ct
# unless the caller explicitly requests all.  Global atom IDs (0-based) are
# assigned sequentially across all atoms in the first CT block.
#
# Column variance
# ---------------
# Column names vary between files — the parser reads the key list from
# each block header and maps them by name rather than by fixed position.
# Required columns for each request are noted inline.
#
# Token variance
# -------------
# Some fields use the token "<>" to indicate "not present" or "not applicable".
# We replace these with None in the parsed output for better ergonomics.
# This applies to any field but is most common in PDB-related fields
# (atom_name, residue_name, chain_name, etc.).
#


MAE_COL_TO_API: dict[str, tuple[str, str, str]] = {
    #  mae_column                    api_keyword           api_request             atom_key
    's_m_pdb_atom_name':        ('atom_name',           'atom_names',           'atom_name'),
    's_m_atom_name':            ('atom_name_full',      'atom_names_full',      'atom_name_full'),
    's_m_pdb_residue_name':     ('residue_name',        'residue_names',        'residue_name'),
    's_m_chain_name':           ('chain_name',          'chain_names',          'chain_name'),
    's_m_pdb_segment_name':     ('segment_name',        'segment_names',        'segment_name'),
    'i_m_residue_number':       ('residue_ids',         'residue_ids',          'residue_id'),
    'r_m_x_coord':              ('x',                   'x',                    'x'),
    'r_m_y_coord':              ('y',                   'y',                    'y'),
    'r_m_z_coord':              ('z',                   'z',                    'z'),
    'r_ffio_x_vel':             ('v_x',                 'v_x',                  'v_x'),
    'r_ffio_y_vel':             ('v_y',                 'v_y',                  'v_y'),
    'r_ffio_z_vel':             ('v_z',                 'v_z',                  'v_z'),
    'i_m_atomic_number':        ('atomic_number',       'atomic_numbers',       'atomic_number'),
    'i_m_mmod_type':            ('mmod_type',           'mmod_types',           'mmod_type'),
    'i_m_color':                ('color',               'colors',               'color'),
    'i_m_visibility':           ('visibility',          'visibilities',         'visibility'),
    'i_m_formal_charge':        ('formal_charge',       'formal_charges',       'formal_charge'),
    'r_m_charge1':              ('partial_charge_1',    'partial_charges_1',    'partial_charge_1'),
    'r_m_charge2':              ('partial_charge_2',    'partial_charges_2',    'partial_charge_2'),
    's_m_mmod_res':             ('mmod_res',            'mmod_res',             'mmod_res'),
    's_m_grow_name':            ('grow_name',           'grow_names',           'grow_name'),
    's_m_insertion_code':       ('insertion_code',      'insertion_codes',      'insertion_code'),
    'i_m_secondary_structure':  ('secondary_structure', 'secondary_structures', 'secondary_structure'),
    'r_m_pdb_tfactor':          ('pdb_tfactor',         'pdb_tfactors',         'pdb_tfactor'),
    'r_m_pdb_occupancy':        ('pdb_occupancy',       'pdb_occupancies',      'pdb_occupancy'),
    'i_m_Hcount':               ('h_count',             'h_counts',             'h_count'),
    'i_m_representation':       ('representation',      'representations',      'representation'),
    'i_m_template_index':       ('template_index',      'template_indices',     'template_index'),
}

_MAE_ATOM_KEY_TO_COL: dict[str, str] = {
    atom_key: mae_col for mae_col, (_, _, atom_key) in MAE_COL_TO_API.items()
}

MAE_FFIO_COL_TO_API: dict[str, tuple[str, str, str]] = {
    #  ffio_column        api_keyword    api_request      atom_key
    'r_ffio_charge':  ('charge',      'charges',       'ffio_charge'),
    'r_ffio_mass':    ('mass',        'masses',        'ffio_mass'),
    's_ffio_vdwtype': ('vdw_type',    'vdw_types',     'ffio_vdw_type'),
}

_MAE_FFIO_KEY_TO_COL: dict[str, str] = {
    atom_key: ffio_col for ffio_col, (_, _, atom_key) in MAE_FFIO_COL_TO_API.items()
}


# =========================================================================
# MAE TOKENISER AND LOW-LEVEL BLOCK READER
# =========================================================================

def _tokenise_mae_line(line: str) -> list[str]:

    '''
    Tokenise a single MAE data-row line, respecting double-quoted strings.

    Rules
    -----
    - Quoted strings are returned with their quotes stripped.
    - Unquoted tokens are whitespace-delimited.
    - The leading row-index integer is included as the first token.

    Parameters
    ----------
    line : str
        A raw line from a MAE data block (after the ::: separator).

    Returns
    -------
    list[str]
        List of token strings.
    '''

    tokens: list[str] = []
    i = 0
    n = len(line)

    while i < n:
        # skip whitespace
        while i < n and line[i] in ' \t\r\n':
            i += 1
        if i >= n:
            break

        if line[i] == '"':
            # quoted string: advance past opening quote
            i += 1
            start = i
            while i < n and line[i] != '"':
                i += 1
            tokens.append(line[start:i])
            i += 1  # skip closing quote
        else:
            start = i
            while i < n and line[i] not in ' \t\r\n':
                i += 1
            tokens.append(line[start:i])

    return tokens


def _iter_mae_block_rows(mae_filepath: str | Path, block_name: str):

    '''
    Yield (ct_index, keys, rows) for every f_m_ct block in the file that
    contains a sub-block named ``block_name``.

    CTs that do not contain the named block are skipped silently.  The
    ``ct_index`` counter increments for every f_m_ct seen regardless of
    whether it contained the block, so it always reflects the true position
    of the CT in the file.

    Parameters
    ----------
    mae_filepath : str | Path
    block_name : str
        E.g. ``"m_atom"``, ``"m_bond"``.
        Matched against the base name of block headers, so ``"m_atom"``
        matches ``"m_atom[39] {"`` as well as ``"m_atom {"``.

    Yields
    ------
    tuple[int, list[str], list[list[str]]]
        ct_index : int
            0-based index of the f_m_ct block in the file.
        keys : list[str]
            Ordered column names from the block header.
        rows : list[list[str]]
            Tokenised data rows; tokens[0] is the 1-based row-index (local_id),
            tokens[1:] align to keys.
    '''

    mae_filepath = Path(mae_filepath)
    ct_index     = -1

    with open(mae_filepath, 'rt', encoding='utf-8', errors='replace') as f:
        in_fmct      = False
        in_block     = False
        reading_keys = False
        reading_data = False
        keys: list[str]       = []
        rows: list[list[str]] = []

        depth = 0

        f_m_ct_depth    = 1
        block_name_depth = 2

        for raw in f:
            line = raw.strip()

        # only update depth on structural lines, not data rows
            if not reading_data:
                if '{' in line:
                    depth += line.count('{')
                if '}' in line:
                    depth -= line.count('}')

            # f_m_ct check
            if not in_fmct:
                if 'f_m_ct {' in line and depth == f_m_ct_depth:
                    in_fmct      = True
                    in_block     = False
                    reading_keys = False
                    reading_data = False
                    keys         = []
                    rows         = []
                    ct_index    += 1
                continue

            # in f_m_ct, search for block_name
            if not in_block:
                if '}' in line and depth < f_m_ct_depth:
                    in_fmct = False
                    continue

                base = line.split('[')[0].strip()
                if base == block_name and depth == block_name_depth:
                    in_block     = True
                    reading_keys = True
                continue

            # inside the named block
            if reading_keys:
                if line == ':::':
                    reading_keys = False
                    reading_data = True
                else:
                    keys.append(line)
                continue

            if reading_data:
                if line == ':::' or '}' in line:
                    yield ct_index, keys, rows
                    in_block     = False
                    reading_data = False
                    in_fmct      = False
                    continue
                if not line:
                    continue
                tokens = _tokenise_mae_line(raw)
                if tokens:
                    rows.append(tokens)


def _iter_mae_ffio_block_rows(mae_filepath: str | Path, ffio_ff_block: str):

    '''
    Yield (ct_index, keys, rows) for every f_m_ct block in the file that
    contains an ffio_ff sub-block named ``ffio_ff_block``.

    CTs that do not contain ffio_ff or the named sub-block are skipped
    silently. ct_index increments for every f_m_ct seen regardless.

    Parameters
    ----------
    mae_filepath : str | Path
    ffio_ff_block : str
        E.g. ``"ffio_sites"``, ``"ffio_bonds"``.

    Yields
    ------
    tuple[int, list[str], list[list[str]]]
        ct_index : int
        keys     : list[str]
        rows     : list[list[str]]
    '''

    mae_filepath = Path(mae_filepath)
    ct_index     = -1

    with open(mae_filepath, 'rt', encoding='utf-8', errors='replace') as f:
        in_fmct      = False
        in_ffio_ff   = False
        in_block     = False
        reading_keys = False
        reading_data = False
        keys: list[str]       = []
        rows: list[list[str]] = []

        depth = 0
        
        f_m_ct_depth = 1
        ffio_ff_depth = 2 
        ffio_ff_block_depth = 3

        for raw in f:
            line = raw.strip()

            if "{" in line:
                depth += line.count("{")
            if "}" in line:
                depth -= line.count("}")

            # f_m_ct check
            if not in_fmct:
                if 'f_m_ct {' in line and depth == f_m_ct_depth:
                    in_fmct      = True
                    in_ffio_ff   = False
                    in_block     = False
                    reading_keys = False
                    reading_data = False
                    keys         = []
                    rows         = []
                    ct_index    += 1
                continue

            # in f_m_ct, ffio_ff check 
            if not in_ffio_ff:
                if '}' in line and depth < f_m_ct_depth: # no force field in ct
                    in_fmct = False
                    continue

                if 'ffio_ff {' in line and depth == ffio_ff_depth:
                    in_ffio_ff = True
                continue

            # in f_m_ct, in ffio_ff, check for named block 
            if not in_block:
                if '}' in line and depth < ffio_ff_depth: # no ffio_ff block in ct
                    in_ffio_ff = False
                    in_fmct    = False
                    continue

                base = line.split('[')[0].strip()
                if base == ffio_ff_block:
                    in_block     = True
                    reading_keys = True
                continue

            # inside the named block
            if reading_keys:
                if line == ':::':
                    reading_keys = False
                    reading_data = True
                else:
                    keys.append(line)
                continue

            if reading_data:
                if line == ':::' or '}' in line:
                    yield ct_index, keys, rows
                    in_block     = False
                    reading_data = False
                    in_ffio_ff   = False
                    in_fmct      = False
                    continue
                if not line:
                    continue
                tokens = _tokenise_mae_line(raw)
                if tokens:
                    rows.append(tokens)


def _parse_token_to_dict(row: list[str], col_idx: dict[str, int | None]) -> dict:

    '''
    Parse a tokenised m_atom data row into an atom dict.

    Parameters
    ----------
    row : list[str]
        Output of _tokenise_mae_line. row[0] is the local_id;
        row[1:] are column values aligned to the block's key list.
    col_idx : dict[str, int | None]
        Mapping of atom_key → index into row[1:], built from
        _MAE_ATOM_KEY_TO_COL against the block's key list.

    Returns
    -------
    dict
        Atom dict with all canonical keys populated. global_id is not
        set here — the caller assigns it from the CT offset.
    '''

    vals = row[1:]

    def _get(atom_key: str, default: str = '') -> str:
        i = col_idx[atom_key]
        if i is None or i >= len(vals):
            return default
        return vals[i]

    return {

        # global_ids is missing since only the true iterator knows the global offset
        'local_id':           int(row[0]),
        # strings
        'atom_name':          _get('atom_name').strip(),
        'atom_name_full':     _get('atom_name_full').strip(),
        'residue_name':       _get('residue_name').strip(),
        'chain_name':         _get('chain_name').strip(),
        'segment_name':       _get('segment_name').strip(),
        'mmod_res':           _get('mmod_res').strip(),
        'grow_name':          _get('grow_name').strip(),
        'insertion_code':     _get('insertion_code').strip(),
        # integers
        'residue_id':         int(_get('residue_id',          '0') or 0),
        'atomic_number':      int(_get('atomic_number',       '0') or 0),
        'mmod_type':          int(_get('mmod_type',           '0') or 0),
        'color':              int(_get('color',               '0') or 0),
        'visibility':         int(_get('visibility',          '0') or 0),
        'formal_charge':      int(_get('formal_charge',       '0') or 0),
        'secondary_structure':int(_get('secondary_structure', '0') or 0),
        'h_count':            int(_get('h_count',             '0') or 0),
        'representation':     int(_get('representation',      '0') or 0),
        'template_index':     int(_get('template_index',      '0') or 0),
        # floats
        'x':                  float(_get('x',   '0.0') or 0.0),
        'y':                  float(_get('y',   '0.0') or 0.0),
        'z':                  float(_get('z',   '0.0') or 0.0),
        'v_x':                float(_get('v_x', '0.0') or 0.0),
        'v_y':                float(_get('v_y', '0.0') or 0.0),
        'v_z':                float(_get('v_z', '0.0') or 0.0),
        'partial_charge_1':   float(_get('partial_charge_1', '0.0') or 0.0),
        'partial_charge_2':   float(_get('partial_charge_2', '0.0') or 0.0),
        'pdb_tfactor':   float(_get('pdb_tfactor',   '0.0')) if _get('pdb_tfactor',   '0.0') != '<>' else None,
        'pdb_occupancy': float(_get('pdb_occupancy', '0.0')) if _get('pdb_occupancy', '0.0') != '<>' else None,
    }


###########################################################################


# =========================================================================
# TYPING INTERFACE  (m_atom block — positions, PDB atom/residue/chain info)
# =========================================================================

def _update_type_globals_mae(mae_filepath: str | Path) -> dict:

    '''
    Extract global system properties from a MAE typing file.

    Reads the atom count from the m_atom block header (O(1)) then makes
    a single pass to gather residue IDs and bounding-box coordinates.

    Parameters
    ----------
    mae_filepath : str | Path

    Returns
    -------
    dict
        Keys matching ``sim.global_system_properties``.
    '''

    num_atoms = 0
    residue_ids: set[int] = set()
    xs, ys, zs = [], [], []

    for atom in _iter_over_m_atoms(mae_filepath):
        num_atoms += 1
        residue_ids.add(atom['residue_id'])
        xs.append(atom['x'])
        ys.append(atom['y'])
        zs.append(atom['z'])

    result: dict = {}
    if num_atoms:
        result['num_atoms'] = num_atoms
        result['num_residues'] = len(residue_ids)

    if xs:
        result['start_box_size'] = (
            min(xs), max(xs),
            min(ys), max(ys),
            min(zs), max(zs),
        )

    return result


def _get_type_keys_reqs_mae(mae_filepath: str | Path):

    '''
    Return available typing keywords and requests for a MAE file.

    Scans every f_m_ct m_atom block in the file and takes the union of all
    column key lists. A keyword is advertised if it appears in at least one
    CT's m_atom block. Columns absent from a given CT produce zero matches
    for that CT when queried — they do not raise an error.

    Static fields (global_ids, local_ids, positions, velocities, property-*)
    are always included regardless of which columns are present.

    Parameters
    ----------
    mae_filepath : str | Path

    Returns
    -------
    tuple[set[str], set[str]]
        (keywords, requests)
    '''

    COORD_COLS    = {'r_m_x_coord', 'r_m_y_coord', 'r_m_z_coord'}
    VELOCITY_COLS = {'r_ffio_x_vel', 'r_ffio_y_vel', 'r_ffio_z_vel'}

    mae_cols = _get_all_mae_m_atom_keys(mae_filepath)

    set_of_keywords: set[str] = {'global_ids', 'local_ids'}
    set_of_requests: set[str] = {
        'global_ids',
        'local_ids',
        'property-number_of_atoms',
        'property-number_of_residues',
        'property-box_size',
        'property-system_formal_charge',
    }

    for mae_col, (kw, req, _) in MAE_COL_TO_API.items():
        if mae_col in mae_cols:
            set_of_keywords.add(kw)
            set_of_requests.add(req)

    # Combined requests — only if all three component columns are present
    if COORD_COLS.issubset(mae_cols):
        set_of_requests.add('positions')

    if VELOCITY_COLS.issubset(mae_cols):
        set_of_requests.add('velocities')

    return set_of_keywords, set_of_requests


def _plan_type_query_mae(
    mae_filepath: str | Path,
    query_dictionary: dict,
    request_string: str,
    keywords_available: set,
    requests_available: set,
    ):

    if request_string not in requests_available:
        raise ValueError(
            f'Unsupported request_string {request_string!r}. '
            f'Available requests: {sorted(requests_available)}'
        )

    output_kind, trailing_shape, bytes_per_match = _get_mae_type_request_plan_shape(request_string)

    if output_kind == 'scalar_property':
        return {
            'planner_mode': 'stochastic',
            'file_type':    'mae',
            'request':      request_string,
            'supported':    False,
            'reason':       f'{request_string!r} is a scalar property, not estimated.',
            'query_dictionary': query_dictionary,
        }

    SAMPLE_SIZE     = 500
    predicate_state = _get_m_atom_predicate(query_dictionary)

    # Get atom count per CT from block rows — O(headers + data) but we
    # need rows anyway to know CT sizes for proportional allocation
    ct_sizes: list[int] = [
        len(rows)
        for _, _, rows in _iter_mae_block_rows(mae_filepath, 'm_atom')
    ]

    n_total = sum(ct_sizes)
    if n_total == 0:
        return {'planner_mode': 'stochastic', 'n_atoms': 0, 'confidence': 'none'}

    # Allocate sample budget proportionally across CTs — minimum 1 per CT
    ct_budgets: list[int] = [
        max(1, int(round(SAMPLE_SIZE * ct_size / n_total)))
        for ct_size in ct_sizes
    ]

    # Sample from each CT up to its budget
    n_sampled  = 0
    n_matched  = 0
    ct_iter    = iter(enumerate(ct_budgets))
    ct_idx, budget = next(ct_iter)
    ct_count   = 0
    current_ct = 0

    for atom in _iter_over_m_atoms(mae_filepath):
        # track which CT we are in by counting atoms
        if current_ct < len(ct_sizes) and ct_count >= ct_sizes[current_ct]:
            ct_count   = 0
            current_ct += 1

        if current_ct < len(ct_budgets) and ct_count < ct_budgets[current_ct]:
            n_sampled += 1
            if _does_m_atom_query_match(atom, predicate_state):
                n_matched += 1

        ct_count += 1

    if n_sampled == 0:
        return {'planner_mode': 'stochastic', 'n_atoms': 0, 'confidence': 'none'}

    match_frac        = n_matched / n_sampled
    estimated_matches = int(round(n_total * match_frac))
    estimated_bytes   = estimated_matches * bytes_per_match

    confidence = (
        'none'   if n_matched == 0  else
        'low'    if n_matched < 10  else
        'medium' if n_matched < 100 else
        'high'
    )

    return {
        'planner_mode':    'stochastic',
        'n_atoms_sampled': n_sampled,
        'n_atoms_matched': n_matched,
        'n_atoms':         estimated_matches,
        'estimated_bytes': estimated_bytes,
        'estimated_mib':   estimated_bytes / (1024 ** 2),
        'confidence':      confidence,
    }


def _get_type_query_mae(
    mae_filepath: str | Path,
    query_dictionary: dict,
    request_string: str,
    keywords_available: set,
    requests_available: set,
    ):

    '''
    Execute a typing query against a MAE file.

    Parameters
    ----------
    mae_filepath : str | Path
    query_dictionary : dict
    request_string : str
    keywords_available : set
    requests_available : set

    Returns
    -------
    list | int | tuple | np.ndarray
        Depends on request_string.
    '''

    predicate_state = _get_m_atom_predicate(query_dictionary)

    def _matched():
        for atom in _iter_over_m_atoms(mae_filepath):
            if _does_m_atom_query_match(atom, predicate_state):
                yield atom

    match request_string:

        case 'global_ids':
            return [a['global_id'] for a in _matched()]

        case 'local_ids':
            return [a['local_id'] for a in _matched()]

        case 'atom_names':
            return [a['atom_name'] for a in _matched()]

        case 'atom_names_full':
            return [a['atom_name_full'] for a in _matched()]

        case 'residue_names':
            return [a['residue_name'] for a in _matched()]

        case 'residue_ids':
            return [a['residue_id'] for a in _matched()]

        case 'chain_names':
            return [a['chain_name'] for a in _matched()]

        case 'segment_names':
            return [a['segment_name'] for a in _matched()]

        case 'atomic_numbers':
            return [a['atomic_number'] for a in _matched()]

        case 'mmod_types':
            return [a['mmod_type'] for a in _matched()]

        case 'colors':
            return [a['color'] for a in _matched()]

        case 'visibilities':
            return [a['visibility'] for a in _matched()]

        case 'formal_charges':
            return [a['formal_charge'] for a in _matched()]

        case 'partial_charges_1':
            return [a['partial_charge_1'] for a in _matched()]

        case 'partial_charges_2':
            return [a['partial_charge_2'] for a in _matched()]

        case 'mmod_res':
            return [a['mmod_res'] for a in _matched()]

        case 'grow_names':
            return [a['grow_name'] for a in _matched()]

        case 'insertion_codes':
            return [a['insertion_code'] for a in _matched()]

        case 'secondary_structures':
            return [a['secondary_structure'] for a in _matched()]

        case 'pdb_tfactors':
            return [a['pdb_tfactor'] for a in _matched()]

        case 'pdb_occupancies':
            return [a['pdb_occupancy'] for a in _matched()]

        case 'h_counts':
            return [a['h_count'] for a in _matched()]

        case 'representations':
            return [a['representation'] for a in _matched()]

        case 'template_indices':
            return [a['template_index'] for a in _matched()]

        case 'x':
            return [a['x'] for a in _matched()]

        case 'y':
            return [a['y'] for a in _matched()]

        case 'z':
            return [a['z'] for a in _matched()]

        case 'v_x':
            return [a['v_x'] for a in _matched()]

        case 'v_y':
            return [a['v_y'] for a in _matched()]

        case 'v_z':
            return [a['v_z'] for a in _matched()]

        case 'positions':
            rows = [(a['x'], a['y'], a['z']) for a in _matched()]
            arr = np.array(rows, dtype=np.float32).reshape(-1, 3)
            return arr[np.newaxis, :, :]  # (1, n, 3)

        case 'velocities':
            rows = [(a['v_x'], a['v_y'], a['v_z']) for a in _matched()]
            arr = np.array(rows, dtype=np.float32).reshape(-1, 3)
            return arr[np.newaxis, :, :]  # (1, n, 3)

        case 'property-number_of_atoms':
            return sum(1 for _ in _iter_over_m_atoms(mae_filepath))

        case 'property-number_of_residues':
            return len({a['residue_id'] for a in _iter_over_m_atoms(mae_filepath)})

        case 'property-box_size':
            xs, ys, zs = [], [], []
            for a in _iter_over_m_atoms(mae_filepath):
                xs.append(a['x'])
                ys.append(a['y'])
                zs.append(a['z'])
            if not xs:
                raise ValueError('No atom records found in MAE file.')
            return (min(xs), max(xs), min(ys), max(ys), min(zs), max(zs))

        case 'property-system_formal_charge':
            return sum(a['formal_charge'] for a in _iter_over_m_atoms(mae_filepath))

        case _:
            raise ValueError(f'Unsupported request_string for MAE typing: {request_string!r}')


# =========================================================================
# TYPING HELPERS
# =========================================================================


def _iter_over_m_atoms(mae_filepath: str | Path):

    '''
    Yield atom dicts from the m_atom block of ALL f_m_ct blocks.

    global_id is assigned sequentially across all CTs (0-based, never
    restarting). local_id is the 1-based row index within each CT as it
    appears in the file.

    Every field in MAE_COL_TO_API is mapped. Missing columns in a given CT
    default to '' (strings), 0 (integers), or 0.0 (floats). The yielded
    dict is always complete regardless of which columns are present in the
    file.

    Column-to-atom-key mapping is driven entirely by the module-level
    MAE_COL_TO_API constant via _MAE_ATOM_KEY_TO_COL — there is no local
    copy of the mapping in this function.

    Parameters
    ----------
    mae_filepath : str | Path

    Yields
    ------
    dict
    '''

    global_offset = 0

    for _ct_index, keys, rows in _iter_mae_block_rows(mae_filepath, 'm_atom'):

        # Rebuild col_idx per CT — each CT may have a different column set
        col_idx: dict[str, int | None] = {
            atom_key: (keys.index(mae_col) if mae_col in keys else None)
            for atom_key, mae_col in _MAE_ATOM_KEY_TO_COL.items()
        }

        def _get(vals: list[str], atom_key: str, default: str = '') -> str:
            i = col_idx[atom_key]
            if i is None or i >= len(vals):
                return default
            return vals[i]

        for ct_local_id, row in enumerate(rows):
            local_id  = int(row[0]) if row else ct_local_id + 1
            global_id = global_offset + ct_local_id
            vals = row[1:]  # column values aligned to keys[]
            yield {
                'global_id':          global_id,
                'local_id':           local_id,
                # strings
                'atom_name':          _get(vals, 'atom_name').strip(),
                'atom_name_full':     _get(vals, 'atom_name_full').strip(),
                'residue_name':       _get(vals, 'residue_name').strip(),
                'chain_name':         _get(vals, 'chain_name').strip(),
                'segment_name':       _get(vals, 'segment_name').strip(),
                'mmod_res':           _get(vals, 'mmod_res').strip(),
                'grow_name':          _get(vals, 'grow_name').strip(),
                'insertion_code':     _get(vals, 'insertion_code').strip(),
                # integers
                'residue_id':         int(_get(vals, 'residue_id',          '0') or 0),
                'atomic_number':      int(_get(vals, 'atomic_number',       '0') or 0),
                'mmod_type':          int(_get(vals, 'mmod_type',           '0') or 0),
                'color':              int(_get(vals, 'color',               '0') or 0),
                'visibility':         int(_get(vals, 'visibility',          '0') or 0),
                'formal_charge':      int(_get(vals, 'formal_charge',       '0') or 0),
                'secondary_structure':int(_get(vals, 'secondary_structure', '0') or 0),
                'h_count':            int(_get(vals, 'h_count',             '0') or 0),
                'representation':     int(_get(vals, 'representation',      '0') or 0),
                'template_index':     int(_get(vals, 'template_index',      '0') or 0),
                # floats
                'x':                  float(_get(vals, 'x',   '0.0') or 0.0),
                'y':                  float(_get(vals, 'y',   '0.0') or 0.0),
                'z':                  float(_get(vals, 'z',   '0.0') or 0.0),
                'v_x':                float(_get(vals, 'v_x', '0.0') or 0.0),
                'v_y':                float(_get(vals, 'v_y', '0.0') or 0.0),
                'v_z':                float(_get(vals, 'v_z', '0.0') or 0.0),
                'partial_charge_1':   float(_get(vals, 'partial_charge_1', '0.0') or 0.0),
                'partial_charge_2':   float(_get(vals, 'partial_charge_2', '0.0') or 0.0),
                'pdb_tfactor':   float(_get(vals, 'pdb_tfactor',   '0.0')) if _get(vals, 'pdb_tfactor',   '0.0') != '<>' else None,
                'pdb_occupancy': float(_get(vals, 'pdb_occupancy', '0.0')) if _get(vals, 'pdb_occupancy', '0.0') != '<>' else None,
            }

        global_offset += len(rows)


def _get_m_atom_predicate(query_dictionary: dict) -> dict:

    '''
    Pre-compute include/exclude sets from the typing query dictionary.

    Parameters
    ----------
    query_dictionary : dict

    Returns
    -------
    dict
    '''

    def _s(key):
        return qh._normalise_query_pair(query_dictionary.get(key))

    def _r(key):
        return qh._normalise_query_pair(query_dictionary.get(key), range_style=True)

    # string fields
    atom_inc,    atom_exc    = _s('atom_name')
    atf_inc,     atf_exc     = _s('atom_name_full')
    resn_inc,    resn_exc    = _s('residue_name')
    chain_inc,   chain_exc   = _s('chain_name')
    seg_inc,     seg_exc     = _s('segment_name')
    mres_inc,    mres_exc    = _s('mmod_res')
    grow_inc,    grow_exc    = _s('grow_name')
    ins_inc,     ins_exc     = _s('insertion_code')

    # integer / float range fields
    
    li_inc,      li_exc      = _r('local_ids')
    ri_inc,      ri_exc      = _r('residue_ids')
    anum_inc,    anum_exc    = _r('atomic_number')
    mtype_inc,   mtype_exc   = _r('mmod_type')
    color_inc,   color_exc   = _r('color')
    vis_inc,     vis_exc     = _r('visibility')
    fchg_inc,    fchg_exc    = _r('formal_charge')
    pc1_inc,     pc1_exc     = _r('partial_charge_1')
    pc2_inc,     pc2_exc     = _r('partial_charge_2')
    ss_inc,      ss_exc      = _r('secondary_structure')
    tfac_inc,    tfac_exc    = _r('pdb_tfactor')
    occ_inc,     occ_exc     = _r('pdb_occupancy')
    hc_inc,      hc_exc      = _r('h_count')
    repr_inc,    repr_exc    = _r('representation')
    tmpl_inc,    tmpl_exc    = _r('template_index')
    x_inc,       x_exc       = _r('x')
    y_inc,       y_exc       = _r('y')
    z_inc,       z_exc       = _r('z')
    vx_inc,      vx_exc      = _r('v_x')
    vy_inc,      vy_exc      = _r('v_y')
    vz_inc,      vz_exc      = _r('v_z')

    return {
        'atom_inc': atom_inc,  'atom_exc': atom_exc,
        'atf_inc':  atf_inc,   'atf_exc':  atf_exc,
        'resn_inc': resn_inc,  'resn_exc': resn_exc,
        'chain_inc':chain_inc, 'chain_exc':chain_exc,
        'seg_inc':  seg_inc,   'seg_exc':  seg_exc,
        'mres_inc': mres_inc,  'mres_exc': mres_exc,
        'grow_inc': grow_inc,  'grow_exc': grow_exc,
        'ins_inc':  ins_inc,   'ins_exc':  ins_exc,
        'li_inc':   li_inc,    'li_exc':   li_exc,
        'ri_inc':   ri_inc,    'ri_exc':   ri_exc,
        'anum_inc': anum_inc,  'anum_exc': anum_exc,
        'mtype_inc':mtype_inc, 'mtype_exc':mtype_exc,
        'color_inc':color_inc, 'color_exc':color_exc,
        'vis_inc':  vis_inc,   'vis_exc':  vis_exc,
        'fchg_inc': fchg_inc,  'fchg_exc': fchg_exc,
        'pc1_inc':  pc1_inc,   'pc1_exc':  pc1_exc,
        'pc2_inc':  pc2_inc,   'pc2_exc':  pc2_exc,
        'ss_inc':   ss_inc,    'ss_exc':   ss_exc,
        'tfac_inc': tfac_inc,  'tfac_exc': tfac_exc,
        'occ_inc':  occ_inc,   'occ_exc':  occ_exc,
        'hc_inc':   hc_inc,    'hc_exc':   hc_exc,
        'repr_inc': repr_inc,  'repr_exc': repr_exc,
        'tmpl_inc': tmpl_inc,  'tmpl_exc': tmpl_exc,
        'x_inc':    x_inc,     'x_exc':    x_exc,
        'y_inc':    y_inc,     'y_exc':    y_exc,
        'z_inc':    z_inc,     'z_exc':    z_exc,
        'vx_inc':   vx_inc,    'vx_exc':   vx_exc,
        'vy_inc':   vy_inc,    'vy_exc':   vy_exc,
        'vz_inc':   vz_inc,    'vz_exc':   vz_exc,
        # need flags
        'need_atom':  bool(atom_inc  or atom_exc),
        'need_atf':   bool(atf_inc   or atf_exc),
        'need_resn':  bool(resn_inc  or resn_exc),
        'need_chain': bool(chain_inc or chain_exc),
        'need_seg':   bool(seg_inc   or seg_exc),
        'need_mres':  bool(mres_inc  or mres_exc),
        'need_grow':  bool(grow_inc  or grow_exc),
        'need_ins':   bool(ins_inc   or ins_exc),
        'need_li':    li_inc    != (None, None) or li_exc    != (None, None),
        'need_ri':    ri_inc    != (None, None) or ri_exc    != (None, None),
        'need_anum':  anum_inc  != (None, None) or anum_exc  != (None, None),
        'need_mtype': mtype_inc != (None, None) or mtype_exc != (None, None),
        'need_color': color_inc != (None, None) or color_exc != (None, None),
        'need_vis':   vis_inc   != (None, None) or vis_exc   != (None, None),
        'need_fchg':  fchg_inc  != (None, None) or fchg_exc  != (None, None),
        'need_pc1':   pc1_inc   != (None, None) or pc1_exc   != (None, None),
        'need_pc2':   pc2_inc   != (None, None) or pc2_exc   != (None, None),
        'need_ss':    ss_inc    != (None, None) or ss_exc    != (None, None),
        'need_tfac':  tfac_inc  != (None, None) or tfac_exc  != (None, None),
        'need_occ':   occ_inc   != (None, None) or occ_exc   != (None, None),
        'need_hc':    hc_inc    != (None, None) or hc_exc    != (None, None),
        'need_repr':  repr_inc  != (None, None) or repr_exc  != (None, None),
        'need_tmpl':  tmpl_inc  != (None, None) or tmpl_exc  != (None, None),
        'need_x':     x_inc     != (None, None) or x_exc     != (None, None),
        'need_y':     y_inc     != (None, None) or y_exc     != (None, None),
        'need_z':     z_inc     != (None, None) or z_exc     != (None, None),
        'need_vx':    vx_inc    != (None, None) or vx_exc    != (None, None),
        'need_vy':    vy_inc    != (None, None) or vy_exc    != (None, None),
        'need_vz':    vz_inc    != (None, None) or vz_exc    != (None, None),
    }


def _does_m_atom_query_match(atom: dict, predicate_state: dict) -> bool:

    '''
    Shared MAE atom-selection predicate used by both typing and topology.
    Tests m_atom fields only — ffio fields are handled by _does_force_field_query_match.

    Parameters
    ----------
    atom : dict
    predicate_state : dict

    Returns
    -------
    bool
    '''

    m  = qh._match
    mr = qh._match_range_scalar
    ps = predicate_state
    ok = True

    # Integer range checks first (cheapest for most queries)
    if ps['need_li']:    ok = mr(atom['local_id'],            ps['li_inc'],    ps['li_exc'])
    if ok and ps['need_ri']:    ok = mr(atom['residue_id'],   ps['ri_inc'],    ps['ri_exc'])
    if ok and ps['need_anum']:  ok = mr(atom['atomic_number'], ps['anum_inc'], ps['anum_exc'])
    if ok and ps['need_mtype']: ok = mr(atom['mmod_type'],    ps['mtype_inc'], ps['mtype_exc'])
    if ok and ps['need_color']: ok = mr(atom['color'],        ps['color_inc'], ps['color_exc'])
    if ok and ps['need_vis']:   ok = mr(atom['visibility'],   ps['vis_inc'],   ps['vis_exc'])
    if ok and ps['need_fchg']:  ok = mr(atom['formal_charge'], ps['fchg_inc'], ps['fchg_exc'])
    if ok and ps['need_ss']:    ok = mr(atom['secondary_structure'], ps['ss_inc'], ps['ss_exc'])
    if ok and ps['need_hc']:    ok = mr(atom['h_count'],      ps['hc_inc'],    ps['hc_exc'])
    if ok and ps['need_repr']:  ok = mr(atom['representation'], ps['repr_inc'], ps['repr_exc'])
    if ok and ps['need_tmpl']:  ok = mr(atom['template_index'], ps['tmpl_inc'], ps['tmpl_exc'])
    # String set checks
    if ok and ps['need_atom']:  ok = m(atom['atom_name'],     ps['atom_inc'],  ps['atom_exc'])
    if ok and ps['need_atf']:   ok = m(atom['atom_name_full'], ps['atf_inc'],  ps['atf_exc'])
    if ok and ps['need_resn']:  ok = m(atom['residue_name'],  ps['resn_inc'],  ps['resn_exc'])
    if ok and ps['need_chain']: ok = m(atom['chain_name'],    ps['chain_inc'], ps['chain_exc'])
    if ok and ps['need_seg']:   ok = m(atom['segment_name'],  ps['seg_inc'],   ps['seg_exc'])
    if ok and ps['need_mres']:  ok = m(atom['mmod_res'],      ps['mres_inc'],  ps['mres_exc'])
    if ok and ps['need_grow']:  ok = m(atom['grow_name'],     ps['grow_inc'],  ps['grow_exc'])
    if ok and ps['need_ins']:   ok = m(atom['insertion_code'], ps['ins_inc'],  ps['ins_exc'])
    # Float range checks
    if ok and ps['need_pc1']:   ok = mr(atom['partial_charge_1'], ps['pc1_inc'],  ps['pc1_exc'])
    if ok and ps['need_pc2']:   ok = mr(atom['partial_charge_2'], ps['pc2_inc'],  ps['pc2_exc'])
    if ok and ps['need_tfac']:  ok = mr(atom['pdb_tfactor'],      ps['tfac_inc'], ps['tfac_exc'])
    if ok and ps['need_occ']:   ok = mr(atom['pdb_occupancy'],    ps['occ_inc'],  ps['occ_exc'])
    if ok and ps['need_x']:     ok = mr(atom['x'],  ps['x_inc'],  ps['x_exc'])
    if ok and ps['need_y']:     ok = mr(atom['y'],  ps['y_inc'],  ps['y_exc'])
    if ok and ps['need_z']:     ok = mr(atom['z'],  ps['z_inc'],  ps['z_exc'])
    if ok and ps['need_vx']:    ok = mr(atom['v_x'], ps['vx_inc'], ps['vx_exc'])
    if ok and ps['need_vy']:    ok = mr(atom['v_y'], ps['vy_inc'], ps['vy_exc'])
    if ok and ps['need_vz']:    ok = mr(atom['v_z'], ps['vz_inc'], ps['vz_exc'])

    return ok


# size of the plan
def _get_mae_type_request_plan_shape(request_string: str) -> tuple[str, tuple, int | None]:

    '''
    Return (output_kind, trailing_shape, bytes_per_match) for the typing planner.

    Parameters
    ----------
    request_string : str

    Returns
    -------
    tuple[str, tuple, int | None]
    '''

    match request_string:
        case 'global_ids':              return 'per_atom', (),    8
        case 'local_ids':               return 'per_atom', (),    8
        case 'atom_names':              return 'per_atom', (),   16
        case 'atom_names_full':         return 'per_atom', (),   16
        case 'residue_names':           return 'per_atom', (),    8
        case 'residue_ids':             return 'per_atom', (),    8
        case 'chain_names':             return 'per_atom', (),    4
        case 'segment_names':           return 'per_atom', (),    8
        case 'mmod_types':              return 'per_atom', (),    4
        case 'colors':                  return 'per_atom', (),    4
        case 'visibilities':            return 'per_atom', (),    4
        case 'formal_charges':          return 'per_atom', (),    4
        case 'partial_charges_1':       return 'per_atom', (),    8
        case 'partial_charges_2':       return 'per_atom', (),    8
        case 'mmod_res':                return 'per_atom', (),    4
        case 'grow_names':              return 'per_atom', (),    8
        case 'insertion_codes':         return 'per_atom', (),    4
        case 'secondary_structures':    return 'per_atom', (),    4
        case 'pdb_tfactors':            return 'per_atom', (),    8
        case 'pdb_occupancies':         return 'per_atom', (),    8
        case 'h_counts':                return 'per_atom', (),    4
        case 'representations':         return 'per_atom', (),    4
        case 'template_indices':        return 'per_atom', (),    4
        case 'atomic_numbers':          return 'per_atom', (),    4
        case 'x':                       return 'per_atom', (),    8
        case 'y':                       return 'per_atom', (),    8
        case 'z':                       return 'per_atom', (),    8
        case 'v_x':                     return 'per_atom', (),    8
        case 'v_y':                     return 'per_atom', (),    8
        case 'v_z':                     return 'per_atom', (),    8
        case 'positions':               return 'per_atom', (3,), 12
        case 'velocities':              return 'per_atom', (3,), 12
        case 'property-system_formal_charge': return 'scalar_property', (), None
        case 'property-number_of_atoms':      return 'scalar_property', (), None
        case 'property-number_of_residues':   return 'scalar_property', (), None
        case 'property-box_size':             return 'scalar_property', (6,), None
        case _:
            raise ValueError(f'Unsupported request_string for mae type planner: {request_string!r}')

# get all type keys
def _get_all_mae_m_atom_keys(mae_filepath: str | Path) -> set[str]:

    '''
    Walk every f_m_ct block in the file and return the union of all m_atom
    column key lists as a set of raw MAE column names.

    This is header-only (no data rows are read) so the cost is O(number of
    block headers) regardless of atom count or CT count.

    A keyword advertised here means it exists in at least one CT's m_atom
    block. If the column is absent from a particular CT, queries using it
    will return zero matches for that CT — not an error.

    Parameters
    ----------
    mae_filepath : str | Path

    Returns
    -------
    set[str]
        Union of all m_atom column names across all f_m_ct blocks.
    '''

    mae_filepath = Path(mae_filepath)
    all_keys: set[str] = set()

    with open(mae_filepath, 'rt', encoding='utf-8', errors='replace') as f:
        in_fmct      = False
        in_m_atom    = False
        reading_keys = False

        for raw in f:
            line = raw.strip()

            # Detect any f_m_ct opening — reset per-CT state each time
            if line == 'f_m_ct {':
                in_fmct      = True
                in_m_atom    = False
                reading_keys = False
                continue

            if not in_fmct:
                continue

            # Top-level } closes the current f_m_ct
            if line == '}' and not in_m_atom:
                in_fmct = False
                continue

            if not in_m_atom:
                base = line.split('[')[0].split('{')[0].strip()
                if base == 'm_atom' and line.endswith('{'):
                    in_m_atom    = True
                    reading_keys = True
                continue

            # Inside m_atom header — collect keys until :::
            if reading_keys:
                if line == ':::':
                    # Done with this block's key list; skip data rows by
                    # resetting — the outer loop will find the next f_m_ct
                    reading_keys = False
                    in_m_atom    = False
                    in_fmct      = False
                else:
                    all_keys.add(line)

    return all_keys




###########################################################################




# =========================================================================
# TOPOLOGY INTERFACE  (ffio_sites charges/masses + m_bond connectivity)
# =========================================================================

def _update_topology_globals_mae(mae_filepath: str | Path) -> dict:

    '''
    Extract global system properties from a MAE topology file.

    Reads the m_atom block count from the block header (O(1)), consistent
    with the typing globals function.

    Parameters
    ----------
    mae_filepath : str | Path

    Returns
    -------
    dict
        Keys matching ``sim.global_system_properties``.
    '''

    # Delegate to typing globals — both read from m_atom.
    return _update_type_globals_mae(mae_filepath)


def _get_topology_keys_reqs_mae(mae_filepath: str | Path):

    '''
    Return available topology keywords and requests for a MAE file.

    The topology interface is a strict superset of typing: all m_atom
    keywords and requests are available (via _get_type_keys_reqs_mae),
    plus m_bond (bonded_with) and any ffio_sites columns that are
    actually present in the file.

    ffio block availability is discovered by _get_all_mae_force_field_keys so
    that future ffio extensions (angles, dihedrals, etc.) can be wired
    in here without changing the discovery logic.

    Angles, dihedrals, impropers, and CMAP are out of scope for now.

    Parameters
    ----------
    mae_filepath : str | Path

    Returns
    -------
    tuple[set[str], set[str]]
        (keywords, requests)
    '''

    # Start from the full typing set — topology is a superset
    set_of_keywords, set_of_requests = _get_type_keys_reqs_mae(mae_filepath)

    # Bond network — always available as a topology concept even if the
    # file has no m_bond block (queries will return zero matches)
    set_of_keywords = set_of_keywords | {'bonded_with', 'bonded_with_mode'}

    # Discover which ffio sub-blocks and columns actually exist
    ffio = _get_all_mae_force_field_keys(mae_filepath)
    sites = ffio.get('ffio_sites', set())

    for ffio_col, (kw, req, _) in MAE_FFIO_COL_TO_API.items():
        if ffio_col in sites:
            set_of_keywords = set_of_keywords | {kw}
            set_of_requests = set_of_requests | {req}

    if 'r_ffio_charge' in sites:
        set_of_requests = set_of_requests | {'property-system_charge'}

    return set_of_keywords, set_of_requests


def _plan_topology_query_mae(
    mae_filepath: str | Path,
    query_dictionary: dict,
    request_string: str,
    keywords_available: set,
    requests_available: set,
    ):

    '''
    Stochastic planner for MAE topology queries.

    Gets the atom-count estimate from the typing planner (same m_atom
    predicate, same CT-proportional sampling), then applies the correct
    bytes-per-match for the topology request. bonded_with constraints are
    not estimated — the plan is returned without them, which gives a
    conservative upper-bound estimate.

    Parameters
    ----------
    mae_filepath : str | Path
    query_dictionary : dict
    request_string : str
    keywords_available : set
    requests_available : set

    Returns
    -------
    dict
        Planner metadata.
    '''

    if request_string not in requests_available:
        raise ValueError(
            f'Unsupported request_string {request_string!r}. '
            f'Available requests: {sorted(requests_available)}'
        )

    output_kind, trailing_shape, bytes_per_match = _get_mae_topo_request_plan_shape(request_string)

    if output_kind == 'scalar_property':
        return {
            'planner_mode': 'stochastic',
            'file_type':    'mae',
            'request':      request_string,
            'supported':    False,
            'reason':       f'{request_string!r} is a scalar property, not estimated.',
            'query_dictionary': query_dictionary,
        }

    # Strip bonded_with before passing to typing planner — bonds are not
    # estimated, giving a conservative upper-bound on matched atom count.
    atom_query = {
        k: v for k, v in query_dictionary.items()
        if k not in ('bonded_with', 'bonded_with_mode')
    }

    type_keys, type_reqs = _get_type_keys_reqs_mae(mae_filepath)
    type_plan = _plan_type_query_mae(
        mae_filepath=mae_filepath,
        query_dictionary=atom_query,
        request_string='global_ids',   # always per_atom, gives us n_atoms
        keywords_available=type_keys,
        requests_available=type_reqs,
    )

    # Re-apply the topology bytes-per-match for the actual request
    estimated_matches = type_plan.get('n_atoms', 0)
    estimated_bytes   = estimated_matches * bytes_per_match

    return {
        **type_plan,
        'estimated_bytes': estimated_bytes,
        'estimated_mib':   estimated_bytes / (1024 ** 2),
        'bonded_with_estimated': False,
    }

# FINISH 
def _get_topology_query_mae(
    mae_filepath: str | Path,
    query_dictionary: dict,
    request_string: str,
    keywords_available: set,
    requests_available: set,
    ):

    '''
    Execute a topology query against a MAE file.

    Reads from m_atom (same atom dict as typing) and applies bonded_with
    filtering via m_bond when requested. ffio_sites fields (charges, masses,
    vdw_types) are resolved via _iter_over_m_atoms_with_force_field_sites using a separate
    ffio predicate so that filtering on 'charge' correctly tests ffio_charge
    rather than formal_charge.

    Parameters
    ----------
    mae_filepath : str | Path
    query_dictionary : dict
    request_string : str
    keywords_available : set
    requests_available : set

    Returns
    -------
    list
        Depends on request_string.
    '''

    # Strip bonded_with keys before passing to the atom predicate so they
    # are not treated as atom-field constraints.
    atom_query = {
        k: v for k, v in query_dictionary.items()
        if k not in ('bonded_with', 'bonded_with_mode')
    }

    predicate_state = _get_m_atom_predicate(atom_query)
    ffio_ps         = _get_force_field_predicate(atom_query)

    # Use sites iterator only when ffio predicate is active or request
    # requires ffio fields. For pure m_atom requests use the lighter iterator.
    need_sites = (
        ffio_ps['need_chg'] or ffio_ps['need_mass'] or ffio_ps['need_vdw']
        or request_string in {'charges', 'masses', 'vdw_types', 'property-system_charge'}
    )

    def _matched():
        it = (
            _iter_over_m_atoms_with_force_field_sites(mae_filepath)
            if need_sites else
            _iter_over_m_atoms(mae_filepath)
        )
        for atom in it:
            if _does_m_atom_query_match(atom, predicate_state) and _does_force_field_query_match(atom, ffio_ps):
                yield atom


    def _resolve_global_ids_with_bonds(base_ids: list[int]) -> list[int]:
        bond_inc, bond_exc = query_dictionary.get('bonded_with', ([], []))
        bond_mode, _ = query_dictionary.get('bonded_with_mode', ('all', None))
        if not bond_inc and not bond_exc:
            return base_ids
        return _filter_mae_by_bonded_with(
            mae_filepath=mae_filepath,
            candidate_globals=base_ids,
            include_blocks=bond_inc,
            exclude_blocks=bond_exc,
            mode=bond_mode,
            query_dictionary=query_dictionary,
            keywords_available=keywords_available,
            requests_available=requests_available,
        )

    match request_string:

        case 'global_ids':
            base_ids = [a['global_id'] for a in _matched()]
            return _resolve_global_ids_with_bonds(base_ids)

        case 'local_ids':
            return [a['local_id'] for a in _matched()]

        case 'atom_names':
            return [a['atom_name'] for a in _matched()]

        case 'atom_names_full':
            return [a['atom_name_full'] for a in _matched()]

        case 'residue_names':
            return [a['residue_name'] for a in _matched()]

        case 'residue_ids':
            return [a['residue_id'] for a in _matched()]

        case 'chain_names':
            return [a['chain_name'] for a in _matched()]

        case 'segment_names':
            return [a['segment_name'] for a in _matched()]

        case 'atomic_numbers':
            return [a['atomic_number'] for a in _matched()]

        case 'mmod_types':
            return [a['mmod_type'] for a in _matched()]

        case 'colors':
            return [a['color'] for a in _matched()]

        case 'visibilities':
            return [a['visibility'] for a in _matched()]

        case 'formal_charges':
            return [a['formal_charge'] for a in _matched()]

        case 'partial_charges_1':
            return [a['partial_charge_1'] for a in _matched()]

        case 'partial_charges_2':
            return [a['partial_charge_2'] for a in _matched()]

        case 'mmod_res':
            return [a['mmod_res'] for a in _matched()]

        case 'grow_names':
            return [a['grow_name'] for a in _matched()]

        case 'insertion_codes':
            return [a['insertion_code'] for a in _matched()]

        case 'secondary_structures':
            return [a['secondary_structure'] for a in _matched()]

        case 'pdb_tfactors':
            return [a['pdb_tfactor'] for a in _matched()]

        case 'pdb_occupancies':
            return [a['pdb_occupancy'] for a in _matched()]

        case 'h_counts':
            return [a['h_count'] for a in _matched()]

        case 'representations':
            return [a['representation'] for a in _matched()]

        case 'template_indices':
            return [a['template_index'] for a in _matched()]

        case 'x':
            return [a['x'] for a in _matched()]

        case 'y':
            return [a['y'] for a in _matched()]

        case 'z':
            return [a['z'] for a in _matched()]

        case 'v_x':
            return [a['v_x'] for a in _matched()]

        case 'v_y':
            return [a['v_y'] for a in _matched()]

        case 'v_z':
            return [a['v_z'] for a in _matched()]

        case 'positions':
            rows = [(a['x'], a['y'], a['z']) for a in _matched()]
            arr = np.array(rows, dtype=np.float32).reshape(-1, 3)
            return arr[np.newaxis, :, :]  # (1, n, 3)

        case 'velocities':
            rows = [(a['v_x'], a['v_y'], a['v_z']) for a in _matched()]
            arr = np.array(rows, dtype=np.float32).reshape(-1, 3)
            return arr[np.newaxis, :, :]  # (1, n, 3)

        case 'property-number_of_atoms':
            return sum(1 for _ in _iter_over_m_atoms(mae_filepath))

        case 'property-number_of_residues':
            return len({a['residue_id'] for a in _iter_over_m_atoms(mae_filepath)})

        case 'property-box_size':
            xs, ys, zs = [], [], []
            for a in _iter_over_m_atoms(mae_filepath):
                xs.append(a['x'])
                ys.append(a['y'])
                zs.append(a['z'])
            if not xs:
                raise ValueError('No atom records found in MAE file.')
            return (min(xs), max(xs), min(ys), max(ys), min(zs), max(zs))

        case 'property-system_formal_charge':
            return sum(a['formal_charge'] for a in _iter_over_m_atoms(mae_filepath))

        # -------------------------------------------------------------------------
        # ffio_sites fields — charge, mass, vdw_type
        # Filtering on 'charge'/'mass'/'vdw_type' keywords is handled by
        # _does_force_field_query_match via the shared _matched() above.
        # -------------------------------------------------------------------------

        case 'charges':
            return [a['ffio_charge'] for a in _matched()]

        case 'masses':
            return [a['ffio_mass'] for a in _matched()]

        case 'vdw_types':
            return [a['ffio_vdw_type'] for a in _matched()]

        case 'property-system_charge':
            # Walk all atoms through the site map — never sum unique sites
            # directly, as that would give the wrong answer for template CTs
            # where n_atoms != n_sites.
            return sum(a['ffio_charge'] for a in _iter_over_m_atoms_with_force_field_sites(mae_filepath))

        case _:
            raise ValueError(f'Unsupported request_string for MAE topology: {request_string!r}')



# =========================================================================
# TOPOLOGY HELPERS
# =========================================================================


def _iter_over_m_atoms_with_force_field_sites(mae_filepath: str | Path):

    '''
    Yield atom dicts from ALL f_m_ct blocks, each augmented with FF site
    properties (ffio_charge, ffio_mass, ffio_vdw_type) from the corresponding
    ffio_sites block in the same CT, matched by ct_index.

    The site mapping uses a repeating-pattern scheme per CT:

        site_index = ct_local_atom_index % len(ct_site_list)

    If a CT has no ffio_sites block, ffio_charge and ffio_mass default to
    0.0 and ffio_vdw_type to ''.

    Parameters
    ----------
    mae_filepath : str | Path

    Yields
    ------
    dict
    '''

    # Build ct_index → site_list from ffio_sites blocks
    ct_sites: dict[int, list[dict]] = {}

    for ct_index, keys, rows in _iter_mae_ffio_block_rows(mae_filepath, 'ffio_sites'):

        col_idx: dict[str, int | None] = {
            atom_key: (keys.index(ffio_col) if ffio_col in keys else None)
            for atom_key, ffio_col in _MAE_FFIO_KEY_TO_COL.items()
        }

        def _get(vals: list[str], atom_key: str, default: str = '') -> str:
            i = col_idx[atom_key]
            if i is None or i >= len(vals):
                return default
            return vals[i]

        ct_sites[ct_index] = [
            {
                'ffio_charge':   float(_get(row[1:], 'ffio_charge',   '0.0') or 0.0),
                'ffio_mass':     float(_get(row[1:], 'ffio_mass',     '0.0') or 0.0),
                'ffio_vdw_type': _get(row[1:], 'ffio_vdw_type', '').strip(),
            }
            for row in rows
        ]

    # Yield atoms from all m_atom CTs, augmented with their matched CT sites
    empty         = {'ffio_charge': 0.0, 'ffio_mass': 0.0, 'ffio_vdw_type': ''}
    global_offset = 0

    for ct_index, keys, rows in _iter_mae_block_rows(mae_filepath, 'm_atom'):

        col_idx: dict[str, int | None] = {
            atom_key: (keys.index(mae_col) if mae_col in keys else None)
            for atom_key, mae_col in _MAE_ATOM_KEY_TO_COL.items()
        }

        site_list = ct_sites.get(ct_index, [])
        n_sites   = len(site_list)

        for ct_local_id, row in enumerate(rows):
            atom              = _parse_token_to_dict(row, col_idx)
            atom['global_id'] = global_offset + ct_local_id
            site              = site_list[ct_local_id % n_sites] if n_sites else empty
            yield {**atom, **site}

        global_offset += len(rows)


def _get_force_field_predicate(query_dictionary: dict) -> dict:

    '''
    Pre-compute include/exclude ranges for ffio_sites fields from the
    topology query dictionary.

    This is separate from _get_m_atom_predicate because ffio fields
    (ffio_charge, ffio_mass, ffio_vdw_type) exist only on the augmented dict
    yielded by _iter_over_m_atoms_with_force_field_sites, not on the plain m_atom dict.
    Keeping them separate means the two predicate systems never collide —
    the m_atom predicate filters on formal_charge, the ffio predicate filters
    on ffio_charge.

    Parameters
    ----------
    query_dictionary : dict

    Returns
    -------
    dict
    '''

    def _r(key):
        return qh._normalise_query_pair(query_dictionary.get(key), range_style=True)

    def _s(key):
        return qh._normalise_query_pair(query_dictionary.get(key))

    chg_inc,  chg_exc  = _r('charge')
    mass_inc, mass_exc = _r('mass')
    vdw_inc,  vdw_exc  = _s('vdw_type')

    return {
        'chg_inc':  chg_inc,  'chg_exc':  chg_exc,
        'mass_inc': mass_inc, 'mass_exc': mass_exc,
        'vdw_inc':  vdw_inc,  'vdw_exc':  vdw_exc,
        'need_chg':  chg_inc  != (None, None) or chg_exc  != (None, None),
        'need_mass': mass_inc != (None, None) or mass_exc != (None, None),
        'need_vdw':  bool(vdw_inc or vdw_exc),
    }


def _does_force_field_query_match(atom: dict, ffio_ps: dict) -> bool:

    '''
    Predicate for ffio_sites fields on the augmented dict from
    _iter_over_m_atoms_with_force_field_sites.

    Parameters
    ----------
    atom : dict
        Augmented atom dict containing ffio_charge, ffio_mass, ffio_vdw_type.
    ffio_ps : dict
        Predicate state from _get_force_field_predicate.

    Returns
    -------
    bool
    '''

    mr = qh._match_range_scalar
    m  = qh._match
    ps = ffio_ps
    ok = True

    if ps['need_chg']:          ok = mr(atom['ffio_charge'], ps['chg_inc'],  ps['chg_exc'])
    if ok and ps['need_mass']:  ok = mr(atom['ffio_mass'],   ps['mass_inc'], ps['mass_exc'])
    if ok and ps['need_vdw']:   ok = m(atom['ffio_vdw_type'], ps['vdw_inc'], ps['vdw_exc'])

    return ok



# =========================================================================
# BOND LOGIC
# =========================================================================

def _filter_mae_by_bonded_with(
    mae_filepath: str | Path,
    candidate_globals: list[int],
    include_blocks: list[dict],
    exclude_blocks: list[dict],
    mode: str,
    query_dictionary: dict,
    keywords_available: set,
    requests_available: set,
    ) -> list[int]:

    '''
    Filter candidate global IDs by bonded-with constraints using the m_bond
    block.  Mirrors the PSF bonded_with API.

    Only ``total`` bond-count constraints and per-neighbor-set constraints
    (resolved recursively via ``global_ids`` queries) are supported.

    Parameters
    ----------
    mae_filepath : str | Path
    candidate_globals : list[int]
    include_blocks : list[dict]
    exclude_blocks : list[dict]
    mode : str  ("all" | "any")
    query_dictionary : dict
    keywords_available : set
    requests_available : set

    Returns
    -------
    list[int]
        Filtered candidate global IDs.
    '''

    if mode not in ('all', 'any'):
        raise ValueError("bonded_with_mode must be 'all' or 'any'.")

    n_cand = len(candidate_globals)
    if n_cand == 0:
        return []

    # Build local_id -> global_id map from m_atom block
    local_to_global: dict[int, int] = {}
    for atom in _iter_over_m_atoms(mae_filepath):
        local_to_global[atom['local_id']] = atom['global_id']

    natom = max(local_to_global.values()) + 1 if local_to_global else 0

    cand_set = set(candidate_globals)
    cand_index = {g: i for i, g in enumerate(candidate_globals)}

    def _parse_cmp(count_dict: dict) -> tuple[str, int]:
        if not isinstance(count_dict, dict) or len(count_dict) != 1:
            raise ValueError('bonded_with block count must be a dict with exactly one comparator.')
        (op, v), = count_dict.items()
        if op not in ('eq', 'ne', 'ge', 'le', 'gt', 'lt'):
            raise ValueError(f'Unsupported comparator: {op!r}')
        return op, int(v)

    def _cmp(x: int, op: str, v: int) -> bool:
        if op == 'eq': return x == v
        if op == 'ne': return x != v
        if op == 'ge': return x >= v
        if op == 'le': return x <= v
        if op == 'gt': return x > v
        if op == 'lt': return x < v
        raise ValueError(f'Unsupported comparator: {op!r}')

    def _resolve_neighbor_set(block: dict) -> set[int] | None:
        if block.get('total', False):
            return None
        neighbor_q = dict(block.get('neighbor', {}))
        neighbor_q.pop('bonded_with', None)
        neighbor_q.pop('bonded_with_mode', None)
        ids = _get_topology_query_mae(
            mae_filepath=mae_filepath,
            query_dictionary=neighbor_q,
            request_string='global_ids',
            keywords_available=keywords_available,
            requests_available=requests_available,
        )
        return set(ids)

    # Resolve neighbor sets
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

    for block in include_blocks:
        op, val = _parse_cmp(block['count'])
        if block.get('total', False):
            inc_norm.append(('total', op, val))
        else:
            s = _resolve_neighbor_set(block)
            inc_norm.append(('set', _col_for_set(s), op, val))

    for block in exclude_blocks:
        op, val = _parse_cmp(block['count'])
        if block.get('total', False):
            exc_norm.append(('total', op, val))
        else:
            s = _resolve_neighbor_set(block)
            exc_norm.append(('set', _col_for_set(s), op, val))

    if not inc_norm and not exc_norm:
        return candidate_globals

    # Build neighbor-membership bitmask array
    neighbor_mask = np.zeros(natom, dtype=object)
    for col, s in enumerate(unique_sets):
        bit = 1 << col
        for g in s:
            if g < natom:
                neighbor_mask[g] = int(neighbor_mask[g]) | bit

    counts_total   = np.zeros(n_cand, dtype=np.int32)
    counts_by_set  = np.zeros((n_cand, len(unique_sets)), dtype=np.int32) if unique_sets else None

    def _bump(i: int, other_global: int):
        counts_total[i] += 1
        if counts_by_set is None or other_global >= natom:
            return
        m = int(neighbor_mask[other_global])
        while m:
            lsb = m & -m
            k = lsb.bit_length() - 1
            counts_by_set[i, k] += 1
            m ^= lsb

    # MAE m_bond lists every physical bond twice (A->B and B->A).
    # We deduplicate by only processing each pair once (frm_local < to_local).
    for frm_local, to_local, _ in _iter_over_m_bonds(mae_filepath):
        if frm_local >= to_local:   # skip the reverse-direction duplicate
            continue
        a_g = local_to_global.get(frm_local)
        b_g = local_to_global.get(to_local)
        if a_g is not None and a_g in cand_set:
            if b_g is not None:
                _bump(cand_index[a_g], b_g)
        if b_g is not None and b_g in cand_set:
            if a_g is not None:
                _bump(cand_index[b_g], a_g)

    def _eval_one(i: int, c: tuple) -> bool:
        if c[0] == 'total':
            _, op, v = c
            return _cmp(int(counts_total[i]), op, v)
        _, col, op, v = c
        return _cmp(int(counts_by_set[i, col]), op, v)

    mask = np.ones(n_cand, dtype=bool)

    if inc_norm:
        if mode == 'all':
            for i in range(n_cand):
                for c in inc_norm:
                    if not _eval_one(i, c):
                        mask[i] = False
                        break
        else:  # any
            for i in range(n_cand):
                ok_any = any(_eval_one(i, c) for c in inc_norm)
                mask[i] = ok_any

    if exc_norm:
        for i in range(n_cand):
            if not mask[i]:
                continue
            for c in exc_norm:
                if _eval_one(i, c):
                    mask[i] = False
                    break

    return [g for g, ok in zip(candidate_globals, mask) if ok]


def _iter_over_m_bonds(mae_filepath: str | Path):

    '''
    Yield (from_local, to_local, order) tuples from the m_bond block of the
    first f_m_ct.

    Local IDs correspond to ``local_id`` in the m_atom rows (1-based).

    Parameters
    ----------
    mae_filepath : str | Path

    Yields
    ------
    tuple[int, int, int]
        (from_local_id, to_local_id, bond_order)
    '''

    for _ct_index, keys, rows in _iter_mae_block_rows(mae_filepath, 'm_bond'):

        def _idx(name: str) -> int | None:
            return keys.index(name) if name in keys else None

        idx_from  = _idx('i_m_from')
        idx_to    = _idx('i_m_to')
        idx_order = _idx('i_m_order')

        for row in rows:
            try:
                vals = row[1:]
                frm   = int(vals[idx_from])  if idx_from  is not None else 0
                to    = int(vals[idx_to])    if idx_to    is not None else 0
                order = int(vals[idx_order]) if idx_order is not None else 1
                yield frm, to, order
            except (ValueError, IndexError):
                continue


# size of the plan 
def _get_mae_topo_request_plan_shape(request_string: str) -> tuple[str, tuple, int | None]:

    '''
    Return (output_kind, trailing_shape, bytes_per_match) for the topology planner.

    Strict superset of the typing plan shape — all typing requests are valid
    here plus ffio_sites requests.

    Parameters
    ----------
    request_string : str

    Returns
    -------
    tuple[str, tuple, int | None]
    '''

    # All typing requests are valid for topology too — delegate first
    try:
        return _get_mae_type_request_plan_shape(request_string)
    except ValueError:
        pass

    # Topology-only additions from ffio_sites
    match request_string:
        case 'charges':                return 'per_atom', (), 8
        case 'masses':                 return 'per_atom', (), 8
        case 'vdw_types':              return 'per_atom', (), 16
        case 'property-system_charge': return 'scalar_property', (), None
        case _:
            raise ValueError(f'Unsupported request_string for mae topo planner: {request_string!r}')

# get all topo keys
def _get_all_mae_force_field_keys(mae_filepath: str | Path) -> dict[str, set[str]]:

    '''
    Walk every f_m_ct block in the file and return a mapping of ffio
    sub-block name to the union of all column key lists seen for that
    block type across all CTs.

    Only sub-blocks listed in _FFIO_BLOCKS_OF_INTEREST are collected.
    The ``ffio_ff`` container block itself is excluded since it holds
    scalar metadata rather than per-atom data columns.

    This is header-only (no data rows are read) so cost is O(number of
    block headers).

    File structure expected:
        depth 1 — f_m_ct {
        depth 2 — ffio_ff {
        depth 3 — ffio_sites[N] {  (and other ffio sub-blocks)

    Parameters
    ----------
    mae_filepath : str | Path

    Returns
    -------
    dict[str, set[str]]
        Mapping of ffio block name (e.g. ``"ffio_sites"``) to the set
        of column names seen across all CTs. A block absent from the
        file will not appear as a key.

    Examples
    --------
    Checking whether charge data is available::

        ffio = _get_all_mae_force_field_keys(path)
        has_charge = 'r_ffio_charge' in ffio.get('ffio_sites', set())
    '''
    
    _FFIO_BLOCKS_OF_INTEREST: set[str] = {'ffio_sites'} # change this for future ffio block types

    mae_filepath = Path(mae_filepath)
    result: dict[str, set[str]] = {}

    depth         = 0
    in_fmct       = False
    in_ffio_ff    = False
    current_name: str | None = None
    reading_keys  = False

    with open(mae_filepath, 'rt', encoding='utf-8', errors='replace') as f:
        for raw in f:
            line = raw.strip()

            if not line:
                continue

            opens  = line.endswith('{')
            closes = line == '}'

            #  entering a block 
            if opens:
                depth += 1
                base = line.split('[')[0].split('{')[0].strip()

                if depth == 1 and base == 'f_m_ct':
                    in_fmct = True

                elif depth == 2 and in_fmct and base == 'ffio_ff':
                    in_ffio_ff = True

                elif depth == 3 and in_ffio_ff:
                    if base in _FFIO_BLOCKS_OF_INTEREST:
                        current_name = base
                        reading_keys = True
                        if current_name not in result:
                            result[current_name] = set()

                continue

            #  closing a block 
            if closes:
                if reading_keys:
                    reading_keys = False
                    current_name = None

                depth -= 1

                if depth == 1 and in_ffio_ff:
                    in_ffio_ff = False

                if depth == 0 and in_fmct:
                    in_fmct = False

                continue

            # inside a block 
            if reading_keys:
                if line == ':::':
                    reading_keys = False
                    current_name = None
                else:
                    result[current_name].add(line)

    return result

