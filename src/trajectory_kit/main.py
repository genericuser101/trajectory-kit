# Preloaded Library
import os
from pathlib import Path
import importlib

# Third Party Imports: (numpy)
import numpy as np
from trajectory_kit import _file_parse_help as fph
from trajectory_kit import _query_help as qh

class sim():

    def __init__(self, 
                 typing     : str | Path  = None, 
                 topology   : str | Path  = None, 
                 trajectory : str | Path  = None,
                 *, 
                 verbose             : bool        = False, 
                 globals_dictionary  : dict        = None,) -> None:
        
        '''
        The sim class needs to have at least a typing file to be initialised:
        These files are .xyz, .data, .pdb

        Optionally, a topology file can be provided, which can be .psf, .prmtop, .top

        Finally, a trajectory file can be provided, which can be .dcd, .xtc, .trr

        Parameters:
        ----------

        typing: str | Path, optional
            The file path to the typing file.
        
        topology: str | Path, optional
            The file path to the topology file.

        trajectory: str | Path, optional
            The file path to the trajectory file.

        Returns:
        -------
        None
        
        '''

        self.verbose = verbose

        # explicit empty state
        self.type_file = None
        self.top_file = None
        self.traj_file = None

        self.type_type = None
        self.top_type = None
        self.traj_type = None

        self.type_file_keys, self.type_file_reqs = set(), set()
        self.topo_file_keys, self.topo_file_reqs = set(), set()
        self.traj_file_keys, self.traj_file_reqs = set(), set()

        self._module_cache: dict[str, object] = {}

        # Supported file formats and associated function templates for each domain
        self._domain_registry = {

            "typing": {
                "supported_formats": {".pdb", ".xyz", ".mae"},
                "file_attr": "type_file",
                "type_attr": "type_type",
                "keys_attr": "type_file_keys",
                "reqs_attr": "type_file_reqs",
                "keys_fn_template":       "_get_type_keys_reqs_{fmt}",
                "plan_fn_template":       "_plan_type_query_{fmt}",
                "plan_shape_fn_template": "_get_type_plan_shape_{fmt}",
                "query_fn_template":      "_get_type_query_{fmt}",
                "update_fn_template":     "_update_type_globals_{fmt}",
                "label": "typing",
            },

            "topology": {
                "supported_formats": {".psf", ".mae"},
                "file_attr": "top_file",
                "type_attr": "top_type",
                "keys_attr": "topo_file_keys",
                "reqs_attr": "topo_file_reqs",
                "keys_fn_template":       "_get_topology_keys_reqs_{fmt}",
                "plan_fn_template":       "_plan_topology_query_{fmt}",
                "plan_shape_fn_template": "_get_topology_plan_shape_{fmt}",
                "query_fn_template":      "_get_topology_query_{fmt}",
                "update_fn_template":     "_update_topology_globals_{fmt}",
                "label": "topology",
            },

            "trajectory": {
                "supported_formats": {".dcd", ".coor"},
                "file_attr": "traj_file",
                "type_attr": "traj_type",
                "keys_attr": "traj_file_keys",
                "reqs_attr": "traj_file_reqs",
                "keys_fn_template":       "_get_trajectory_keys_reqs_{fmt}",
                "plan_fn_template":       "_plan_trajectory_query_{fmt}",
                "plan_shape_fn_template": "_get_trajectory_plan_shape_{fmt}",
                "query_fn_template":      "_get_trajectory_query_{fmt}",
                "update_fn_template":     "_update_trajectory_globals_{fmt}",
                "label": "trajectory",
            },
            
        }

        self.global_system_properties = {
            "sim_name": None,
            "timestep": None,
            "num_atoms": None,
            "num_residues": None,
            "num_frames": None,
            "num_segments": None,
            "ensemble_type": None,
            "simulation_type": None,
            "start_box_size": None,
        }

        self._match_system_properties(globals_dictionary)

        if typing is not None:
            self.load_typing(typing)

        if topology is not None:
            self.load_topology(topology)

        if trajectory is not None:
            self.load_trajectory(trajectory)

        


    # -------------------------------------------------------------------------
    # SYSTEM INFO
    # -------------------------------------------------------------------------

    def get_type_keys(self) -> set:
        
        '''
        Return the set of available keywords in the currently loaded typing file.

        Parameters:
        ----------
        None

        Returns:
        -------
        set
            A set of available keywords in the currently loaded typing file.

        Raises:
        ------
        None
        '''

        return self.type_file_keys

    def get_topo_keys(self) -> set:
        
        '''
        Return the set of available keywords in the currently loaded topology file.

        Parameters:
        ----------
        None

        Returns:
        -------
        set
            A set of available keywords in the currently loaded topology file.

        Raises:
        ------
        None
        '''

        return self.topo_file_keys

    def get_traj_keys(self) -> set:
        
        '''
        Return the set of available keywords in the currently loaded trajectory file.

        Parameters:
        ----------
        None

        Returns:
        -------
        set
            A set of available keywords in the currently loaded trajectory file.

        Raises:
        ------
        None
        '''

        return self.traj_file_keys


    def get_type_reqs(self) -> set:
        
        '''
        Return the set of available requests in the currently loaded typing file.

        Parameters:
        ----------
        None

        Returns:
        -------
        set
            A set of available requests in the currently loaded typing file.

        Raises:
        ------
        None
        '''

        return self.type_file_reqs

    def get_topo_reqs(self) -> set:
        
        '''
        Return the set of available requests in the currently loaded topology file.

        Parameters:
        ----------
        None

        Returns:
        -------
        set
            A set of available requests in the currently loaded topology file.

        Raises:
        ------
        None
        '''

        return self.topo_file_reqs

    def get_traj_reqs(self) -> set:
        
        '''
        Return the set of available requests in the currently loaded trajectory file.

        Parameters:
        ----------
        None

        Returns:
        -------
        set
            A set of available requests in the currently loaded trajectory file.

        Raises:
        ------
        None
        '''

        return self.traj_file_reqs


    def print_info(self) -> None:

        '''
        Function to print out the current state of the sim object, including loaded files, their types, and available properties.

        Parameters:
        ----------
        None

        Returns:
        -------
        None

        Raises:
        ------
        None
        '''

        def _fmt_file(f, t):
            return f"{f.name}  ({t})" if f is not None else "—"

        def _sorted_split(s):
            reqs     = sorted(r for r in s if not r.startswith("property-"))
            props    = sorted(r for r in s if r.startswith("property-"))
            return reqs + [f"({r})" for r in props]

        type_keys  = sorted(self.type_file_keys)  if self.type_file  else []
        top_keys   = sorted(self.topo_file_keys)  if self.top_file   else []
        traj_keys  = sorted(self.traj_file_keys)  if self.traj_file  else []

        type_reqs  = _sorted_split(self.type_file_reqs)  if self.type_file  else []
        top_reqs   = _sorted_split(self.topo_file_reqs)  if self.top_file   else []
        traj_reqs  = _sorted_split(self.traj_file_reqs)  if self.traj_file  else []

        # column widths
        W0 = 10   # row label (keywords / requests)
        W1 = max((len(s) for s in type_keys  + type_reqs),  default=0) + 2
        W2 = max((len(s) for s in top_keys   + top_reqs),   default=0) + 2
        W3 = max((len(s) for s in traj_keys  + traj_reqs),  default=0) + 2
        W1 = max(W1, len("typing") + 2)
        W2 = max(W2, len("topology") + 2)
        W3 = max(W3, len("trajectory") + 2)

        sep = f"  {'─' * W0}  {'─' * W1}  {'─' * W2}  {'─' * W3}"

        def _row(label, c1, c2, c3):
            print(f"  {label:<{W0}}  {c1:<{W1}}  {c2:<{W2}}  {c3:<{W3}}")

        print("\n=== SIMULATION INFO ===")

        print("\n  files")
        print(f"    typing     {_fmt_file(self.type_file,  self.type_type)}")
        print(f"    topology   {_fmt_file(self.top_file,   self.top_type)}")
        print(f"    trajectory {_fmt_file(self.traj_file,  self.traj_type)}")

        active_props = {k: v for k, v in self.global_system_properties.items() if v is not None}
        if active_props:
            print("\n  system properties")
            w = max(len(k) for k in active_props) + 2
            for k, v in active_props.items():
                print(f"    {k:<{w}} {v}")

        print("\n  available keywords and requests")
        print(sep)
        _row("",          "typing",   "topology",   "trajectory")
        print(sep)

        _row("keywords",  type_keys[0]  if type_keys  else "—",
                        top_keys[0]   if top_keys   else "—",
                        traj_keys[0]  if traj_keys  else "—")
        n_keys = max(len(type_keys), len(top_keys), len(traj_keys))
        for i in range(1, n_keys):
            _row("", type_keys[i]  if i < len(type_keys)  else "",
                    top_keys[i]   if i < len(top_keys)   else "",
                    traj_keys[i]  if i < len(traj_keys)  else "")

        print(sep)

        _row("requests",  type_reqs[0]  if type_reqs  else "—",
                        top_reqs[0]   if top_reqs   else "—",
                        traj_reqs[0]  if traj_reqs  else "—")
        n_reqs = max(len(type_reqs), len(top_reqs), len(traj_reqs))
        for i in range(1, n_reqs):
            _row("", type_reqs[i]  if i < len(type_reqs)  else "",
                    top_reqs[i]   if i < len(top_reqs)   else "",
                    traj_reqs[i]  if i < len(traj_reqs)  else "")

        print()


    def add_info(self, info_dict: dict) -> bool:

        '''
        Function to add or update global system properties in the sim object.

        Parameters:
        ----------
        info_dict: dict
            A dictionary containing global system properties and their values to be added or updated.

        Returns:
        -------
        bool
            True if the global system properties are successfully updated, otherwise False.
        '''

        return self._match_system_properties(info_dict)


    # -------------------------------------------------------------------------
    # USER INTERFACE FUNCTIONS 
    # -------------------------------------------------------------------------

    def positions(self,
                  TYPE_Q: dict = None,
                  TOPO_Q: dict = None,
                  TRAJ_Q: dict = None,
                  *,
                  devFlag: bool = False,
                  updateFlag: bool = False,
                  planFlag: bool = False   ) -> dict:
        
        '''
        Return positions for atoms selected by typing and/or topology queries.

        When a trajectory file is loaded, positions are read from it (all
        selected frames). When no trajectory file is loaded, positions are
        read directly from the static typing or topology file, and the payload
        has shape ``(1, n_atoms, 3)`` to keep the API uniform.

        Fallback source priority (no trajectory):
        -----------------------------------------
        1. Typing file  (if loaded and supports "positions")
        2. Topology file (if loaded and supports "positions")
        A ValueError is raised if neither loaded file supports positions.

        Selection rules:
        ----------------
        - TYPE_Q only      -> typing selection
        - TOPO_Q only      -> topology selection
        - TYPE_Q + TOPO_Q  -> intersection of both selections
        - neither provided -> all atoms in the source file

        Parameters:
        ----------
        TYPE_Q, TOPO_Q, TRAJ_Q : dict, optional
            Query dictionaries for typing, topology, and trajectory respectively.
            TRAJ_Q is ignored when no trajectory file is loaded.

        updateFlag : bool, default=False
            Passed through to the underlying domain handlers.

        planFlag : bool, default=False
            If True, return the envelope without the payload (plan only).

        Returns:
        -------
        dict
            Standardised five-key envelope:
                mode      = "positions"
                selection = per-domain selection block (intersection mode)
                metadata  = per-domain file metadata for every loaded file
                plan      = single-domain plan for whichever domain produced
                            the coordinates (trajectory if loaded, else the
                            chosen static fallback)
                payload   = ndarray of shape (n_frames, n_atoms_selected, 3),
                            or None when planFlag=True
        '''

        TYPE_Q = self._normalise_query(TYPE_Q)
        TOPO_Q = self._normalise_query(TOPO_Q)
        TRAJ_Q = self._normalise_query(TRAJ_Q)

        type_provided = bool(TYPE_Q)
        topo_provided = bool(TOPO_Q)
        traj_provided = bool(TRAJ_Q)
        static_domain = None

        # ------------------------------------------------------------------ #
        # Decide which domain produces the coordinates and build the plan.   #
        # ------------------------------------------------------------------ #

        if self.traj_file is not None:
            if not type_provided and not topo_provided:
                if self.type_file is None and self.top_file is None:
                    raise ValueError(
                        "positions() requires at least one of: a typing file, a topology "
                        "file, or explicit TYPE_Q / TOPO_Q to define the atom selection."
                    )
            source_domain = "trajectory"
            plan_query    = TRAJ_Q
        else:
            if self.type_file is not None and "positions" in self.type_file_reqs:
                static_domain = "typing"
            elif self.top_file is not None and "positions" in self.topo_file_reqs:
                static_domain = "topology"
            else:
                raise ValueError(
                    "No trajectory file is loaded, and no loaded typing/topology file "
                    "supports 'positions'. Load a trajectory file or a file type that "
                    "carries coordinates (e.g. PDB or XYZ as the typing file)."
                )
            source_domain = static_domain
            plan_query    = TYPE_Q if static_domain == "typing" else TOPO_Q

        if devFlag: # dev hotpath

            plan_result = self._plan_domain_request(
                domain=source_domain,
                query_dictionary=plan_query,
                request_string="positions",
            )
            plan = {source_domain: plan_result} if plan_result is not None else {}

            # Inject the cross-domain combined estimate. For positions() this
            # rolls up a single domain so combined mirrors that one entry.
            combined = self._build_combined_plan_estimate(plan)
            if combined is not None:
                plan["combined"] = combined

            metadata = self._build_metadata_for_loaded_domains()

            # planFlag short-circuit — return envelope without the payload.
            # Selection block carries provided-flags but no resolved counts yet.
            if planFlag:
                selection = self._build_selection_block(
                    merge_mode="intersection",
                    type_q_provided=type_provided,
                    topo_q_provided=topo_provided,
                    traj_q_provided=traj_provided,
                    type_ids=None, topo_ids=None, traj_ids=None,
                    resolved_count=None,
                )
                return self._build_envelope(
                    mode="positions", selection=selection,
                    metadata=metadata, plan=plan, payload=None,
                )

        # ------------------------------------------------------------------ #
        # Resolve atom selection from typing / topology queries.             #
        # ------------------------------------------------------------------ #

        type_ids = None
        topo_ids = None
        traj_ids = None  # positions() never collects a trajectory id constraint

        if type_provided:
            type_ids = self.get_types(
                QUERY=TYPE_Q, REQUEST="global_ids", updateFlag=updateFlag,
            )
        if topo_provided:
            topo_ids = self.get_topology(
                QUERY=TOPO_Q, REQUEST="global_ids", updateFlag=updateFlag,
            )

        if type_ids is not None and topo_ids is not None:
            selected_globals = sorted(set(type_ids) & set(topo_ids))
            if not selected_globals:
                raise ValueError(
                    "Typing and topology selections produced an empty intersection."
                )
        elif type_ids is not None:
            selected_globals = sorted(type_ids)
        elif topo_ids is not None:
            selected_globals = sorted(topo_ids)
        else:
            selected_globals = None

        # ------------------------------------------------------------------ #
        # Trajectory path                                                     #
        # ------------------------------------------------------------------ #

        if self.traj_file is not None:

            if selected_globals is None:
                if self.type_file is not None:
                    selected_globals = self.get_types(
                        QUERY={}, REQUEST="global_ids", updateFlag=updateFlag,
                    )
                elif self.top_file is not None:
                    selected_globals = self.get_topology(
                        QUERY={}, REQUEST="global_ids", updateFlag=updateFlag,
                    )

            traj_query = dict(TRAJ_Q)
            traj_query["global_ids"] = (selected_globals, set())

            pos = self.get_trajectory(
                QUERY=traj_query,
                REQUEST="positions",
                updateFlag=updateFlag,
            )

            if devFlag: # dev hotpath

                selection = self._build_selection_block(
                    merge_mode="intersection",
                    type_q_provided=type_provided,
                    topo_q_provided=topo_provided,
                    traj_q_provided=traj_provided,
                    type_ids=type_ids, topo_ids=topo_ids, traj_ids=traj_ids,
                    resolved_count=len(selected_globals),
                )

                return self._build_envelope(
                    mode="positions", selection=selection,
                    metadata=metadata, plan=plan, payload=pos,
                )
            
            else:

                return pos

        # ------------------------------------------------------------------ #
        # Static path                                                        #
        # ------------------------------------------------------------------ #

        assert static_domain is not None, \
            "static_domain was not resolved — this is a bug. also how?"

        static_query = dict(TYPE_Q if static_domain == "typing" else TOPO_Q)

        if selected_globals is None:
            selected_globals = (
                type_ids if static_domain == "typing" and type_ids is not None
                else (topo_ids if static_domain == "topology" and topo_ids is not None
                      else self._execute_domain_request(
                          domain=static_domain, query_dictionary=static_query,
                          request_string="global_ids", updateFlag=updateFlag,
                      ))
            )

        pos = self._execute_domain_request(
            domain=static_domain,
            query_dictionary=static_query,
            request_string="positions",
            updateFlag=updateFlag,
        )

        if pos is not None and pos.shape[1] != len(selected_globals):
            # Static-file positions came back with all-matches for the static
            # query; we still need to reduce to the resolved selection if it
            # is narrower (e.g. typing AND topology both provided).
            all_ids = self._execute_domain_request(
                domain=static_domain,
                query_dictionary=static_query,
                request_string="global_ids",
                updateFlag=updateFlag,
            )
            gid_set = set(selected_globals)
            mask    = [i for i, g in enumerate(all_ids) if g in gid_set]
            pos     = pos[:, mask, :]

        if devFlag: # dev hotpath

            selection = self._build_selection_block(
                merge_mode="intersection",
                type_q_provided=type_provided,
                topo_q_provided=topo_provided,
                traj_q_provided=traj_provided,
                type_ids=type_ids, topo_ids=topo_ids, traj_ids=traj_ids,
                resolved_count=len(selected_globals),
            )

            return self._build_envelope(
                mode="positions", selection=selection,
                metadata=metadata, plan=plan, payload=pos,
            )
        
        else: 

            return pos
    
    
    def select(self,
                *,
                TYPE_Q: dict = None,
                TOPO_Q: dict = None,
                TRAJ_Q: dict = None,
                TYPE_R: str = None,
                TOPO_R: str = None,
                TRAJ_R: str = None,
                devFlag: bool = False,
                updateFlag: bool = False,
                planFlag: bool = False   ) -> dict:
        
        '''
        Return property values from typing, topology, and/or trajectory domains.

        This function is for property-style requests only (those starting with
        ``property-``). It does not return per-atom payloads or coordinate
        arrays — use ``fetch()`` or ``positions()`` for those.

        Parameters:
        ----------
        TYPE_Q, TOPO_Q, TRAJ_Q : dict, optional
            Query dictionaries for typing, topology, and trajectory respectively.
            (Currently unused for property requests, but accepted to preserve
            envelope symmetry across positions/fetch/select.)

        TYPE_R, TOPO_R, TRAJ_R : str, optional
            Property request strings for typing, topology, and trajectory.

        updateFlag : bool, default=False
            Passed through to the underlying domain handlers.

        planFlag : bool, default=False
            If True, return the envelope without the payload.

        Returns:
        -------
        dict
            Standardised five-key envelope. ``selection.merge_mode`` is
            ``"none"`` for select() — property requests do not perform
            cross-domain atom-set intersection.
        '''
        
        TYPE_Q = self._normalise_query(TYPE_Q)
        TOPO_Q = self._normalise_query(TOPO_Q)
        TRAJ_Q = self._normalise_query(TRAJ_Q)

        TYPE_R = self._normalise_request(TYPE_R)
        TOPO_R = self._normalise_request(TOPO_R)
        TRAJ_R = self._normalise_request(TRAJ_R)

        if TYPE_R is None and TOPO_R is None and TRAJ_R is None:
            raise ValueError("select() requires at least one property request string.")

        if TYPE_R is not None and self._classify_request("typing", TYPE_R) != "property":
            raise ValueError(
                f"TYPE_R='{TYPE_R}' is not a property request. "
                "Use positions() for coordinate payloads, fetch() for per-atom lists."
            )

        if TOPO_R is not None and self._classify_request("topology", TOPO_R) != "property":
            raise ValueError(
                f"TOPO_R='{TOPO_R}' is not a property request. "
                "Use fetch() for per-atom lists."
            )

        if TRAJ_R is not None and self._classify_request("trajectory", TRAJ_R) != "property":
            raise ValueError(
                f"TRAJ_R='{TRAJ_R}' is not a property request. "
                "Use positions() for trajectory coordinates."
            )

        if devFlag: # dev hotpath

            # Build per-domain plans for every requested domain.
            plan: dict = {}
            for label, q, r in (("typing", TYPE_Q, TYPE_R),
                                ("topology", TOPO_Q, TOPO_R),
                                ("trajectory", TRAJ_Q, TRAJ_R)):
                if r is not None:
                    result = self._plan_domain_request(
                        domain=label, query_dictionary=q, request_string=r,
                    )
                    if result is not None:
                        plan[label] = result

            # select() typically only handles property-* requests, which
            # short-circuit to None in _plan_domain_request — so plan is
            # usually empty here and combined will be omitted.
            combined = self._build_combined_plan_estimate(plan)
            if combined is not None:
                plan["combined"] = combined

            metadata = self._build_metadata_for_loaded_domains()

            # select() never intersects — selection block carries query_provided
            # flags only, with merge_mode="none" and ids_provided always False.
            selection = self._build_selection_block(
                merge_mode="none",
                type_q_provided=bool(TYPE_Q),
                topo_q_provided=bool(TOPO_Q),
                traj_q_provided=bool(TRAJ_Q),
                type_ids=None, topo_ids=None, traj_ids=None,
                resolved_count=None,
            )

            if planFlag:
                return self._build_envelope(
                    mode="select", selection=selection,
                    metadata=metadata, plan=plan, payload=None,
                )

        payload: dict = {}
        if TYPE_R is not None:
            payload["typing"] = self.get_types(
                QUERY=TYPE_Q, REQUEST=TYPE_R, updateFlag=updateFlag,
            )
        if TOPO_R is not None:
            payload["topology"] = self.get_topology(
                QUERY=TOPO_Q, REQUEST=TOPO_R, updateFlag=updateFlag,
            )
        if TRAJ_R is not None:
            payload["trajectory"] = self.get_trajectory(
                QUERY=TRAJ_Q, REQUEST=TRAJ_R, updateFlag=updateFlag,
            )

        if devFlag: # dev hotpath

            return self._build_envelope(
                mode="select", selection=selection,
                metadata=metadata, plan=plan, payload=payload,
            )
        
        else: 

            return payload
    

    def fetch(self,
              *,
              TYPE_Q: dict = None,
              TOPO_Q: dict = None,
              TRAJ_Q: dict = None,
              TYPE_R: str = None,
              TOPO_R: str = None,
              TRAJ_R: str = None,
              devFlag: bool = False,
              updateFlag: bool = False,
              planFlag: bool = False) -> dict:

        '''
        Return per-atom or coordinate payloads from any combination of
        typing, topology, and trajectory domains.

        This is the general-purpose extraction function. ``positions()`` is a
        convenience wrapper around the trajectory positions case.

            fetch()     — any non-property request across all three domains
            select()    — scalar property-* requests only
            positions() — dedicated wrapper for trajectory coordinate extraction

        Cross-domain intersection
        -------------------------
        When queries are provided across multiple domains, fetch() resolves
        the intersection of their global_id selections and applies it as a
        post-filter to every payload uniformly. All payloads returned in a
        single call are guaranteed to correspond to the same set of atoms.

        Typing and topology selections are resolved statically from their
        predicate queries. Trajectory selection is evaluated per-frame via
        ``_resolve_trajectory_selection_{fmt}`` — for DCD this is a hotpath
        returning [None, None, ...] (no constraint). Future trajectory formats
        that encode per-atom properties (e.g. per-frame charges) can return
        per-frame id lists to participate in the intersection.

        Returns:
        -------
        dict
            Standardised five-key envelope:
                mode      = "fetch"
                selection = per-domain selection block (intersection mode)
                metadata  = per-domain file metadata for every loaded file
                plan      = per-domain plan for every queried domain, plus
                            a "combined" sub-dict with the cross-domain
                            n_atoms_upper_bound and total_estimated_bytes
                payload   = {domain: <bare value | ndarray>, ...} — array
                            payload requests (positions/velocities) yield
                            ndarrays of shape (n_frames, n_atoms_selected, 3)
        '''

        TYPE_Q = self._normalise_query(TYPE_Q)
        TOPO_Q = self._normalise_query(TOPO_Q)
        TRAJ_Q = self._normalise_query(TRAJ_Q)

        type_q_provided = bool(TYPE_Q)
        topo_q_provided = bool(TOPO_Q)
        traj_q_provided = bool(TRAJ_Q)

        TYPE_R = self._normalise_request(TYPE_R)
        TOPO_R = self._normalise_request(TOPO_R)
        TRAJ_R = self._normalise_request(TRAJ_R)

        if TYPE_R is None and TOPO_R is None and TRAJ_R is None:
            raise ValueError("fetch() requires at least one request string (TYPE_R, TOPO_R, or TRAJ_R).")

        for label, r in (("TYPE_R", TYPE_R), ("TOPO_R", TOPO_R), ("TRAJ_R", TRAJ_R)):
            if r is not None:
                domain = {"TYPE_R": "typing", "TOPO_R": "topology", "TRAJ_R": "trajectory"}[label]
                if self._classify_request(domain, r) == "property":
                    raise ValueError(
                        f"{label}='{r}' is a property request. "
                        f"Use select() for property-* requests."
                    )

        if devFlag: # dev hotpath

            # ------------------------------------------------------------------ #
            # Build plans + metadata up-front (zero file I/O for plans beyond    #
            # what the parser planner does).                                      #
            # ------------------------------------------------------------------ #

            plan: dict = {}
            for label, q, r in (("typing", TYPE_Q, TYPE_R),
                                ("topology", TOPO_Q, TOPO_R),
                                ("trajectory", TRAJ_Q, TRAJ_R)):
                if r is not None:
                    result = self._plan_domain_request(
                        domain=label, query_dictionary=q, request_string=r,
                    )
                    if result is not None:
                        plan[label] = result

            # Inject the cross-domain combined estimate. n_atoms_upper_bound
            # = min across per-domain plans (the true intersection cannot
            # exceed any contributor); total_estimated_bytes uses that bound
            # times each domain's n_frames * bytes_per_atom_per_frame.
            combined = self._build_combined_plan_estimate(plan)
            if combined is not None:
                plan["combined"] = combined

            metadata = self._build_metadata_for_loaded_domains()

            if planFlag:
                selection = self._build_selection_block(
                    merge_mode="intersection",
                    type_q_provided=type_q_provided,
                    topo_q_provided=topo_q_provided,
                    traj_q_provided=traj_q_provided,
                    type_ids=None, topo_ids=None, traj_ids=None,
                    resolved_count=None,
                )
                return self._build_envelope(
                    mode="fetch", selection=selection,
                    metadata=metadata, plan=plan, payload=None,
                )

        # ------------------------------------------------------------------ #
        # Resolve global_ids for intersection. Only domains with a query     #
        # constraint contribute an id set.                                    #
        # ------------------------------------------------------------------ #

        type_ids = None
        topo_ids = None
        traj_ids = None

        if type_q_provided:
            self._ensure_domain_loaded("typing")
            type_ids = self.get_types(
                QUERY=TYPE_Q, REQUEST="global_ids", updateFlag=updateFlag,
            )

        if topo_q_provided:
            self._ensure_domain_loaded("topology")
            topo_ids = self.get_topology(
                QUERY=TOPO_Q, REQUEST="global_ids", updateFlag=updateFlag,
            )

        if traj_q_provided:
            self._ensure_domain_loaded("trajectory")
            traj_selection = self.get_trajectory(
                QUERY=TRAJ_Q, REQUEST="global_ids", updateFlag=updateFlag,
            )
            # Hotpath: trajectory returns None or [None, None, ...] -> no constraint.
            # When the trajectory format actually carries per-atom data and
            # contributes ids, traj_selection is a list of per-frame id lists.
            if traj_selection is None:
                pass
            elif isinstance(traj_selection, list):
                if not all(f is None for f in traj_selection):
                    traj_ids_set: set[int] = set()
                    for frame_ids in traj_selection:
                        if frame_ids is not None:
                            traj_ids_set.update(frame_ids)
                    traj_ids = sorted(traj_ids_set)

        # None means "no constraint from this domain" — distinct from empty
        # set, which means "constrained to zero atoms". Only sets that are
        # not None participate in the intersection.
        active = [set(ids) for ids in (type_ids, topo_ids, traj_ids) if ids is not None]

        if len(active) > 1:
            selected_globals = sorted(set.intersection(*active))
        elif len(active) == 1:
            selected_globals = sorted(active[0])
        else:
            selected_globals = None

        if selected_globals is not None and not selected_globals:
            raise ValueError("Cross-domain selection produced an empty intersection.")

        # ------------------------------------------------------------------ #
        # Ensure global_ids are cached for any domain that has a request and  #
        # will need post-filtering against a cross-domain intersection.       #
        # ------------------------------------------------------------------ #

        cross_domain = selected_globals is not None and len(active) > 1

        if cross_domain:
            if TYPE_R is not None and type_ids is None:
                type_ids = self.get_types(
                    QUERY=TYPE_Q, REQUEST="global_ids", updateFlag=updateFlag,
                )
            if TOPO_R is not None and topo_ids is None:
                topo_ids = self.get_topology(
                    QUERY=TOPO_Q, REQUEST="global_ids", updateFlag=updateFlag,
                )

        # ------------------------------------------------------------------ #
        # Execute and post-filter. Per-atom array payloads (positions,       #
        # velocities) come back as bare ndarrays; everything else as lists   #
        # or scalars. Cross-domain intersection mask is applied uniformly.   #
        # ------------------------------------------------------------------ #

        all_globals = list(range(self.global_system_properties['num_atoms']))
        gid_set = set(selected_globals) if selected_globals is not None else None
        payload: dict = {}

        if TYPE_R is not None:
            raw = self.get_types(QUERY=TYPE_Q, REQUEST=TYPE_R, updateFlag=updateFlag)
            if isinstance(raw, np.ndarray):
                if gid_set is not None and type_ids is not None:
                    mask = [i for i, g in enumerate(type_ids) if g in gid_set]
                    raw  = raw[:, mask, :]
                payload["typing"] = raw
            else:
                if gid_set is not None and isinstance(raw, list):
                    ids = type_ids if type_ids is not None else all_globals
                    raw = [v for v, g in zip(raw, ids) if g in gid_set]
                payload["typing"] = raw

        if TOPO_R is not None:
            raw = self.get_topology(QUERY=TOPO_Q, REQUEST=TOPO_R, updateFlag=updateFlag)
            if isinstance(raw, np.ndarray):
                if gid_set is not None and topo_ids is not None:
                    mask = [i for i, g in enumerate(topo_ids) if g in gid_set]
                    raw  = raw[:, mask, :]
                payload["topology"] = raw
            else:
                if gid_set is not None and isinstance(raw, list):
                    ids = topo_ids if topo_ids is not None else all_globals
                    raw = [v for v, g in zip(raw, ids) if g in gid_set]
                payload["topology"] = raw

        if TRAJ_R is not None:
            traj_query = dict(TRAJ_Q)
            if selected_globals is not None:
                traj_query["global_ids"] = (selected_globals, set())
            raw = self.get_trajectory(QUERY=traj_query, REQUEST=TRAJ_R, updateFlag=updateFlag)
            payload["trajectory"] = raw

        if devFlag: # dev hotpath

            selection = self._build_selection_block(
                merge_mode="intersection",
                type_q_provided=type_q_provided,
                topo_q_provided=topo_q_provided,
                traj_q_provided=traj_q_provided,
                type_ids=type_ids, topo_ids=topo_ids, traj_ids=traj_ids,
                resolved_count=(len(selected_globals) if selected_globals is not None else None),
            )

            return self._build_envelope(
                mode="fetch", selection=selection,
                metadata=metadata, plan=plan, payload=payload,
            )
        
        else: 

            return payload


    # -------------------------------------------------------------------------
    # QUERY PROCESSING SWITCHING 
    # -------------------------------------------------------------------------

    def get_types(self, 
                  QUERY: dict = None,
                  REQUEST: str = None,
                  updateFlag: bool = False):
        
        '''
        A switchboard function that takes in a query dictionary and returns the types of the atoms that match the query. 
        The query dictionary should be of the form:
        
        Parameters:
        ----------
        QUERY: dict
            A dictionary based query.
        REQUEST: str
            A string specifying the type of information wanted.


        Returns:
        -------
        list [int]
            A list of global indicies that match the query.
        '''

        REQUEST = self._normalise_request(REQUEST)
        QUERY = self._normalise_query(QUERY)

        self._ensure_domain_loaded("typing")
        self._validate_query(query_dict=QUERY, query_type="typing")
        self._validate_request(request_string=REQUEST, query_type="typing")

        return self._execute_domain_request(
            domain="typing",
            query_dictionary=QUERY,
            request_string=REQUEST,
            updateFlag=updateFlag,
        )
    
    def get_topology(self, 
                     QUERY: dict = None,
                     REQUEST: str = None,
                     updateFlag : bool = False):
        
        '''
        A switchboard function that takes in a query dictionary and returns the topology information of the atoms that match the query.

        Parameters:
        ----------
        QUERY: dict
            A dictionary based query.
        REQUEST: str
            A string specifying the type of topology information requested, e.g. "bonds", "angles", "dihedrals", etc.

        Returns:
        -------
        list [int]
            A list of global indicies that match the query.
        '''
        
        REQUEST = self._normalise_request(REQUEST)
        QUERY = self._normalise_query(QUERY)

        self._ensure_domain_loaded("topology")
        self._validate_query(query_dict=QUERY, query_type="topology")
        self._validate_request(request_string=REQUEST, query_type="topology")

        return self._execute_domain_request(
            domain="topology",
            query_dictionary=QUERY,
            request_string=REQUEST,
            updateFlag=updateFlag,
        )

    def get_trajectory(self, 
                         QUERY: dict = None,
                         REQUEST: str = None,
                         updateFlag : bool = False):  
        
        '''
        A switchboard function that takes in a query dictionary and returns the trajectory information of the atoms that match the query.

        Parameters:
        ----------
        QUERY: dict
            A dictionary based query.
        REQUEST: str
            A string specifying the type of trajectory information requested, e.g. "typings", "velocities", "forces", etc.

        Returns:
        -------
        list [int]
            A list of global indicies that match the query.
        '''

        REQUEST = self._normalise_request(REQUEST)
        QUERY = self._normalise_query(QUERY)

        self._ensure_domain_loaded("trajectory")
        self._validate_query(query_dict=QUERY, query_type="trajectory")
        self._validate_request(request_string=REQUEST, query_type="trajectory")

        return self._execute_domain_request(
            domain="trajectory",
            query_dictionary=QUERY,
            request_string=REQUEST,
            updateFlag=updateFlag,
        )


    # -------------------------------------------------------------------------
    # RESULT ENVELOPE STANDARDISATION
    # -------------------------------------------------------------------------
    
    # Every user-facing call (positions, fetch, select) returns the same
    # five-key envelope:
    #
    #   {
    #     "mode":      "positions" | "fetch" | "select",
    #     "selection": { ... per-domain query/ids info, identical schema ... },
    #     "metadata":  { domain: { ... per-domain file facts ... }, ... },
    #     "plan":      { domain: { ... per-domain query plan ... }, ... },
    #     "payload":   <ndarray>                              # positions()
    #                | { domain: <bare value>, ... }          # fetch() / select()
    #   }
    #
    # `metadata` is query-independent — it reflects every loaded file.
    # `plan` is query-dependent — it only includes domains touched by the call.
    # `selection` always carries blocks for all three domains so the schema
    # the user sees is identical across modes.
    #
    # Within metadata and plan, every per-domain dict has three tiers:
    #   tier 1 — required keys, hard-fail if a parser omits them
    #   tier 2 — optional, omitted entirely (never None) when not meaningful
    #   tier 3 — `format_specific: {...}` for genuinely format-unique extras
    # -------------------------------------------------------------------------

    # Tier-1 metadata keys per domain.
    _METADATA_TIER_1_BY_DOMAIN = {
        "typing":     ("source", "file_path", "file_format", "file_size_bytes", "n_atoms"),
        "topology":   ("source", "file_path", "file_format", "file_size_bytes", "n_atoms"),
        "trajectory": ("source", "file_path", "file_format", "file_size_bytes", "n_atoms", "n_frames"),
    }

    # Tier-2 metadata keys the standardiser knows how to surface.
    # Anything in raw_meta with one of these names (or its alias) is promoted
    # to top-level when present and non-None.
    _METADATA_TIER_2_KEYS = frozenset({
        "n_residues", "n_segments", "box_size", "ensemble_type",
        "simulation_type", "timestep", "sim_name", "units",
    })

    # Aliases mapping legacy/global_system_properties names -> canonical names.
    _METADATA_KEY_ALIASES = {
        "num_atoms":      "n_atoms",
        "num_frames":     "n_frames",
        "num_residues":   "n_residues",
        "num_segments":   "n_segments",
        "start_box_size": "box_size",
    }

    # Tier-1 plan keys.
    #
    # n_atoms semantics depend on planner_mode:
    #   "header"     → file-wide atom count (exact, but does NOT reflect
    #                  cross-domain selection; the selection.resolved_count
    #                  is the post-intersection truth)
    #   "stochastic" → estimated atoms matching the domain's query predicate
    #
    # n_frames follows the same split: header gives file total (or
    # frame_interval-reduced total), stochastic always reports 1 for
    # static files.
    #
    # bytes_per_atom_per_frame is sourced from the parser's plan_shape
    # function (NOT raw_plan). estimated_bytes is computed by the
    # standardiser as n_atoms * n_frames * bytes_per_atom_per_frame.
    _PLAN_TIER_1_KEYS = ("planner_mode", "source", "file_format", "request",
                         "n_atoms", "n_frames", "bytes_per_atom_per_frame",
                         "estimated_bytes")

    # Plan keys the standardiser absorbs into the `sampling` sub-block when
    # present (stochastic mode only).
    _PLAN_SAMPLING_KEYS = frozenset({
        "n_lines_sampled", "n_lines_eligible", "n_lines_matching",
        "n_atoms_sampled", "n_atoms_matched",
        "rng_seed", "target_sample_size", "sample_probability",
    })

    # Plan keys the standardiser intentionally drops from raw_plan (legacy
    # noise or values now sourced authoritatively elsewhere).
    _PLAN_DROP_KEYS = frozenset({
        "file_type",                # superseded by file_format
        "estimated_mib",            # standardiser only exposes estimated_bytes
        "bytes_per_atom_per_frame", # standardiser sources from plan_shape
        "query_dictionary",         # the user passed it in; no need to echo it back
    })


    def _build_selection_block(self,
                               *,
                               merge_mode: str,
                               type_q_provided: bool,
                               topo_q_provided: bool,
                               traj_q_provided: bool,
                               type_ids,
                               topo_ids,
                               traj_ids,
                               resolved_count) -> dict:
        '''
        Build the selection sub-envelope. Schema is identical for every mode;
        select() simply passes empty contents with merge_mode="none".

        Parameters
        ----------
        merge_mode : str
            "intersection" for positions()/fetch(), "none" for select().
        type_q_provided, topo_q_provided, traj_q_provided : bool
            Whether the user passed a query dict for that domain.
        type_ids, topo_ids, traj_ids : list[int] | None
            The id sets each domain contributed to the intersection. None
            means the domain did not contribute (e.g. DCD hotpath returns
            no constraint even when a query is provided).
        resolved_count : int | None
            Size of the final intersected selection. None for select() and
            for positions()/fetch() calls with no domain constraints.

        Returns
        -------
        dict
            Selection block in the standardised schema.
        '''

        def _domain_block(query_provided: bool, ids):
            return {
                "query_provided": bool(query_provided),
                "ids_provided":   ids is not None,
                "n_matched":      (len(ids) if ids is not None else None),
            }

        return {
            "merge_mode":     merge_mode,
            "typing":         _domain_block(type_q_provided, type_ids),
            "topology":       _domain_block(topo_q_provided, topo_ids),
            "trajectory":     _domain_block(traj_q_provided, traj_ids),
            "resolved_count": resolved_count,
        }


    def _standardise_metadata(self,
                              domain: str,
                              file_path,
                              file_format: str,
                              raw_meta: dict) -> dict:
        '''
        Build the standardised metadata dict for a single loaded domain.

        Parameters
        ----------
        domain : str
            One of "typing", "topology", "trajectory".
        file_path : str | Path
            Path to the file backing this domain.
        file_format : str
            File extension without the leading dot (e.g. "pdb", "dcd").
        raw_meta : dict
            Raw metadata dict from the parser's _update_*_globals_* function.
            May be empty.

        Returns
        -------
        dict
            Standardised metadata envelope. Tier-1 keys are guaranteed
            present; tier-2 keys are present only when meaningful;
            format-unique keys land in `format_specific` if any remain.

        Raises
        ------
        ValueError
            If a tier-1 key required by the domain cannot be resolved from
            raw_meta.
        '''

        # Canonicalise alias keys up-front so downstream lookups are uniform.
        canonical_meta: dict = {}
        for k, v in (raw_meta or {}).items():
            canonical_key = self._METADATA_KEY_ALIASES.get(k, k)
            # If both alias and canonical present, canonical wins.
            if canonical_key not in canonical_meta or canonical_meta[canonical_key] is None:
                canonical_meta[canonical_key] = v

        file_path = Path(file_path)
        try:
            file_size_bytes = file_path.stat().st_size
        except OSError:
            file_size_bytes = None

        # -------------------- tier 1 --------------------
        out: dict = {
            "source":          domain,
            "file_path":       str(file_path),
            "file_format":     file_format,
            "file_size_bytes": file_size_bytes,
        }

        n_atoms = canonical_meta.get("n_atoms")
        if n_atoms is None:
            raise ValueError(
                f"Metadata contract violated: parser '{file_format}' did not report "
                f"'n_atoms' (or alias 'num_atoms') for domain '{domain}'. "
                f"Tier-1 metadata keys must be reported by every parser."
            )
        out["n_atoms"] = int(n_atoms)

        if domain == "trajectory":
            n_frames = canonical_meta.get("n_frames")
            if n_frames is None:
                raise ValueError(
                    f"Metadata contract violated: parser '{file_format}' did not "
                    f"report 'n_frames' (or alias 'num_frames') for trajectory domain. "
                    f"Tier-1 metadata keys must be reported by every parser."
                )
            out["n_frames"] = int(n_frames)

        # -------------------- tier 2 --------------------
        consumed = {"n_atoms", "n_frames"}
        for key in self._METADATA_TIER_2_KEYS:
            if key in canonical_meta and canonical_meta[key] is not None:
                out[key] = canonical_meta[key]
                consumed.add(key)

        # -------------------- tier 3 (format_specific) --------------------
        format_specific = {
            k: v for k, v in canonical_meta.items()
            if k not in consumed and v is not None
        }
        if format_specific:
            out["format_specific"] = format_specific

        return out


    def _build_metadata_for_loaded_domains(self) -> dict:
        '''
        Build the metadata sub-envelope covering every currently loaded domain.

        Returns
        -------
        dict
            ``{domain: standardised_metadata_dict, ...}``. Domains without a
            loaded file are omitted entirely (never present with None).
        '''

        metadata: dict = {}
        for domain, cfg in self._domain_registry.items():
            filepath = getattr(self, cfg["file_attr"])
            if filepath is None:
                continue

            filetype = getattr(self, cfg["type_attr"])
            fmt = filetype[1:]
            module = self._get_parse_module(fmt)

            update_fn = getattr(module, cfg["update_fn_template"].format(fmt=fmt), None)
            raw_meta = update_fn(filepath) if update_fn is not None else {}
            if not isinstance(raw_meta, dict):
                raw_meta = {}

            metadata[domain] = self._standardise_metadata(
                domain=domain,
                file_path=filepath,
                file_format=fmt,
                raw_meta=raw_meta,
            )

        return metadata


    def _standardise_plan(self,
                          domain: str,
                          file_format: str,
                          request: str,
                          raw_plan: dict,
                          plan_shape: tuple) -> dict:
        
        '''
        Build the standardised plan dict for a single domain query.

        Parameters
        ----------
        domain : str
            One of "typing", "topology", "trajectory".
        file_format : str
            File extension without the leading dot.
        request : str
            The request string the planner was asked about.
        raw_plan : dict
            Raw planner output from the parser's _plan_*_query_* function.
            Must include planner_mode, n_atoms, n_frames.
        plan_shape : tuple
            Output of the parser's _get_*_plan_shape_* function — a
            3-tuple ``(output_kind, trailing_shape, bytes_per_match)``.
            ``bytes_per_match`` is the authoritative source for
            bytes_per_atom_per_frame; raw_plan values are ignored.

        Returns
        -------
        dict
            Standardised plan envelope with computed estimated_bytes.

        Raises
        ------
        ValueError
            If a tier-1 plan key is missing from raw_plan (excluding the
            `supported: False` short-circuit case where only planner_mode and
            reason are required), or if plan_shape reports no
            bytes_per_match for a non-property request.
        '''

        if not isinstance(raw_plan, dict):
            raise ValueError(
                f"Plan contract violated: parser '{file_format}' planner for "
                f"domain '{domain}' request '{request}' returned a "
                f"{type(raw_plan).__name__}, expected dict."
            )

        planner_mode = raw_plan.get("planner_mode")
        if planner_mode is None:
            raise ValueError(
                f"Plan contract violated: parser '{file_format}' planner did not "
                f"report 'planner_mode' for domain '{domain}' request '{request}'."
            )

        # ---------------- unsupported short-circuit ----------------
        if raw_plan.get("supported") is False:
            base = {
                "planner_mode": planner_mode,
                "source":       domain,
                "file_format":  file_format,
                "request":      request,
                "supported":    False,
                "reason":       raw_plan.get("reason", "Planner does not support this request."),
            }
            consumed = {"planner_mode", "request", "source", "file_format",
                        "supported", "reason"} | self._PLAN_DROP_KEYS
            extras = {k: v for k, v in raw_plan.items() if k not in consumed and v is not None}
            if extras:
                base["format_specific"] = extras
            return base

        # ---------------- tier 1 (supported branch) ----------------
        n_atoms = raw_plan.get("n_atoms")
        if n_atoms is None:
            raise ValueError(
                f"Plan contract violated: parser '{file_format}' planner did not "
                f"report 'n_atoms' for domain '{domain}' request '{request}'. "
                f"All planners must report estimated/exact n_atoms at tier 1."
            )

        n_frames = raw_plan.get("n_frames")
        if n_frames is None:
            raise ValueError(
                f"Plan contract violated: parser '{file_format}' planner did not "
                f"report 'n_frames' for domain '{domain}' request '{request}'. "
                f"Static (non-frame) data must report n_frames=1."
            )

        # bytes_per_atom_per_frame is sourced from plan_shape, not raw_plan.
        # Any raw_plan value for this key is silently dropped (see _PLAN_DROP_KEYS).
        bytes_per = plan_shape[2]
        if bytes_per is None:
            raise ValueError(
                f"plan_shape contract violated: parser '{file_format}' "
                f"_get_{domain}_plan_shape_{file_format} reported "
                f"bytes_per_match=None for non-property request "
                f"'{request}'. None is reserved for scalar_property "
                f"requests, which should short-circuit before this point."
            )

        n_atoms_int = int(n_atoms)
        n_frames_int = int(n_frames)
        bytes_per_int = int(bytes_per)
        estimated_bytes = n_atoms_int * n_frames_int * bytes_per_int

        base: dict = {
            "planner_mode":             planner_mode,
            "source":                   domain,
            "file_format":              file_format,
            "request":                  request,
            "n_atoms":                  n_atoms_int,
            "n_frames":                 n_frames_int,
            "bytes_per_atom_per_frame": bytes_per_int,
            "estimated_bytes":          estimated_bytes,
        }

        # ---------------- tier 2 ----------------
        if planner_mode == "stochastic":
            if "confidence" in raw_plan and raw_plan["confidence"] is not None:
                base["confidence"] = raw_plan["confidence"]

            sampling = {
                k: raw_plan[k] for k in self._PLAN_SAMPLING_KEYS
                if k in raw_plan and raw_plan[k] is not None
            }
            if sampling:
                base["sampling"] = sampling

        # ---------------- tier 3 (format_specific) ----------------
        consumed = (
            set(self._PLAN_TIER_1_KEYS)
            | {"confidence", "supported", "reason"}
            | self._PLAN_SAMPLING_KEYS
            | self._PLAN_DROP_KEYS
        )
        extras = {k: v for k, v in raw_plan.items() if k not in consumed and v is not None}
        if extras:
            base["format_specific"] = extras

        return base


    def _build_combined_plan_estimate(self, plan: dict) -> dict | None:
        '''
        Compute the cross-domain rollup estimate from per-domain plans.

        Returns a sub-dict for ``plan["combined"]`` with two fields:

        - ``n_atoms_upper_bound``: minimum n_atoms across all per-domain
          plans. The true intersection size is bounded above by this.
        - ``total_estimated_bytes``: assuming the upper bound holds, the
          summed payload size across every requested domain
          (``n_atoms_upper_bound * n_frames_d * bytes_per_atom_per_frame_d``
          summed over domains).

        Parameters
        ----------
        plan : dict
            Per-domain plan dict, where keys are domain labels ("typing",
            "topology", "trajectory") and values are standardised
            per-domain plan dicts. May be empty.

        Returns
        -------
        dict | None
            The combined estimate, or None if `plan` is empty (no per-atom
            requests issued — e.g. select() with property-only requests).
        '''

        if not plan:
            return None

        # Filter to plans with the supported tier-1 keys (skip any
        # supported:False short-circuited entries that may have crept in).
        valid = [p for p in plan.values()
                 if isinstance(p, dict) and "n_atoms" in p and "n_frames" in p
                 and "bytes_per_atom_per_frame" in p]

        if not valid:
            return None

        n_atoms_upper_bound = min(p["n_atoms"] for p in valid)

        total_estimated_bytes = sum(
            n_atoms_upper_bound * p["n_frames"] * p["bytes_per_atom_per_frame"]
            for p in valid
        )

        return {
            "n_atoms_upper_bound":   n_atoms_upper_bound,
            "total_estimated_bytes": int(total_estimated_bytes),
        }


    def _build_envelope(self,
                        *,
                        mode: str,
                        selection: dict,
                        metadata: dict,
                        plan: dict,
                        payload) -> dict:
        '''
        Assemble the five-key result envelope. Pure structural helper.
        '''

        return {
            "mode":      mode,
            "selection": selection,
            "metadata":  metadata,
            "plan":      plan,
            "payload":   payload,
        }


    # -------------------------------------------------------------------------
    # INTERNAL DOMAIN EXECUTION
    # -------------------------------------------------------------------------

    def _plan_domain_request(self,
                             domain: str,
                             query_dictionary: dict,
                             request_string: str, ):

        '''
        A function for creating a query plan for a specified domain.

        Calls the parser's plan_shape function first; for scalar property
        requests this short-circuits to None (no per-atom payload size to
        estimate). For per-atom requests, calls the planner and routes the
        result through the standardiser, which uses plan_shape's
        bytes_per_match to compute estimated_bytes.

        Parameters:
        ----------
        domain: str
            The domain on which to execute the query, one of "typing", "topology", or "trajectory".
        query_dictionary: dict
            A dictionary containing the query parameters for the specified domain.
        request_string: str
            A string specifying the type of information requested for the specified domain.

        Returns:
        -------
        dict | None
            The standardised query plan for the specified domain, or None
            if the request is a scalar property (no per-atom payload size).
        '''

        cfg = self._domain_registry[domain]
        filepath = getattr(self, cfg["file_attr"])
        filetype = getattr(self, cfg["type_attr"])
        keywords_available = getattr(self, cfg["keys_attr"])
        requests_available = getattr(self, cfg["reqs_attr"])

        fmt = filetype[1:]
        module = self._get_parse_module(fmt)

        # Plan shape first — anything without a per-atom-per-frame byte cost
        # (scalar properties, selectors) has no payload size to estimate, so
        # we skip the planner entirely and return None. The standardiser is
        # only invoked for requests that yield a sized per-atom payload.
        plan_shape_fn = getattr(module, cfg["plan_shape_fn_template"].format(fmt=fmt))
        plan_shape = plan_shape_fn(request_string)

        if plan_shape[2] is None:
            return None

        plan_fn = getattr(module, cfg["plan_fn_template"].format(fmt=fmt))
        raw_plan = plan_fn(
            filepath,
            query_dictionary=query_dictionary,
            request_string=request_string,
            keywords_available=keywords_available,
            requests_available=requests_available,
        )

        # Route through the standardiser. Hard-fails if the parser violates
        # the tier-1 plan contract. Returns the supported:False short-circuit
        # envelope when the planner explicitly opts out.
        standardised = self._standardise_plan(
            domain=domain,
            file_format=fmt,
            request=request_string,
            raw_plan=raw_plan,
            plan_shape=plan_shape,
        )
        if standardised is not None and standardised.get("supported") is False:
            return None
        return standardised

    def _execute_domain_request(self,
                                domain: str,
                                query_dictionary: dict,
                                request_string: str,
                                updateFlag: bool = False):
        
        '''
        Internal function to execute a query on a specific domain (typing, topology, trajectory) 
        based on the provided query dictionary and request string.

        Return shape contract
        ---------------------
        The parser returns its native bare value for the request — a list,
        scalar, tuple, ndarray, or None. No wrapping, no metadata tuples.
        Per-atom array payloads (positions, velocities) are returned as
        ndarrays of shape ``(n_frames, n_atoms_selected, 3)``.

        File-level metadata (frame counts, endian, etc.) is built
        independently via ``_build_metadata_for_loaded_domains`` and is not
        threaded through this layer.

        Parameters:
        ----------
        domain: str
            The domain on which to execute the query, one of "typing", "topology", or "trajectory".
        query_dictionary: dict
            A dictionary containing the query parameters for the specified domain.
        request_string: str
            A string specifying the type of information requested for the specified domain.
        updateFlag: bool, default=False
            A flag indicating whether to update the global system properties based on the information extracted from the files after executing the query.

        Returns:
        -------
        The parser's native return value (list, scalar, tuple, ndarray, or None).
        '''

        cfg = self._domain_registry[domain]
        filepath = getattr(self, cfg["file_attr"])
        filetype = getattr(self, cfg["type_attr"])
        keywords_available = getattr(self, cfg["keys_attr"])
        requests_available = getattr(self, cfg["reqs_attr"])

        fmt = filetype[1:]
        module = self._get_parse_module(fmt)

        query_fn = getattr(module, cfg["query_fn_template"].format(fmt=fmt))
        output = query_fn(
            filepath,
            query_dictionary=query_dictionary,
            request_string=request_string,
            keywords_available=keywords_available,
            requests_available=requests_available,
        )

        if updateFlag:
            update_fn = getattr(module, cfg["update_fn_template"].format(fmt=fmt))
            updated_globals = update_fn(filepath)
            self._match_system_properties(updated_globals)

        return output

    def _load_domain_file(self, domain: str, filepath: str | Path) -> None:

        '''
        Internal function to load a file for a specific domain (typing, topology, trajectory) 
        and extract its available keywords and requests.

        Parameters:
        ----------
        domain: str
            The domain for which to load the file, one of "typing", "topology", or "trajectory".
        filepath: str | Path
            The file path to the file to be loaded for the specified domain.
        
        Returns:
        -------
        None
        '''

        cfg = self._domain_registry[domain]

        filepath = Path(filepath)
        self._validate_filepath(filepath)

        filetype = self._get_filetype(filepath)
        if filetype not in cfg["supported_formats"]:
            raise ValueError(
                f"Unsupported {cfg['label']} file format: {filetype}. "
                f"Supported: {sorted(cfg['supported_formats'])}"
            )

        setattr(self, cfg["file_attr"], filepath)
        setattr(self, cfg["type_attr"], filetype)

        fmt = filetype[1:]
        module = self._get_parse_module(fmt)

        keys_fn = getattr(module, cfg["keys_fn_template"].format(fmt=fmt))
        keys, reqs = keys_fn(filepath)

        setattr(self, cfg["keys_attr"], keys)
        setattr(self, cfg["reqs_attr"], reqs)

        # Auto-populate globals from newly loaded file
        update_fn = getattr(module, cfg["update_fn_template"].format(fmt=fmt))
        updated_globals = update_fn(filepath)
        self._match_system_properties(updated_globals)

        # Cross-validate atom count against already-loaded domains
        self._validate_atom_count_consistency(domain)

    def _get_parse_module(self, fmt: str):

        '''
        Internal function to dynamically import and cache the parsing module for a given file format.

        Parameters:
        ----------
        fmt: str
            The file format for which to import the parsing module, e.g. "pdb", "psf", "dcd".
        
        Returns:
        -------
        module
            The imported parsing module corresponding to the specified file format.
        '''

        if fmt not in self._module_cache:
            module = importlib.import_module(f"trajectory_kit.{fmt}_parse")
            domain = self._get_domain_from_fmt(fmt)
            self._validate_domain_module_contract(domain, fmt, module)
            self._module_cache[fmt] = module
        return self._module_cache[fmt]

    def _ensure_domain_loaded(self, domain: str) -> None:
        
        '''
        Internal function to ensure that a file for the specified domain is loaded before executing a query.
        
        Parameters:
        ----------
        domain: str
            The domain for which to check if a file is loaded, one of "typing", "topology", or "trajectory".
        
        Returns:
        -------
        None

        Raises:
        ------
        ValueError
            If no file is currently loaded for the specified domain.
        '''

        cfg = self._domain_registry[domain]
        filepath = getattr(self, cfg["file_attr"])
        if filepath is None:
            raise ValueError(f"No {cfg['label']} file is currently loaded.")


    # -------------------------------------------------------------------------
    # REQUEST CLASSIFICATION
    # -------------------------------------------------------------------------

    def _classify_request(self, domain: str, request: str) -> str:

        '''
        Classify the request into one of the following categories:
        - selector: requests that return global ids of atoms matching the query
        - property: requests that return a global property of the system, e.g. box size, number of atoms, etc.
        - payload: requests that return atom-specific information, e.g. positions, atom types, etc.

        Parameters:
        ----------
        domain: str
            The domain for which to classify the request, one of "typing", "topology", or "trajectory".
        request: str
            The request string to classify.
        
        Returns:
        -------
        str
            The classification of the request, one of "selector", "property", or "payload".

        Raises:
        ------
        ValueError
            If the request string does not match any known request type.
        '''

        request = self._normalise_request(request)

        if request.startswith("property-"):
            return "property"

        reqs = getattr(self, self._domain_registry[domain]["reqs_attr"])
        if request in reqs:
            return "payload"

        raise ValueError(f"Unknown {domain} request class for '{request}'.")


    # -------------------------------------------------------------------------
    # FILE LOADING 
    # -------------------------------------------------------------------------

    def load_typing(self, typing_filepath: str | Path):

        '''
        Function to load a typing file into the sim object.

        Parameters:
        ----------
        typing_filepath: str | Path
            The file path to the typing file. This is required and should be a .pdb

        Returns:
        -------
        bool
            True if the typing file is successfully loaded, otherwise False.
        '''

        self._load_domain_file("typing", typing_filepath)
        return True

    def load_topology(self, topology_filepath: str | Path):

        '''
        Function to load a topology file into the sim object.

        Parameters:
        ----------
        topology_filepath: str | Path
            The file path to the topology file. This is required and should be a .pdb

        Returns:
        -------
        bool
            True if the topology file is successfully loaded, otherwise False.
        '''

        self._load_domain_file("topology", topology_filepath)
        return True

    def load_trajectory(self, trajectory_filepath: str | Path):

        '''
        Function to load a `trajectory` file into the sim object.

        Parameters:
        ----------
        trajectory_filepath: str | Path
            The file path to the trajectory file. This is required and should be a .pdb

        Returns:
        -------
        bool
            True if the trajectory file is successfully loaded, otherwise False.
        '''

        self._load_domain_file("trajectory", trajectory_filepath)
        return True


    # -------------------------------------------------------------------------
    # RESOLVE AND NORMALISE REQUESTS
    # -------------------------------------------------------------------------

    def _normalise_request(self, request_string: str | None):#
        
        '''
        Function to normalise the request string 
        by stripping leading and trailing whitespace and validating that it is a non-empty string.
        
        Parameters:
        ----------
        request_string: str | None
            The request string to normalise. If None, None is returned.
        
        Returns:
        -------
        str | None
            The normalised request string if it is valid, otherwise raises an exception.
        
        Raises:
        ------
        TypeError
            If the input request_string is not a string or None.
        ValueError
            If the input request_string is empty or contains only whitespace.
        '''

        if request_string is None:
            return None
        if not isinstance(request_string, str):
            raise TypeError(
                f"Request must be a string. Received type: {type(request_string).__name__}"
            )

        request_string = request_string.strip()
        if request_string == "":
            raise ValueError("Request string cannot be empty or whitespace only.")
        return request_string

    def _normalise_query(self, query_dict: dict | None):

        '''
        Function to normalise the query dictionary by validating its type and returning a copy.

        Parameters:
        ----------
        query_dict: dict | None
            The query dictionary to normalise. If None, an empty dictionary is returned.
        
        Returns:
        -------
        dict
            A copy of the input query dictionary if it is valid, otherwise raises an exception.

        Raises:
        ------
        TypeError
            If the input query_dict is not a dictionary or None.
        '''

        if query_dict is None:
            return {}
        if not isinstance(query_dict, dict):
            raise TypeError(
                f"Query must be a dictionary. Received type: {type(query_dict).__name__}"
            )
        return dict(query_dict)


    # -------------------------------------------------------------------------
    # MISC SUPPORT FUNCTIONALITY
    # -------------------------------------------------------------------------

    def _get_domain_from_fmt(self, fmt: str) -> str:
        
        '''
        Helper function to get the domain associated with a given file format.

        Parameters:
        ----------
        fmt: str
            The file format for which to get the associated domain, e.g. "pdb", "psf", "dcd".

        Returns:
        -------
        out: str
            The domain associated with the specified file format, one of "typing", "topology", or "trajectory".

        Raises:
        ------
        ValueError
            If the specified file format is not associated with any known domain.
        '''

        for domain, cfg in self._domain_registry.items():
            if f".{fmt}" in cfg["supported_formats"]:
                return domain
        raise ValueError(f"File format '{fmt}' is not associated with any known domain.")

    def _get_filetype(self, filepath):

        '''
        Helper function to extract the file extension from a given file path.

        Parameters:
        ----------
        filepath: str
            The file path from which to extract the file extension.

        Returns:
        -------
        str
        '''
        
        return Path(filepath).suffix.lower()

    def _match_system_properties(self, update_dict: dict):

        '''
        Function to match the global system properties with the provided dictionary. 
        This can be used to update the global system properties based on the information extracted from the files.

        Parameters:
        ----------
        update_dict: dict
            A dictionary containing global system properties and their values.

        Returns:
        -------
        bool
            True if the global system properties are successfully updated, otherwise False.
        '''

        if update_dict is None:
            return False

        for update_key in update_dict.keys():
            if update_key not in self.global_system_properties:
                print(f"Key Warning: Invalid global system property: {update_key}")

        for key in self.global_system_properties.keys():
            if key in update_dict:
                self.global_system_properties[key] = update_dict[key]

        return True


    # -------------------------------------------------------------------------
    # VALIDATION FUNCTIONS
    # -------------------------------------------------------------------------

    def _validate_filepath(self, filepath):

        '''
        Function to check if a given file path exists and is a regular file. Raises appropriate exceptions if the path does not exist, is a directory, or is not a regular file.

        Parameters:
        ----------
        filepath: str
            The file path to check for existence and validity.

        Returns:
        -------
        bool
            True if the file exists and is a regular file, otherwise raises an exception.
        '''
        
        p = Path(filepath)

        if not p.exists():
            raise FileNotFoundError(f"Path does not exist: {filepath}")

        if p.is_dir():
            raise IsADirectoryError(f"Expected a file but got a directory: {filepath}")

        if not p.is_file():
            raise OSError(f"Path exists but is not a regular file: {filepath}")
        
        return True

    def _validate_query(self,
                        query_dict: dict,
                        query_type: str,  ):
        
        '''
        Function checks validity of a query dictionary against the available keywords 
        for the specified query type (typing, topology, trajectory).

        Parameters:
        ----------
        query_dict: dict
            The query dictionary to be validated.
        query_type: str
            The type of query, e.g. "typing", "topology", "trajectory".
        
        Return:
        ------
        bool
            True for valid request.
        
        Raises:
        ------
        TypeError
            If query_dict is not a dictionary.
        ValueError
            If query_type is invalid or query keys are not recognised.
        '''

        if query_dict is None:
            return True

        if not isinstance(query_dict, dict):
            raise TypeError(
                f"Query must be a dictionary. Received type: {type(query_dict).__name__}"
            )

        match query_type:
            case "typing":
                keywords_available = self.type_file_keys
            case "topology":
                keywords_available = self.topo_file_keys
            case "trajectory":
                keywords_available = self.traj_file_keys
            case _:
                raise ValueError(
                    f"Invalid query_type '{query_type}'. "
                    f"Valid options are: 'typing', 'topology', 'trajectory'."
                )

        invalid_keys = [key for key in query_dict if key not in keywords_available]

        if invalid_keys:
            raise ValueError(
                f"Invalid query keyword(s) for {query_type} query: {invalid_keys}\n"
                f"Available keywords are: {sorted(keywords_available)}"
            )

        return True

    def _validate_request(self,
                          request_string: str,
                          query_type:     str, ):
        
        '''
        Function checks validity of a request string against the available requests 
        for the specified query type (typing, topology, trajectory).

        Parameters:
        ----------
        request_string: str
            The request string to be validated.
        query_type: str
            The type of query, e.g. "typing", "topology", "trajectory".
        
        Return:
        ------
        bool
            True for valid request.
        
        Raises:
        ------
        TypeError
            If request_string is not a string.
        ValueError
            If request_string or query_type is invalid.
        '''

        if not isinstance(request_string, str):
            raise TypeError(
                f"Request must be a string. Received type: {type(request_string).__name__}"
            )

        request_string = request_string.strip()

        match query_type:
            case "typing":
                requests_available = self.type_file_reqs
            case "topology":
                requests_available = self.topo_file_reqs
            case "trajectory":
                requests_available = self.traj_file_reqs
            case _:
                raise ValueError(
                    f"Invalid query_type '{query_type}'. "
                    f"Valid options are: 'typing', 'topology', 'trajectory'."
                )

        if request_string not in requests_available:
            raise ValueError(
                f"Invalid request '{request_string}' for {query_type} query.\n"
                f"Available requests are: {sorted(requests_available)}"
            )

        return True
    
    def _validate_domain_module_contract(self, domain: str, fmt: str, module) -> None:

        '''
        Validate that a parsing module satisfies the required interface for a
        given domain and file format.

        Parameters
        ----------
        domain : str
            The domain for which to validate the parsing module, one of
            "typing", "topology", or "trajectory".
        fmt : str
            The file format for which to validate the parsing module,
            e.g. "pdb", "psf", "dcd".
        module : module
            The imported parsing module to validate against the contract
            defined in the domain registry.

        Returns
        -------
        None

        Raises
        ------
        AttributeError
            If the parsing module is missing any required function specified
            in the domain registry for the given domain.
        '''

        cfg = self._domain_registry[domain]
        required = [
            cfg["keys_fn_template"].format(fmt=fmt),
            cfg["plan_fn_template"].format(fmt=fmt),
            cfg["plan_shape_fn_template"].format(fmt=fmt),
            cfg["query_fn_template"].format(fmt=fmt),
            cfg["update_fn_template"].format(fmt=fmt),
        ]

        missing = [name for name in required if not hasattr(module, name)]
        if missing:
            raise AttributeError(
                f"Parser module '{fmt}_parse' is missing required {domain} "
                f"function(s): {missing}"
            )
        
    def _validate_atom_count_consistency(self, just_loaded_domain: str) -> None:

        '''
        Cross-validate atom counts across all currently loaded domain files.

        Called automatically after every file load. If two or more domains
        report a ``num_atoms`` value and those values disagree, a ``ValueError``
        is raised with a clear per-file breakdown so the user can immediately
        identify the mismatch.

        A domain contributes to the check only when:
          - a file for that domain is loaded, AND
          - its ``_update_*_globals_*`` function returns a dict containing
            a non-None ``num_atoms`` key.

        Parameters:
        ----------
        just_loaded_domain : str
            The domain that was *just* loaded ("typing", "topology", or
            "trajectory").  Used only for the error message so the user knows
            which file triggered the check.

        Returns:
        -------
        None

        Raises:
        ------
        ValueError
            If two or more loaded domains report different atom counts.
        '''

        # Collect (domain_label, filepath, num_atoms) for each loaded domain
        # that has successfully reported a count.
        domain_counts: list[tuple[str, Path, int]] = []

        for domain, cfg in self._domain_registry.items():
            filepath = getattr(self, cfg["file_attr"])
            if filepath is None:
                continue

            fmt = getattr(self, cfg["type_attr"])[1:]   # strip the leading dot
            module = self._module_cache.get(fmt)
            if module is None:
                continue

            update_fn = getattr(module, cfg["update_fn_template"].format(fmt=fmt), None)
            if update_fn is None:
                continue

            reported = update_fn(filepath)
            if not isinstance(reported, dict):
                continue

            n = reported.get("num_atoms")
            if n is not None:
                domain_counts.append((cfg["label"], filepath, int(n)))

        if len(domain_counts) < 2:
            return  # nothing to compare yet

        counts = {n for _, _, n in domain_counts}
        if len(counts) == 1:
            return  # all agree. happy path

        lines = ["Atom count mismatch across loaded files — these files may not"]
        lines.append("belong to the same simulation:\n")
        for label, fp, n in domain_counts:
            lines.append(f"  {label:10s}  {n:>8d} atoms   ({fp.name})")
        raise ValueError("\n".join(lines))