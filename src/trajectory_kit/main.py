# Preloaded Library
import os
from pathlib import Path
import importlib

# Third Party Imports: (numpy)
import numpy as np
from trajectory_kit import file_parse_help as fph
from trajectory_kit import query_help as qh

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
                "keys_fn_template": "_get_type_keys_reqs_{fmt}",
                "plan_fn_template": "_plan_type_query_{fmt}",
                "query_fn_template": "_get_type_query_{fmt}",
                "update_fn_template": "_update_type_globals_{fmt}",
                "label": "typing",
            },

            "topology": {
                "supported_formats": {".psf", ".mae"},
                "file_attr": "top_file",
                "type_attr": "top_type",
                "keys_attr": "topo_file_keys",
                "reqs_attr": "topo_file_reqs",
                "keys_fn_template": "_get_topology_keys_reqs_{fmt}",
                "plan_fn_template": "_plan_topology_query_{fmt}",
                "query_fn_template": "_get_topology_query_{fmt}",
                "update_fn_template": "_update_topology_globals_{fmt}",
                "label": "topology",
            },

            "trajectory": {
                "supported_formats": {".dcd"},
                "file_attr": "traj_file",
                "type_attr": "traj_type",
                "keys_attr": "traj_file_keys",
                "reqs_attr": "traj_file_reqs",
                "keys_fn_template":  "_get_trajectory_keys_reqs_{fmt}",
                "plan_fn_template":  "_plan_trajectory_query_{fmt}",
                "query_fn_template": "_get_trajectory_query_{fmt}",
                "update_fn_template":"_update_trajectory_globals_{fmt}",
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

        #print(sep)
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
                  updateFlag: bool = False,
                  planFlag: bool = False   ) -> dict:
        
        '''
        Return positions for atoms selected by typing and/or topology queries.

        When a trajectory file is loaded, positions are read from it (all
        selected frames).  When no trajectory file is loaded, positions are
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

        The typing and topology layers are always treated as global_id selectors
        in this function.

        Parameters:
        ----------
        TYPE_Q, TOPO_Q, TRAJ_Q : dict, optional
            Query dictionaries for typing, topology, and trajectory respectively.
            TRAJ_Q is ignored when no trajectory file is loaded.

        updateFlag : bool, default=False
            Passed through to the underlying domain handlers.

        planFlag : bool, default=False
            If True, return the execution plan without executing the query.

        Returns:
        -------
        dict
            Structured output containing:
                - mode          "positions"
                - selection     atom summary
                - plan          execution plan
                - payload       positions array and metadata
        '''

        TYPE_Q = self._normalise_query(TYPE_Q)
        TOPO_Q = self._normalise_query(TOPO_Q)
        TRAJ_Q = self._normalise_query(TRAJ_Q)

        type_provided = bool(TYPE_Q)
        topo_provided = bool(TOPO_Q)
        static_domain = None

        # ------------------------------------------------------------------ #
        # Build plan — validation, zero file I/O, stochastic planner only.   #
        # ------------------------------------------------------------------ #

        if self.traj_file is not None:
            if not type_provided and not topo_provided:
                if self.type_file is None and self.top_file is None:
                    raise ValueError(
                        "positions() requires at least one of: a typing file, a topology "
                        "file, or explicit TYPE_Q / TOPO_Q to define the atom selection."
                    )
            plan = self._plan_domain_request(
                domain="trajectory",
                query_dictionary=TRAJ_Q,
                request_string="positions",
            )
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
            plan = self._plan_domain_request(
                domain=static_domain,
                query_dictionary=TYPE_Q if static_domain == "typing" else TOPO_Q,
                request_string="positions",
            )

        out = {
            "mode": "positions",
            "selection": {
                "count":      None,
                "sources":    [],
                "merge_mode": "intersection",
            },
            "plan":    plan,
            "payload": None,
        }

        if planFlag:
            return out

        # ------------------------------------------------------------------ #
        # Resolve atom selection                                              #
        # ------------------------------------------------------------------ #

        if type_provided or topo_provided:
            selected_globals = self._resolve_atom_selection(
                TYPE_Q=TYPE_Q if type_provided else None,
                TOPO_Q=TOPO_Q if topo_provided else None,
                updateFlag=updateFlag,
            )
        else:
            selected_globals = None

        # ------------------------------------------------------------------ #
        # Trajectory path                                                      #
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

            out["selection"]["count"] = len(selected_globals)
            out["selection"]["sources"] = [
                name for name, q in (("typing", TYPE_Q), ("topology", TOPO_Q)) if q
            ]

            out["payload"] = self.get_trajectory(
                QUERY=traj_query,
                REQUEST="positions",
                updateFlag=updateFlag,
            )
            return out

        # ------------------------------------------------------------------ #
        # Static path                                                        #
        # ------------------------------------------------------------------ #

        assert static_domain is not None, \
            "static_domain was not resolved — this is a bug. also how?"

        static_query = dict(TYPE_Q if static_domain == "typing" else TOPO_Q)

        if selected_globals is None:
            if static_domain == "typing":
                selected_globals = self.get_types(
                    QUERY=static_query, REQUEST="global_ids", updateFlag=updateFlag,
                )
            else:
                selected_globals = self.get_topology(
                    QUERY=static_query, REQUEST="global_ids", updateFlag=updateFlag,
                )

        out["selection"]["count"] = len(selected_globals)
        out["selection"]["sources"] = [
            name for name, q in (("typing", TYPE_Q), ("topology", TOPO_Q)) if q
        ]

        pos = self._execute_domain_request(
            domain=static_domain,
            query_dictionary=static_query,
            request_string="positions",
            updateFlag=updateFlag,
        )

        if pos is not None:
            all_ids = self._execute_domain_request(
                domain=static_domain,
                query_dictionary=static_query,
                request_string="global_ids",
                updateFlag=updateFlag,
            )
            gid_set = set(selected_globals)
            mask    = [i for i, g in enumerate(all_ids) if g in gid_set]
            pos     = pos[:, mask, :]

        out["payload"] = pos
        return out
    
    
    def select(self,
                *,
                TYPE_Q: dict = None,
                TOPO_Q: dict = None,
                TRAJ_Q: dict = None,
                TYPE_R: str = None,
                TOPO_R: str = None,
                TRAJ_R: str = None,
                updateFlag: bool = False,
                planFlag: bool = False   ) -> dict:
        
        '''
        Return property values from typing, topology, and/or trajectory domains.

        This function is for property-style requests only.
        It does not return trajectory coordinate payloads.

        Parameters:
        ----------
        TYPE_Q, TOPO_Q, TRAJ_Q : dict, optional
            Query dictionaries for typing, topology, and trajectory respectively.

        TYPE_R, TOPO_R, TRAJ_R : str, optional
            Property request strings for typing, topology, and trajectory.

        updateFlag : bool, default=False
            Passed through to the underlying domain handlers.

        Returns:
        -------
        dict
            Structured output containing returned property payloads by domain.
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

        plan = {}
        payload = {}

        if TYPE_R is not None:
            plan["typing"] = self._plan_domain_request(
                domain="typing",
                query_dictionary=TYPE_Q,
                request_string=TYPE_R,
            )

        if TOPO_R is not None:
            plan["topology"] = self._plan_domain_request(
                domain="topology",
                query_dictionary=TOPO_Q,
                request_string=TOPO_R,
            )

        if TRAJ_R is not None:
            plan["trajectory"] = self._plan_domain_request(
                domain="trajectory",
                query_dictionary=TRAJ_Q,
                request_string=TRAJ_R,
            )

        out = {
            "mode": "property",
            "plan": plan,
            "payload": None,
        }

        if planFlag:
            return out

        if TYPE_R is not None:
            payload["typing"] = self.get_types(
                QUERY=TYPE_Q,
                REQUEST=TYPE_R,
                updateFlag=updateFlag,
            )

        if TOPO_R is not None:
            payload["topology"] = self.get_topology(
                QUERY=TOPO_Q,
                REQUEST=TOPO_R,
                updateFlag=updateFlag,
            )

        if TRAJ_R is not None:
            payload["trajectory"] = self.get_trajectory(
                QUERY=TRAJ_Q,
                REQUEST=TRAJ_R,
                updateFlag=updateFlag,
            )

        out["payload"] = payload
        return out
    

    def fetch(self,
              *,
              TYPE_Q: dict = None,
              TOPO_Q: dict = None,
              TRAJ_Q: dict = None,
              TYPE_R: str = None,
              TOPO_R: str = None,
              TRAJ_R: str = None,
              updateFlag: bool = False,
              planFlag: bool = False) -> dict:

        '''
        Return per-atom or coordinate payloads from any combination of
        typing, topology, and trajectory domains.

        This is the general-purpose extraction function. positions() is a
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

        Examples:

            TYPE_Q + TOPO_R="charges"
                → resolve typing global_ids, return charges for those atoms

            TYPE_Q + TOPO_Q + TOPO_R="charges" + TYPE_R="atom_names"
                → intersect typing and topology selections, apply to both

            TYPE_Q + TOPO_Q + TRAJ_R="positions" + TRAJ_Q={"frame_interval": ...}
                → intersect typing and topology, pass as global_ids to trajectory

        Parameters:
        ----------
        TYPE_Q, TOPO_Q, TRAJ_Q : dict, optional
            Query dictionaries for typing, topology, and trajectory respectively.

        TYPE_R, TOPO_R, TRAJ_R : str, optional
            Request strings for typing, topology, and trajectory respectively.
            Any non-property request is accepted, e.g. "charges", "masses",
            "atom_names", "positions".

        updateFlag : bool, default=False
            Passed through to the underlying domain handlers.

        planFlag : bool, default=False
            If True, return the execution plan without executing the query.

        Returns:
        -------
        dict
            {
                "mode":      "fetch",
                "selection": {
                    "count":      int   | None,
                    "sources":    [...],
                    "merge_mode": "intersection",
                },
                "plan":    { domain: plan, ... },
                "payload": { domain: payload, ... },
            }

        Raises:
        ------
        ValueError
            If no request string is provided, if a request is a property-*
            request (use select()), or if the cross-domain intersection is empty.
        '''

        TYPE_Q = self._normalise_query(TYPE_Q)
        TOPO_Q = self._normalise_query(TOPO_Q)
        TRAJ_Q = self._normalise_query(TRAJ_Q)

        type_provided = bool(TYPE_Q)
        topo_provided = bool(TOPO_Q)
        traj_provided = bool(TRAJ_Q)

        TYPE_R = self._normalise_request(TYPE_R)
        TOPO_R = self._normalise_request(TOPO_R)
        TRAJ_R = self._normalise_request(TRAJ_R)

        if TYPE_R is None and TOPO_R is None and TRAJ_R is None:
            raise ValueError("fetch() requires at least one request string (TYPE_R, TOPO_R, or TRAJ_R).")

        if TYPE_R is not None and self._classify_request("typing", TYPE_R) == "property":
            raise ValueError(
                f"TYPE_R='{TYPE_R}' is a property request. "
                "Use select() for property-* requests."
            )

        if TOPO_R is not None and self._classify_request("topology", TOPO_R) == "property":
            raise ValueError(
                f"TOPO_R='{TOPO_R}' is a property request. "
                "Use select() for property-* requests."
            )

        if TRAJ_R is not None and self._classify_request("trajectory", TRAJ_R) == "property":
            raise ValueError(
                f"TRAJ_R='{TRAJ_R}' is a property request. "
                "Use select() for property-* requests."
            )

        # ------------------------------------------------------------------ #
        # Build plans — zero file I/O, stochastic planner only.              #
        # ------------------------------------------------------------------ #

        plan = {}

        if TYPE_R is not None:
            plan["typing"] = self._plan_domain_request(
                domain="typing",
                query_dictionary=TYPE_Q,
                request_string=TYPE_R,
            )

        if TOPO_R is not None:
            plan["topology"] = self._plan_domain_request(
                domain="topology",
                query_dictionary=TOPO_Q,
                request_string=TOPO_R,
            )

        if TRAJ_R is not None:
            plan["trajectory"] = self._plan_domain_request(
                domain="trajectory",
                query_dictionary=TRAJ_Q,
                request_string=TRAJ_R,
            )

        out = {
            "mode": "fetch",
            "selection": {
                "count":      None,
                "sources":    [],
                "merge_mode": "intersection",
            },
            "plan":    plan,
            "payload": None,
        }

        if planFlag:
            return out

        # ------------------------------------------------------------------ #
        # Resolve global_ids for intersection.                                #
        # Only domains with a query constraint contribute an id set.          #
        # ------------------------------------------------------------------ #

        type_global_ids = None
        topo_global_ids = None
        traj_global_ids = None

        if type_provided:
            self._ensure_domain_loaded("typing")
            type_global_ids = self.get_types(
                QUERY=TYPE_Q, REQUEST="global_ids", updateFlag=updateFlag,
            )

        if topo_provided:
            self._ensure_domain_loaded("topology")
            topo_global_ids = self.get_topology(
                QUERY=TOPO_Q, REQUEST="global_ids", updateFlag=updateFlag,
            )

        if traj_provided:
            self._ensure_domain_loaded("trajectory")
            traj_selection = self.get_trajectory(
                QUERY=TRAJ_Q, REQUEST="global_ids", updateFlag=updateFlag,
            )
            if traj_selection is None:
                pass   # DCD hotpath — no constraint from trajectory
            elif isinstance(traj_selection, list):
                if not all(f is None for f in traj_selection):
                    traj_ids_set: set[int] = set()
                    for frame_ids in traj_selection:
                        if frame_ids is not None:
                            traj_ids_set.update(frame_ids)
                    traj_global_ids = sorted(traj_ids_set)

        # THIS IS IMPORTANT: we hotpath by allowing None to propagate as "no constraint" from any domain, 
        # which means the intersection logic only applies to domains that actually return id sets. 
        # For example, a trajectory query that returns [None, None, ...] (no per-frame constraints) 
        # will not interfere with the intersection of typing and topology selections.

        # None is however not the same as an empty set — an empty set means "constrained to zero atoms" 
        # and will produce an empty intersection result, whereas None means "no constraint from this domain" 
        # and is effectively ignored in the intersection logic.

        active = [  # Intersect all non-None id sets
            set(ids) for ids in (type_global_ids, topo_global_ids, traj_global_ids)
            if ids is not None
        ]

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
            if TYPE_R is not None and type_global_ids is None:
                type_global_ids = self.get_types(
                    QUERY=TYPE_Q, REQUEST="global_ids", updateFlag=updateFlag,
                )
            if TOPO_R is not None and topo_global_ids is None:
                topo_global_ids = self.get_topology(
                    QUERY=TOPO_Q, REQUEST="global_ids", updateFlag=updateFlag,
                )

        out["selection"]["count"] = len(selected_globals) if selected_globals is not None else None
        out["selection"]["sources"] = [
            name for name, ids in (
                ("typing",     type_global_ids),
                ("topology",   topo_global_ids),
                ("trajectory", traj_global_ids),
            ) if ids is not None
        ]

        # ------------------------------------------------------------------ #
        # Execute and post-filter                                             #
        # ------------------------------------------------------------------ #

        all_globals = list(range(self.global_system_properties['num_atoms']))

        payload = {}
        gid_set = set(selected_globals) if selected_globals is not None else None

        if TYPE_R is not None:
            raw = self.get_types(QUERY=TYPE_Q, REQUEST=TYPE_R, updateFlag=updateFlag)
            if gid_set is not None and isinstance(raw, list):
                ids = type_global_ids if type_global_ids is not None else all_globals
                raw = [v for v, g in zip(raw, ids) if g in gid_set]
            payload["typing"] = raw

        if TOPO_R is not None:
            raw = self.get_topology(QUERY=TOPO_Q, REQUEST=TOPO_R, updateFlag=updateFlag)
            if gid_set is not None and isinstance(raw, list):
                ids = topo_global_ids if topo_global_ids is not None else all_globals
                raw = [v for v, g in zip(raw, ids) if g in gid_set]
            payload["topology"] = raw

        if TRAJ_R is not None:
            traj_query = dict(TRAJ_Q)
            if selected_globals is not None:
                traj_query["global_ids"] = (selected_globals, set())
            payload["trajectory"] = self.get_trajectory(
                QUERY=traj_query,
                REQUEST=TRAJ_R,
                updateFlag=updateFlag,
            )

        out["payload"] = payload
        return out


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
    # INTERNAL DOMAIN EXECUTION
    # -------------------------------------------------------------------------

    def _plan_domain_request(self,
                             domain: str,
                             query_dictionary: dict,
                             request_string: str, ):

        '''
        A function for creating a query plan for specified domain.

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
        The query plan for the specified domain.
        '''

        cfg = self._domain_registry[domain]
        filepath = getattr(self, cfg["file_attr"])
        filetype = getattr(self, cfg["type_attr"])
        keywords_available = getattr(self, cfg["keys_attr"])
        requests_available = getattr(self, cfg["reqs_attr"])

        fmt = filetype[1:]
        module = self._get_parse_module(fmt)

        query_fn = getattr(module, cfg["plan_fn_template"].format(fmt=fmt))
        plan = query_fn(
            filepath,
            query_dictionary=query_dictionary,
            request_string=request_string,
            keywords_available=keywords_available,
            requests_available=requests_available,
        )

        return plan

    def _execute_domain_request(self,
                                domain: str,
                                query_dictionary: dict,
                                request_string: str,
                                updateFlag: bool = False):
        
        '''
        Internal function to execute a query on a specific domain (typing, topology, trajectory) 
        based on the provided query dictionary and request string.

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
        The output of the executed query, which can be a list of global indices, a property value, or a trajectory payload depending on the request.
        
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

    def _resolve_atom_selection(self,
                                TYPE_Q: dict = None,
                                TOPO_Q: dict = None,
                                updateFlag: bool = False):
        '''
        Resolve atom global_ids from typing and/or topology queries.

        Combination rules:
        ------------------
        - type only      -> use type globals
        - topology only  -> use topology globals
        - both provided  -> use intersection

        Parameters:
        ----------
        TYPE_Q, TOPO_Q : dict, optional
            Query dictionaries for typing and topology.

        updateFlag : bool, default=False
            Passed through to the underlying domain handlers.

        Returns:
        -------
        list[int]
            Sorted global_ids defining the atom selection.
        '''

        globals_from_type = None
        globals_from_topo = None

        if TYPE_Q is not None:
            globals_from_type = self.get_types(
                QUERY=TYPE_Q,
                REQUEST="global_ids",
                updateFlag=updateFlag,
            )

        if TOPO_Q is not None:
            globals_from_topo = self.get_topology(
                QUERY=TOPO_Q,
                REQUEST="global_ids",
                updateFlag=updateFlag,
            )

        if globals_from_type is None and globals_from_topo is None:
            raise ValueError(
                "Atom selection failed: neither typing nor topology produced global_ids."
            )

        if globals_from_type is None:
            selected_globals = sorted(globals_from_topo)

        elif globals_from_topo is None:
            selected_globals = sorted(globals_from_type)

        else:
            selected_globals = sorted(set(globals_from_type) & set(globals_from_topo))

        if not selected_globals:
            print(globals_from_type)
            print(globals_from_topo)
            raise ValueError(
                "Typing and topology selections produced an empty intersection."
            )   
        

        return selected_globals


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