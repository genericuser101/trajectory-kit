"""
test_queries_real.py
====================
Full pipeline integration tests against real simulation files.
All tests are automatically skipped if test_paths.json is absent or
the named paths do not exist on the current machine.

To run locally:
    cp tests/test_paths.template.json tests/test_paths.json
    # fill in your paths, then:
    pytest tests/test_queries_real.py -v
"""

from __future__ import annotations

import numpy as np
import pytest

from conftest import requires_real_file


@requires_real_file("pdb", "psf", "dcd")
class TestRealFullPipeline:

    def test_positions_via_type_query(self, sim_real):
        out = sim_real.positions(
            TYPE_Q={"atom_name": ({"CA"}, set())},
            TRAJ_Q={"frame_interval": (0, 0, 1)},
        )
        assert out["mode"] == "positions"
        assert out["selection"]["count"] > 0
        pos, meta = out["payload"]
        assert pos.ndim == 3
        assert pos.shape[1] == out["selection"]["count"]
        assert pos.shape[2] == 3

    def test_positions_type_topo_intersection(self, sim_real):
        out = sim_real.positions(
            TYPE_Q={"atom_name": ({"K", "POT", "POTD"}, set())},
            TOPO_Q={"residue_name": ({"IONS"}, set())},
            TRAJ_Q={"frame_interval": (0, 0, 1)},
        )
        type_ids = set(sim_real.get_types(
            QUERY={"atom_name": ({"K", "POT", "POTD"}, set())}, REQUEST="global_ids"
        ))
        topo_ids = set(sim_real.get_topology(
            QUERY={"residue_name": ({"IONS"}, set())}, REQUEST="global_ids"
        ))
        # selection["count"] reflects the intersection size; verify it matches
        # the manually computed intersection without relying on a non-existent
        # selection["global_ids"] key.
        expected_count = len(type_ids & topo_ids)
        assert out["selection"]["count"] == expected_count

    def test_positions_output_is_finite(self, sim_real):
        out = sim_real.positions(
            TYPE_Q={"atom_name": ({"CA"}, set())},
            TRAJ_Q={"frame_interval": (0, 0, 1)},
        )
        pos, _ = out["payload"]
        assert np.all(np.isfinite(pos))

    def test_positions_empty_query_returns_all_atoms(self, sim_real):
        # Run without planFlag so count is populated from real execution.
        out = sim_real.positions(
            TYPE_Q={},
            TRAJ_Q={"frame_interval": (0, 0, 1)},
        )
        all_ids = sim_real.get_types(QUERY={}, REQUEST="global_ids")
        assert out["selection"]["count"] == len(all_ids)

    def test_positions_frame_step_shape(self, sim_real):
        out_step1 = sim_real.positions(
            TYPE_Q={"atom_name": ({"CA"}, set())},
            TRAJ_Q={"frame_interval": (0, 9, 1)},
        )
        out_step2 = sim_real.positions(
            TYPE_Q={"atom_name": ({"CA"}, set())},
            TRAJ_Q={"frame_interval": (0, 9, 2)},
        )
        pos1, _ = out_step1["payload"]
        pos2, _ = out_step2["payload"]
        assert pos1.shape[0] == 10
        assert pos2.shape[0] == 5

    def test_num_atoms_matches_global_system_property(self, sim_real):
        all_ids  = sim_real.get_types(QUERY={}, REQUEST="global_ids")
        reported = sim_real.global_system_properties["num_atoms"]
        if reported is not None:
            assert reported == len(all_ids)

    def test_num_frames_matches_trajectory_read(self, sim_real):
        reported = sim_real.global_system_properties["num_frames"]
        if reported is not None:
            out = sim_real.positions(
                TYPE_Q={"atom_name": ({"CA"}, set())},
                TRAJ_Q={"frame_interval": (0, reported - 1, 1)},
            )
            pos, meta = out["payload"]
            assert meta["n_frames_read"] == reported