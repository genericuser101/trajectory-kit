"""
test_trajectory.py
==================
Complete test suite for the trajectory domain (DCD file type).
Covers every available request and frame interval arithmetic.

DCD layout (3 frames, 4 atoms):
    frame 0: atom positions = base + 0.0
    frame 1: atom positions = base + 1.0
    frame 2: atom positions = base + 2.0

Base positions:
    atom 0: x=1  y=2  z=3
    atom 1: x=4  y=5  z=6
    atom 2: x=7  y=8  z=9
    atom 3: x=10 y=11 z=12
"""

from __future__ import annotations

import numpy as np
import pytest

# ===========================================================================
# DCD — request: positions — output shape
# ===========================================================================

class TestDCDPositionsShape:

    def test_single_atom_all_frames_shape(self, sim_synthetic):
        pos, _ = sim_synthetic.get_trajectory(
            QUERY={"global_ids": ([0], set()), "frame_interval": (0, 2, 1)},
            REQUEST="positions",
        )
        assert pos.shape == (3, 1, 3)

    def test_all_atoms_single_frame_shape(self, sim_synthetic):
        pos, _ = sim_synthetic.get_trajectory(
            QUERY={"global_ids": ([0, 1, 2, 3], set()), "frame_interval": (0, 0, 1)},
            REQUEST="positions",
        )
        assert pos.shape == (1, 4, 3)

    def test_two_atoms_two_frames_shape(self, sim_synthetic):
        pos, _ = sim_synthetic.get_trajectory(
            QUERY={"global_ids": ([0, 3], set()), "frame_interval": (0, 1, 1)},
            REQUEST="positions",
        )
        assert pos.shape == (2, 2, 3)

    def test_output_dtype_is_float32(self, sim_synthetic):
        pos, _ = sim_synthetic.get_trajectory(
            QUERY={"global_ids": ([0], set()), "frame_interval": (0, 0, 1)},
            REQUEST="positions",
        )
        assert pos.dtype == np.float32

    def test_output_is_numpy_array(self, sim_synthetic):
        pos, _ = sim_synthetic.get_trajectory(
            QUERY={"global_ids": ([0], set()), "frame_interval": (0, 0, 1)},
            REQUEST="positions",
        )
        assert isinstance(pos, np.ndarray)

    def test_positions_are_finite(self, sim_synthetic):
        pos, _ = sim_synthetic.get_trajectory(
            QUERY={"global_ids": ([0, 1, 2, 3], set()), "frame_interval": (0, 2, 1)},
            REQUEST="positions",
        )
        assert np.all(np.isfinite(pos))

# ===========================================================================
# DCD — request: positions — position values
# ===========================================================================

class TestDCDPositionValues:

    def test_atom0_frame0_xyz(self, sim_synthetic):
        pos, _ = sim_synthetic.get_trajectory(
            QUERY={"global_ids": ([0], set()), "frame_interval": (0, 0, 1)},
            REQUEST="positions",
        )
        assert np.allclose(pos[0, 0, :], [1.0, 2.0, 3.0], atol=1e-4)

    def test_atom3_frame0_xyz(self, sim_synthetic):
        pos, _ = sim_synthetic.get_trajectory(
            QUERY={"global_ids": ([3], set()), "frame_interval": (0, 0, 1)},
            REQUEST="positions",
        )
        assert np.allclose(pos[0, 0, :], [10.0, 11.0, 12.0], atol=1e-4)

    def test_positions_increment_by_frame(self, sim_synthetic):
        pos, _ = sim_synthetic.get_trajectory(
            QUERY={"global_ids": ([0], set()), "frame_interval": (0, 2, 1)},
            REQUEST="positions",
        )
        xs = pos[:, 0, 0]
        assert np.allclose(xs, [1.0, 2.0, 3.0], atol=1e-4)

    def test_atom_ordering_preserved(self, sim_synthetic):
        pos, _ = sim_synthetic.get_trajectory(
            QUERY={"global_ids": ([0, 3], set()), "frame_interval": (0, 0, 1)},
            REQUEST="positions",
        )
        # atom 0 at index 0, atom 3 at index 1
        assert np.allclose(pos[0, 0, :], [1.0, 2.0, 3.0], atol=1e-4)
        assert np.allclose(pos[0, 1, :], [10.0, 11.0, 12.0], atol=1e-4)

# ===========================================================================
# DCD — request: positions — frame interval arithmetic
# ===========================================================================

class TestDCDFrameInterval:

    def test_first_frame_only(self, sim_synthetic):
        pos, meta = sim_synthetic.get_trajectory(
            QUERY={"global_ids": ([0], set()), "frame_interval": (0, 0, 1)},
            REQUEST="positions",
        )
        assert meta["n_frames_read"] == 1
        assert pos.shape[0] == 1

    def test_last_frame_only(self, sim_synthetic):
        pos, meta = sim_synthetic.get_trajectory(
            QUERY={"global_ids": ([0], set()), "frame_interval": (2, 2, 1)},
            REQUEST="positions",
        )
        assert meta["n_frames_read"] == 1
        assert np.allclose(pos[0, 0, 0], 3.0, atol=1e-4)  # x = 1 + 2

    def test_all_frames(self, sim_synthetic):
        pos, meta = sim_synthetic.get_trajectory(
            QUERY={"global_ids": ([0], set()), "frame_interval": (0, 2, 1)},
            REQUEST="positions",
        )
        assert meta["n_frames_read"] == 3
        assert pos.shape[0] == 3

    def test_step_2_returns_frames_0_and_2(self, sim_synthetic):
        pos, meta = sim_synthetic.get_trajectory(
            QUERY={"global_ids": ([0], set()), "frame_interval": (0, 2, 2)},
            REQUEST="positions",
        )
        assert meta["n_frames_read"] == 2
        assert meta["first_frame_read"] == 0
        assert meta["last_frame_read"] == 2
        assert np.allclose(pos[0, 0, 0], 1.0, atol=1e-4)  # frame 0
        assert np.allclose(pos[1, 0, 0], 3.0, atol=1e-4)  # frame 2

    def test_step_2_middle_frame_skipped(self, sim_synthetic):
        pos, _ = sim_synthetic.get_trajectory(
            QUERY={"global_ids": ([0], set()), "frame_interval": (0, 2, 2)},
            REQUEST="positions",
        )
        xs = pos[:, 0, 0].tolist()
        assert 2.0 not in [round(x, 2) for x in xs]  # frame 1 x=2.0 should be absent

    def test_start_frame_1(self, sim_synthetic):
        pos, meta = sim_synthetic.get_trajectory(
            QUERY={"global_ids": ([0], set()), "frame_interval": (1, 2, 1)},
            REQUEST="positions",
        )
        assert meta["first_frame_read"] == 1
        assert meta["n_frames_read"] == 2

    def test_stop_is_inclusive(self, sim_synthetic):
        pos, meta = sim_synthetic.get_trajectory(
            QUERY={"global_ids": ([0], set()), "frame_interval": (0, 1, 1)},
            REQUEST="positions",
        )
        assert meta["n_frames_read"] == 2
        assert meta["last_frame_read"] == 1

# ===========================================================================
# DCD — request: positions — metadata
# ===========================================================================

class TestDCDMetadata:

    def test_meta_contains_required_keys(self, sim_synthetic):
        _, meta = sim_synthetic.get_trajectory(
            QUERY={"global_ids": ([0], set()), "frame_interval": (0, 2, 1)},
            REQUEST="positions",
        )
        required = {
            "first_frame_read",
            "last_frame_read",
            "n_frames_read",
            "start",
            "stop",
            "step",
            "start_inclusive",
            "stop_inclusive",
        }
        assert required.issubset(set(meta.keys()))

    def test_meta_start_inclusive_is_true(self, sim_synthetic):
        _, meta = sim_synthetic.get_trajectory(
            QUERY={"global_ids": ([0], set()), "frame_interval": (0, 2, 1)},
            REQUEST="positions",
        )
        assert meta["start_inclusive"] is True

    def test_meta_stop_inclusive_is_false(self, sim_synthetic):
        _, meta = sim_synthetic.get_trajectory(
            QUERY={"global_ids": ([0], set()), "frame_interval": (0, 2, 1)},
            REQUEST="positions",
        )
        assert meta["stop_inclusive"] is False

    def test_meta_n_frames_matches_array(self, sim_synthetic):
        pos, meta = sim_synthetic.get_trajectory(
            QUERY={"global_ids": ([0], set()), "frame_interval": (0, 2, 2)},
            REQUEST="positions",
        )
        assert meta["n_frames_read"] == pos.shape[0]


# ===========================================================================
# DCD — additional coverage
# ===========================================================================

class TestDCDPositionsShapeExtra:

    def test_third_axis_is_always_3(self, sim_synthetic):
        pos, _ = sim_synthetic.get_trajectory(
            QUERY={"global_ids": ([0, 1], set()), "frame_interval": (0, 2, 1)},
            REQUEST="positions",
        )
        assert pos.shape[2] == 3


class TestDCDPositionValuesExtra:

    def test_all_frames_atom0_x_values(self, sim_synthetic):
        pos, _ = sim_synthetic.get_trajectory(
            QUERY={"global_ids": ([0], set()), "frame_interval": (0, 2, 1)},
            REQUEST="positions",
        )
        import numpy as np
        assert np.allclose(pos[:, 0, 0], [1.0, 2.0, 3.0], atol=1e-4)

    def test_all_frames_atom3_x_values(self, sim_synthetic):
        pos, _ = sim_synthetic.get_trajectory(
            QUERY={"global_ids": ([3], set()), "frame_interval": (0, 2, 1)},
            REQUEST="positions",
        )
        import numpy as np
        assert np.allclose(pos[:, 0, 0], [10.0, 11.0, 12.0], atol=1e-4)


class TestDCDFrameIntervalExtra:

    def test_empty_frame_interval_reads_all(self, sim_synthetic):
        pos, meta = sim_synthetic.get_trajectory(
            QUERY={"global_ids": ([0], set()), "frame_interval": ()},
            REQUEST="positions",
        )
        assert meta["n_frames_read"] == 3

    def test_step_must_be_positive(self, sim_synthetic):
        import pytest
        with pytest.raises(ValueError):
            sim_synthetic.get_trajectory(
                QUERY={"global_ids": ([0], set()), "frame_interval": (0, 2, 0)},
                REQUEST="positions",
            )

    def test_frame_interval_wrong_length_raises(self, sim_synthetic):
        import pytest
        with pytest.raises(ValueError):
            sim_synthetic.get_trajectory(
                QUERY={"global_ids": ([0], set()), "frame_interval": (0,)},
                REQUEST="positions",
            )


class TestDCDMetadataExtra:

    def test_meta_step_value_stored(self, sim_synthetic):
        _, meta = sim_synthetic.get_trajectory(
            QUERY={"global_ids": ([0], set()), "frame_interval": (0, 2, 2)},
            REQUEST="positions",
        )
        assert meta["step"] == 2

    def test_meta_start_value_stored(self, sim_synthetic):
        _, meta = sim_synthetic.get_trajectory(
            QUERY={"global_ids": ([0], set()), "frame_interval": (1, 2, 1)},
            REQUEST="positions",
        )
        assert meta["start"] == 1
