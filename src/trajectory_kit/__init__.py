# src/trajectory_kit/__init__.py

from trajectory_kit.main import sim
from trajectory_kit.pos_writer import write_with_frame, write_with_frame_from_paths

__version__ = "0.4.0"
__all__ = ["sim", "write_with_frame", "write_with_frame_from_paths"]