from typing import Optional
import numpy as np
from pathlib import Path
from omni.isaac.core.robots.robot import Robot
from omni.isaac.core.utils.stage import add_reference_to_stage
import torch
import pathlib

class MyCobot(Robot):
    def __init__(
        self,
        prim_path: str,
        name: Optional[str] = "mycobot",
        usd_path: Optional[str] = None,
        translation: Optional[np.ndarray] = torch.tensor([.0, .0, .0]),
        orientation: Optional[np.ndarray] = None,
        scale: Optional[np.ndarray] = torch.tensor([1.0, 1.0, 1.0]),
        scaling_factor: Optional[float] = 1.0,
    ) -> None:

        self._usd_path = usd_path
        self._name = name
 
        if self._usd_path is None:
            # file_name = "mycobot/robot/urdf_mycobot_rejoint_final.usd"
            file_name = "mycobot/robot/urdf_mycobot_collision_test.usd"
            # file_name = "mycobot/robot/urdf_mycobot_rejoint_without_drive.usd"
            intanceable_asset_usd = pathlib.Path(__file__).resolve().parents[2] / file_name
            self._usd_path = str(intanceable_asset_usd)

        add_reference_to_stage(self._usd_path, prim_path)

        super().__init__(
            prim_path=prim_path,
            name=name,
            translation=translation * scaling_factor,
            orientation=orientation,
            scale=scale * scaling_factor,
            articulation_controller=None,
        )
