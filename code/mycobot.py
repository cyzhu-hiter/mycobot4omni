from typing import Optional
import numpy as np
from pathlib import Path
from omni.isaac.core.robots.robot import Robot
from omni.isaac.core.utils.stage import add_reference_to_stage
from omniisaacgymenvs.tasks.utils.usd_utils import set_drive

from omni.isaac.core.utils.prims import get_prim_at_path
from pxr import PhysxSchema

import math
import torch
import pathlib

class myCobot(Robot):
    def __init__(
        self,
        prim_path: str,
        name: Optional[str] = "mycobot",
        usd_path: Optional[str] = None,
        translation: Optional[np.ndarray] = torch.tensor([.0, .0, .000]),
        orientation: Optional[np.ndarray] = None,
        # scale: Optional[np.ndarray] = torch.tensor([1.0, 1.0, 1.0]),
        # scaling_factor: Optional[float] = 1.0,
    ) -> None:

        self._usd_path = usd_path
        self._name = name

        self._position = torch.tensor([1.0, 0.0, 0.0]) if translation is None else translation
        # self._orientation = torch.tensor([0.0, 0.0, 0.0, 1.0]) if orientation is None else orientation

        if self._usd_path is None:
            file_name = "mycobot4omni/robot/mycobot_v6_instance.usd"
            intanceable_asset_usd = pathlib.Path(__file__).resolve().parents[2] / file_name
            self._usd_path = str(intanceable_asset_usd)

        add_reference_to_stage(self._usd_path, prim_path)

        super().__init__(
            prim_path=prim_path,
            name=name,
            translation=translation, # times scaling_factor
            # orientation=orientation,
            # scale=scale*scaling_factor,
            articulation_controller=None,
        )

        dof_paths = [
            "base_link/joint1",
            "link1/joint2",
            "link2/joint3",
            "link3/joint4",
            "link4/joint5",
            "link5/joint6",
            "gripper_base/left_gear_joint",
            "gripper_base/right_gear_joint",
        ]

        drive_type = ["angular"] * 8
        default_dof_pos = [-180, -45, 0, -45, -90, -90, 0, 0]
        # default_dof_pos = [0, 0, 0, 0, 0, 0, 0]
        stiffness = [400*np.pi/180] * 6 + [30, 30]
        damping = [80*np.pi/180] * 8
        max_force = [70, 50, 50, 50, 12, 12, 100, 100]
        max_velocity = [math.degrees(x) for x in [2.175, 2.175, 2.175, 2.175, 2.61, 2.61]] + [0.5, 0.5]

        for i, dof in enumerate(dof_paths):
            set_drive(
                prim_path=f"{self.prim_path}/{dof}",
                drive_type=drive_type[i],
                target_type="position",
                target_value=default_dof_pos[i],
                stiffness=stiffness[i],
                damping=damping[i],
                max_force=max_force[i]
            )

            PhysxSchema.PhysxJointAPI(get_prim_at_path(f"{self.prim_path}/{dof}")).CreateMaxJointVelocityAttr().Set(max_velocity[i])





