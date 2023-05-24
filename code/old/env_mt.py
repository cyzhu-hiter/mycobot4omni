import __init__

from symbol import parameters
import gym
from gym import spaces
from rl_task import RLTask

import numpy as np
import os
import math
import carb
import torch

from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.utils.types import ArticulationAction


class MyCobotEnv(RLTask):
    metadata = {"render.modes": ["human"]}

    def __init__(self, args, env, sim_config=None) -> None:
        # self._sim_config = sim_config
        # self._cfg = sim_config.config

        self._num_envs = 2
        self._env_spacing = 1 # Unit meter
        self._mycobot_positions = torch.tensor([0.0, 0.0, 0.2])
        self.scaling_factor = 1
        self.revolute_joint_limit = 170
        self.gripper_joint_limit = 45
        self.max_force = 7000

        self._num_observations = 18
        self._num_actions = 7

        self.action_space = spaces.Box(low=-np.ones(6), high=np.ones(6))
        
        self.goal_th = 0.02 

        RLTask.__init__(self, name=args.name, env=env)     
        return
    
    def set_up_scene(self, scene) -> None:
        from mycobot import MyCobot
        mycobot = MyCobot(
            prim_path=self.default_zero_env_path + "/mycobot",
            name="mycobot"
        )
        # self._sim_config.apply_articulation_settings("Mycobot", get_prim_at_path(mycobot.prim_path), self._sim_config.parse_actor_config("Mycobot"))

        from omni.isaac.core.prims import RigidPrimView
        from omni.isaac.core.objects import DynamicCuboid

        scene.add(
            DynamicCuboid(
                prim_path="/World/envs/env_0/cube",
                name="dynamic_cube",
                position=np.array([0.00, -0.20, 0.025]),
                size=0.03,
                color=np.array([1.0, 0, 0]),
            )
        )

        super().set_up_scene(scene)

        # self._set_camera()
        self._mycobots = ArticulationView(
            prim_paths_expr="/World/envs/.*/mycobot", 
            name="mycobot_view",
            reset_xform_properties=False)
        scene.add(self._mycobots)

        self._objects = RigidPrimView(prim_paths_expr="/World/envs/env_.*/cube", 
                                      name="cube_view",
                                      reset_xform_properties=False)
        scene.add(self._objects)

        from omni.isaac.core.prims.xform_prim_view import XFormPrimView
        self._tips = XFormPrimView(prim_paths_expr="/World/envs/env_.*/mycobot/gripper_base/tip", 
                                   name="tip_view",
                                   reset_xform_properties=False)

        self.actuators_idx = [i for i in range(6)]

    def get_observations(self):
        # self._world.render()

        mycobot_joint_pos = self._mycobots.get_joint_positions(clone=False)
        mycobot_joint_vel = self._mycobots.get_joint_velocities(clone=False)
        mycobot_tip_pos, _ = self._tips.get_world_poses()
        goal_world_pos, _ = self._objects.get_world_poses()

        mycobot_actor_pos = mycobot_joint_pos[:,self.actuators_idx]
        mycobot_actor_vel = mycobot_joint_vel[:,self.actuators_idx]

        self.obs_buf[:,0:6] = mycobot_actor_pos
        self.obs_buf[:,6:12] = mycobot_actor_vel
        self.obs_buf[:,12:15] = mycobot_tip_pos
        self.obs_buf[:,15:18] = goal_world_pos
        
        obs = {self._mycobots.name: {"obs_buf": self.obs_buf}}
        return obs
    
    def pre_physics_step(self, actions) -> None:
        if not self._env._world.is_playing():
            return
        
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        self.previous_mycobot_tip_pos, _ = self._tips.get_world_poses()

        if len(reset_env_ids) > 0:
            self.reset_idx(reset_env_ids)

        actions = actions.to(self._device)

        target_pos = torch.zeros((self._mycobots.count, self._num_actions), dtype=torch.float32, device=self._device)
        target_pos[:, self.actuators_idx] = np.radians(self.revolute_joint_limit) * actions
        # force = torch.zeros((self._mycobots.count, self._num_actions), dtype=torch.float32, device=self._device)
        # force[:,self.actuators_idx] = self.max_force * actions

        indices = torch.arange(self._mycobots.count, dtype=torch.int32, device=self._device)
        self._mycobots.set_joint_positions(target_pos, indices=indices) ###?
        # self._mycobots.set_joint_efforts(force, indices=indices, joint_indices=self.actuators_idx) ###?

        # for i in range(2):
        # self._mycobots.apply_action(
        #     ArticulationAction(joint_efforts=force[0,:],joint_indices=self.actuators_idx))

    def reset_idx(self, env_ids):
        num_resets = len(env_ids)

        # randomize DOF positions
        dof_pos = torch.zeros((num_resets, 6), device=self._device)

        # randomize DOF velocities
        dof_vel = torch.zeros((num_resets, 6), device=self._device)

        # apply resets
        indices = env_ids.to(dtype=torch.int32)
        self._mycobots.set_joint_positions(dof_pos, indices=indices, joint_indices=self.actuators_idx)
        self._mycobots.set_joint_velocities(dof_vel, indices=indices, joint_indices=self.actuators_idx)

        # bookkeeping
        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0

    def calculate_metrics(self) -> None:
        # mycobot_actor_pos = self.obs_buf[:,0:6] 
        # mycobot_joint_vel = self.obs_buf[:,6:12] 
        mycobot_tip_pos = self.obs_buf[:,12:15] 
        goal_world_pos = self.obs_buf[:,15:18] 

        previous_dist_to_goal = torch.linalg.norm(goal_world_pos - self.previous_mycobot_tip_pos, dim=1)
        current_dist_to_goal = torch.linalg.norm(goal_world_pos - mycobot_tip_pos, dim=1)
        reward = previous_dist_to_goal - current_dist_to_goal # format?

        reward = torch.where(mycobot_tip_pos[:,2]>0.015, reward, reward-0.1)
        reward = torch.where(current_dist_to_goal>self.goal_th, reward, reward+1)
        self.rew_buf[:] = reward

    def is_done(self) -> None:
        mycobot_tip_pos = self.obs_buf[:,12:15] 
        goal_world_pos = self.obs_buf[:,15:18] 
        current_dist_to_goal = torch.linalg.norm(goal_world_pos - mycobot_tip_pos, dim=1)

        resets = torch.where(mycobot_tip_pos[:,2]>0.015, 0, 1)
        resets = torch.where(current_dist_to_goal<self.goal_th, 1, resets)
        self.reset_buf[:] = resets

    # def set_world_window(self):
    #     import omni.kit
    #     from omni.isaac.synthetic_utils import SyntheticDataHelper

    #     viewport_handle = omni.kit.viewport_legacy.get_viewport_interface().create_instance()
    #     new_viewport_name = omni.kit.viewport_legacy.get_viewport_interface().get_viewport_window_name(viewport_handle)
    #     viewport_window = omni.kit.viewport_legacy.get_viewport_interface().get_viewport_window(viewport_handle)
    #     viewport_window.set_active_camera("/mycobot/camera/Camera")
    #     viewport_window.set_texture_resolution(64, 64)
    #     # viewport_window.set_camera_position("/OmniverseKit_Persp",100,100,60,True)
    #     # viewport_window.set_camera_target("/OmniverseKit_Persp",0,0,0,True)
    #     viewport_window.set_window_pos(1000, 400)
    #     viewport_window.set_window_size(420, 420)
    #     self.viewport_window = viewport_window
    #     self.sd_helper = SyntheticDataHelper()
    #     self.sd_helper.initialize(sensor_names=["rgb"], viewport=self.viewport_window)
    #     self._my_world.render()
    #     self.sd_helper.get_groundtruth(["rgb"], self.viewport_window)
    #     return