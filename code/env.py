from symbol import parameters
import gym
from gym import spaces

import numpy as np
import os
import math
import carb
import torch
import pathlib

import matplotlib.pyplot as plt 
from collections import OrderedDict
from omni.isaac.gym.vec_env import VecEnvBase


class MyCobotEnv(VecEnvBase):
    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        skip_frame=1,
        physics_dt=1.0 / 60.0,
        rendering_dt=1.0 / 60.0,
        max_episode_length= 256,  # 256
        seed=0,
        args=None
    ) -> None:
<<<<<<< HEAD

        self._skip_frame=skip_frame
        self.physics_dt=physics_dt
        self.rendering_dt=rendering_dt
        self.max_episode_length=max_episode_length

=======
>>>>>>> eccb7cfcd3f0b5824167f86968529d970a66c69c
        VecEnvBase.__init__(
            self,
            headless=args.headless,
            sim_device=0, # improve
            enable_livestream=args.enable_livestream,
            enable_viewport=args.enable_viewport
            )
        
        from omni.isaac.core import World
        from omni.isaac.core.objects import VisualCuboid, DynamicCuboid
        self.my_world = World(physics_dt=physics_dt,
                              rendering_dt=rendering_dt,
                              stage_units_in_meters=1.0)
        self.add_default_ground_plan()

        from mycobot import myCobot
        self.my_world.scene.add(
            myCobot(
                prim_path="/mycobot",
                usd_path=None,
                name="mycobot",
                translation=torch.tensor([0.0, 0.0, 0.0]),
<<<<<<< HEAD
                # scaling_factor=1.0
=======
                scaling_factor=1.0
>>>>>>> eccb7cfcd3f0b5824167f86968529d970a66c69c
            )
        )
        self.mycobot = self._my_world.scene.get_object('mycobot')

        self.goal = self._my_world.scene.add(
            VisualCuboid(
                prim_path="/goal_cube_1",
                name="visual_cube",
                position=np.array([0.20, 0, 0.01]),
                size=0.02,
                color=np.array([1.0, 0, 0]),
            )
        )

        self.object = self._my_world.scene.add(
            DynamicCuboid(
                prim_path="/object_cube_1",
                name="visual_cube",
                position=np.array([0.20, 0.05, 0.01]),
                size=0.02,
                color=np.array([0, 0.5, 0]),
            )
        )

        
        
        # from omni.isaac.kit import SimulationApp

        # self._args = args
        # self.headless = args.headless
        # self._simulation_app = SimulationApp({"headless": self.headless, "anti_aliasing": 0})
        # self._skip_frame = skip_frame
        # self._dt = physics_dt * self._skip_frame
        # self._max_episode_length = max_episode_length
        # self._steps_after_reset = int(rendering_dt / physics_dt)
        # self.joint_control_method = args.joint_control
        # self.joint_collision_threshold = 0.5/180*math.pi # .5 degree in radian
        # self.joint_effective_move = 2/180*math.pi
        # from omni.isaac.core import World
        # from omni.isaac.core.objects import VisualCuboid, DynamicCuboid

        # self._my_world = World(physics_dt=physics_dt, rendering_dt=rendering_dt, stage_units_in_meters=1.0)
        # self._my_world.scene.add_default_ground_plane()

        # from mycobot import MyCobot
        # # from omni.isaac.core.articulations import ArticulationView

        # self.mycobot = self._my_world.scene.add(
        #     MyCobot(
        #         prim_path="/mycobot",
        #         usd_path=None,
        #         name="mycobot",
        #         translation=torch.tensor([0.0, 0.0, 0.0]),
        #         scaling_factor=1.0)
        # )
        # self._mycobot = self._my_world.scene.get_object('mycobot')
        # # self.actuators = ['joint1','joint2','joint3','joint4','joint5','joint6','left_gear_joint']
        # self.actuators = ['joint1','joint2','joint3','joint4','joint5']


        # self.goal = self._my_world.scene.add(
        #     VisualCuboid(
        #         prim_path="/new_cube_1",
        #         name="visual_cube",
        #         position=np.array([0.20, 0, 0.01]),
        #         size=0.02,
        #         color=np.array([1.0, 0, 0]),
        #     )
        # )

        # from omni.isaac.core.prims.xform_prim import XFormPrim

        # self._tip = XFormPrim(prim_path="/mycobot/gripper_base/tip", name="robot_arm_tip")

        # from omni.isaac.core.utils.stage import get_current_stage
        # from pxr import UsdLux

        # stage = get_current_stage()
        # light = UsdLux.DomeLight.Define(stage, "/World/defaultDomeLight")
        # light.GetPrim().GetAttribute("intensity").Set(500)

        # self.seed(seed)
        # # self.set_world_window()
        # self.reward_range = (-float("inf"), float("inf"))
        # gym.Env.__init__(self)

        # self.space_limit = 0.5
        # self.angle_limit = math.pi*17/18
        # self.vel_limit = 50
        # self.action_space = spaces.Box(low = -self.angle_limit,
        #                                high = self.angle_limit,
        #                                shape = (len(self.actuators),),
        #                                dtype = np.float32)
        # if self.joint_control_method == 'position':  # adjust here for scale reorgnize  
        #     self.action_scale = 1
        # elif self.joint_control_method == 'efforts':
        #     self.action_scale = 12

        # # Vision Based observation_space, which need to maobservationch the method get_observation return
        # # For original RGB camera feedback, unpolished
        # # self.observation_space = spaces.Box(low=0, high=255, shape=(64, 64, 3), dtype=np.uint8)
        # # For Grey scale 
        # # self.observation_space = spaces.Box(low=0, high=255, shape=(711,), dtype=np.float32)
        # # For direct non-vision based training

        # limit = np.array([self.space_limit] * 9 +
        #                  [self.angle_limit] * len(self.actuators) +
        #                  [self.vel_limit] * len(self.actuators))
        # self.observation_space = spaces.Box(low=-limit, high=limit, dtype=np.float32)
        # # self.observation_space = spaces.Dict(
        # #         {
        # #             "observation": spaces.Box(low=-3.0, high=3, shape=(2*len(self.actuators)+6,), dtype=np.float32),
        # #             "achieved_goal": spaces.Box(low=-0.5, high=0.5, shape=(3,), dtype=np.float32),
        # #             "desired_goal": spaces.Box(low=-0.5, high=0.5, shape=(3,), dtype=np.float32),
        # #         }
        # #     )


        # # self.max_force = 1
        # # self.max_angular_velocity = math.pi
        # self.reset_counter = 0
        
        return

    def get_dt(self):
        return self._dt

    def step(self, action):
        if np.isnan(action).any():
            print('pause')
        previous_mycobot_tip_position, _ = self._tip.get_world_pose()
        previous_joints_angles = self._mycobot.get_joint_positions()[self.actuators_idx]

        # actions need to be scaled from raw to true value
        action = action * self.action_scale
        # action[self._mycobot.dof_names.index(self.actuators[-1])] /= \
        #     (17/18*180/44) if self.joint_control_method == 'position' else 8

        for i in range(self._skip_frame):
            from omni.isaac.core.utils.types import ArticulationAction

            if self.joint_control_method == 'position':
                self._mycobot.apply_action(ArticulationAction(joint_positions=action, joint_indices=self.actuators_idx))
            elif self.joint_control_method == 'efforts':
                self._mycobot.apply_action(ArticulationAction(joint_efforts=action, joint_indices=self.actuators_idx))
            self._my_world.step(render=False)

        observations = self.get_observations()
        info = {}
        done = False

        # if np.isnan(observations).any(): # truncated
        #     observations[0:5] = previous_joints_angles
        #     observations[6:10] = 10 * (action - previous_joints_angles)
        #     return observations, -5, True, info
        
        if self._my_world.current_time_step_index - self._steps_after_reset >= self._max_episode_length:
            done = True

        goal_world_position, _ = self.goal.get_world_pose()
        current_mycobot_tip_position, _ = self._tip.get_world_pose() # observations[3:6]
        previous_dist_to_goal = np.linalg.norm(goal_world_position - previous_mycobot_tip_position)
        current_dist_to_goal = np.linalg.norm(goal_world_position - current_mycobot_tip_position)
        reward = previous_dist_to_goal - current_dist_to_goal
        
        # # if observations[-4] < 0.015: # avoid gripper get too low
        #     reward -= 0.1
        # reward = 0


        # mycobot_joint_position = observations[6:6+len(self.actuators_idx)]
        # angle_diff = mycobot_joint_position - previous_joints_angles
        # joint_angles_to_go = mycobot_joint_position - action
        # complete the first 5 joint collision checking first, the sixth will be added later.
        # collision_flag = any([True if (abs(angle_diff[i]) < self.joint_collision_threshold and 
        #                                abs(joint_angles_to_go[i]) < self.joint_effective_move) else False for i in range(5)])
        # if collision_flag:
        #     done = True
        #     reward = -5


        if current_dist_to_goal < 0.02:
            done = True
            reward += 3
        # add reward to make gripper face downwards

        return observations, reward, done, info

    def reset(self):
        self.my_world.reset()
        self.reset_counter = 0
        # randomize goal location in circle around robot or torus or half sphere
        alpha = 2 * math.pi * (np.random.rand()-0.5) *3/4 # 360 degree
        beta = math.pi * np.random.rand() / 4 # 45 degree
        r = 0.06 * math.sqrt(np.random.rand()) + 0.22
        self.goal.set_world_pose(np.array([math.cos(beta) * math.cos(alpha) * r,
                                           math.cos(beta) * math.sin(alpha) * r,
                                           0.01 + math.sin(beta) * r]))
        
        self.actuators_idx = [self._mycobot.dof_names.index(actuator) for actuator in self.actuators \
                              if actuator in self._mycobot.dof_names]

        # self._mycobot.set_joint_positions(positions=np.concatenate([(np.random.rand(6)-0.5)*math.pi/3, np.zeros(1)]))
        self._mycobot.set_joint_positions(positions=np.concatenate([(np.random.rand(len(self.actuators_idx))-0.5)*math.pi/3, np.array([0])]),
                                          joint_indices=self.actuators_idx+[6])
        # self._mycobot.set_joint_positions(positions=np.array([-90,-30,0,0,90,0])*np.pi/180,joint_indices=self.actuators_idx)
        # observations = self.get_observations()
        # return observations

    def get_observations(self):
        self._my_world.render()
        mycobot_joint_position = self._mycobot.get_joint_positions()[self.actuators_idx]
        mycobot_joint_velocity = self._mycobot.get_joint_velocities()[self.actuators_idx]
        mycobot_tip_position, _ = self._tip.get_world_pose()
        goal_world_position, _ = self.goal.get_world_pose()

        observations = np.concatenate([goal_world_position,
                                      mycobot_tip_position,
                                      goal_world_position-mycobot_tip_position,
                                      mycobot_joint_position,
                                      mycobot_joint_velocity])
        # return np.concatenate(
        #     [   
        #         mycobot_joint_position[:4],
        #         mycobot_joint_velocity[:4],
        #         mycobot_tip_position[0::2],
        #         goal_world_position[0::2]
        #     ]
        # ) # shape(4+4+2+2,1) = (12,1)

        # vision-based
        # gt = self.sd_helper.get_groundtruth(
        #     ["rgb"], self.viewport_window, verify_sensor_init=False, wait_for_sensor_data=0
        # )
        # return self.rgb2grey(gt["rgb"][50:, :50, :3])
        if self._args.algorithm == "PPO":
            return np.concatenate(
                [
                    # self.rgb2grey(gt["rgb"][50:, :50, :3]).reshape(-1),
                    mycobot_joint_position,
                    mycobot_joint_velocity,
                    mycobot_tip_position,
                    goal_world_position,
                    goal_world_position-mycobot_tip_position
                ],
                dtype = np.float32
            )
        elif self._args.algorithm == 'DDPG':
            return OrderedDict(
                [
                    ("observation", observation),
                    ("achieved_goal", mycobot_tip_position),
                    ("desired_goal", goal_world_position),
                ]
            )

    def render(self, mode="human"):
        return

    def close(self):
        self._simulation_app.close()
        return

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        np.random.seed(seed)
        return [seed]

    def set_world_window(self):
        pass
        # import omni.kit
        # from omni.isaac.synthetic_utils import SyntheticDataHelper

        # viewport_handle = omni.kit.viewport_legacy.get_viewport_interface().create_instance()
        # new_viewport_name = omni.kit.viewport_legacy.get_viewport_interface().get_viewport_window_name(viewport_handle)
        # viewport_window = omni.kit.viewport_legacy.get_viewport_interface().get_viewport_window(viewport_handle)
        # viewport_window.set_active_camera("/mycobot/camera/Camera")
        # viewport_window.set_texture_resolution(64, 64)
        # # viewport_window.set_camera_position("/OmniverseKit_Persp",100,100,60,True)
        # # viewport_window.set_camera_target("/OmniverseKit_Persp",0,0,0,True)
        # viewport_window.set_window_pos(1000, 400)
        # viewport_window.set_window_size(420, 420)
        # self.viewport_window = viewport_window
        # self.sd_helper = SyntheticDataHelper()
        # self.sd_helper.initialize(sensor_names=["rgb"], viewport=self.viewport_window)
        # self._my_world.render()
        # self.sd_helper.get_groundtruth(["rgb"], self.viewport_window)
        # return

    def plt_image(self, X):
        plt.imshow(X)
        plt.savefig(fname='viewport2_test.jpg')
    
    def rgb2grey(self, rgb):
        return 0.2989 * rgb[:,:,0] + 0.5870 * rgb[:,:,1] + 0.1140 * rgb[:,:,2]