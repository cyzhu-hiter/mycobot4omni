'''
This file is an upgrade for RL multi-instance training in Omniverse environment.
mt stands for multi-instance training
'''
import __init__

from vec_env_rlgames import VecEnvRLGames
from stable_baselines3.common.callbacks import CheckpointCallback
import torch
import numpy as np
import argparse
import time
import os

import warnings
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()
parser.add_argument("--test", default=False, action="store_true", help="Run in test mode")
parser.add_argument("--headless", default=False, type=bool, help="Whether not to monitor the training process visually")
parser.add_argument("--joint_control", default='position', type=str, help="efforts or position")
parser.add_argument("--algorithm", default="PPO", type=str, choices=["PPO", "DDPG", "Her", "DQN", "TD3", "A2C"])
parser.add_argument("--name", default="mycobot", type=str)
args, unknown = parser.parse_known_args()

total_timesteps = 200000

if args.test is True:
    total_timesteps = 10000

timestr = time.strftime("%Y%m%d_%H%M%S")
log_dir = "./mycobot/cnn_policy"
file_name = "{}_mycobot_policy_{}dof_{}_{}k".format(timestr,6,'efforts',int(total_timesteps/1000))

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
sim_params = {"use_gpu_pipeline":True}

# set headless to false to visualize training
rl_env = VecEnvRLGames(headless=args.headless, sim_device=0)

from env_mt import MyCobotEnv
my_env = MyCobotEnv(args=args, env=rl_env)
rl_env.set_task(task = my_env, backend='torch', sim_params=sim_params)

while rl_env._simulation_app.is_running():
    if rl_env._world.is_playing():
        if rl_env._world.current_time_step_index == 0:
            rl_env._world.reset(soft=True)
        # actions = torch.tensor(
        #     np.array([rl_env.action_space.sample() for _ in range(rl_env.num_envs)]), device=device
        # )
        action = torch.tensor(np.ones(7), dtype=torch.float32, device=device)
        rl_env._task._mycobots.set_joint_positions(action, indices=torch.tensor(np.array([0,1])).to('cuda:0'))
        # rl_env._task.pre_physics_step(actions)
        rl_env._world.step(render=not args.headless)
        rl_env.sim_frame_count += 1
        # rl_env._task.post_physics_step()
    else:
        rl_env._world.step(render=not args.headless)

rl_env._simulation_app.close()
