import time
import argparse
import numpy as np

from env import MyCobotEnv
from stable_baselines3 import PPO

parser = argparse.ArgumentParser()
parser.add_argument("--enable_livestream", default=True, type=bool)
parser.add_argument("--enable_viewport", default=False, type=bool)
parser.add_argument("--model_check", default=True, type=bool)
parser.add_argument("--random_policy", default=False, type=bool,
                    help="Whether using random policy to control the robot")
parser.add_argument("--iter", default=100, type=int,
                    help="The number of iterations will be executed.")
parser.add_argument("--headless", default=True, type=bool,
                    help="Whether not to monitor the training process visually")
parser.add_argument("--joint_control", default='position', type=str,
                    help="efforts or position")
parser.add_argument("--policy_path", type=str, default=None,
                    help="The path to stored trained policy.")
parser.add_argument("--algorithm", default="PPO", type=str,
                    choices=["PPO", "DDPG", "Her", "DQN", "TD3", "A2C"])

args, unknown = parser.parse_known_args()
# args.policy_path = 'mycobot/cnn_policy/PPO_20230307_221328_mycobot_policy_6dof_position_2000k_400000_steps'

my_env = MyCobotEnv(seed=42, args=args)

model = PPO.load(args.policy_path) if args.policy_path is not None else False

while my_env._simulation_app.is_running():
    if args.model_check:
        if my_env.my_world.current_time_step_index == 0:
            breakpoint
            # my_env.my_world.reset()
            # my_env.my_controller.reset() # ?
        my_env.render()
    # obs = my_env.reset()
    # my_env.render()
    # done = False 
    # # for i in range(60):
    # #     actions = np.zeros((5,))
    # #     obs, reward, done, info = my_env.step(actions)
    # #     # time.sleep(1/60)
    # #     my_env.render()
    # while not done:
    #     if args.random_policy:
    #         actions = my_env.action_space.sample() # random policy
    #         actions = np.concatenate([np.zeros((3,)), np.ones((1,))*np.pi/2, -np.ones((1,))*np.pi/2, np.zeros((1,))])
    #         # actions = np.concatenate([np.zeros((6,)), np.ones((1,))])*-1
    #     else:
    #         actions, _ = model.predict(observation=obs, deterministic=True) # trained policy

    #     obs, reward, done, info = my_env.step(actions)
    #     my_env.render()
        


my_env.close()
