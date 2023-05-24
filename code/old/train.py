from env import MyCobotEnv
from stable_baselines3.common.callbacks import CheckpointCallback
import torch as th
import argparse
import time
import os

# import warnings
# warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()
parser.add_argument("--test", default=False, action="store_true", help="Run in test mode")
parser.add_argument("--headless", default=True, type=bool, help="Whether not to monitor the training process visually")
parser.add_argument("--joint_control", default='position', type=str, help="efforts or position")
parser.add_argument("--algorithm", default="PPO", type=str, choices=["PPO", "DDPG", "Her", "DQN", "TD3", "A2C"])
parser.add_argument("--use_her",  default=True, type=bool, help="Whether use Hindsight Experience Replay")
args, unknown = parser.parse_known_args()

total_timesteps = 2000000

if args.test:
    total_timesteps = 10000

timestr = time.strftime("%Y%m%d_%H%M%S")
log_dir = "./mycobot/cnn_policy"
file_name = "{}_{}_mycobot_policy_{}_{}k".format(
    args.algorithm, timestr, args.joint_control, int(total_timesteps/1000))

# set headless to false to visualize training
my_env = MyCobotEnv(args=args)

# improve here
policy_kwargs = dict(activation_fn=th.nn.ReLU, net_arch=[16, dict(vf=[128, 128, 128], pi=[128, 128, 128])])

checkpoint_callback = CheckpointCallback(save_freq=100000, save_path=log_dir, name_prefix=file_name)

if args.algorithm == 'PPO':
    from stable_baselines3 import PPO
    from stable_baselines3.ppo import MlpPolicy, CnnPolicy
    # model = PPO(
    #     CnnPolicy,
    #     my_env,
    #     policy_kwargs=policy_kwargs,
    #     verbose=1,
    #     n_steps=10000,
    #     batch_size=1000,
    #     learning_rate=0.00025,
    #     gamma=0.9995,
    #     device="cuda",
    #     ent_coef=0,
    #     vf_coef=0.5,
    #     max_grad_norm=10,
    #     tensorboard_log=log_dir,
    # )

    model = PPO(
        MlpPolicy,
        my_env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        n_steps=2560,
        batch_size=64,
        learning_rate=0.000125,
        gamma=0.9,
        ent_coef=7.5e-08,
        clip_range=0.3,
        n_epochs=5,
        gae_lambda=1.0,
        max_grad_norm=0.9,
        vf_coef=0.95,
        device="cuda",
        tensorboard_log=log_dir,
    )

elif args.algorithm == 'DDPG':
    pass
#     from stable_baselines3 import DDPG
#     from stable_baselines3.ddpg import MlpPolicy, CnnPolicy, MultiInputPolicy
#     replay_buffer_class, goal_selection_strategy = None, None
#     if args.use_her:
#         from stable_baselines3 import HerReplayBuffer
#         from stable_baselines3.her.goal_selection_strategy import GoalSelectionStrategy
#         from stable_baselines3.common.envs import BitFlippingEnv
#         replay_buffer_class = HerReplayBuffer
#         goal_selection_strategy = GoalSelectionStrategy.FUTURE

#     model = DDPG(
#         "MultiInputPolicy",
#         my_env,
#         replay_buffer_class=replay_buffer_class,
#         replay_buffer_kwargs=dict(
#             n_sampled_goal=4,
#             goal_selection_strategy=goal_selection_strategy,
#             online_sampling=True,
#             max_episode_length=512,
#         ),
#         # policy_kwargs=policy_kwargs,
#         verbose=1,
#         device='cuda'
# )

elif args.algorithm == 'DQN':
    from stable_baselines3 import DQN
    from stable_baselines3.dqn import CnnPolicy, MlpPolicy, MultiInputPolicy
elif args.algorithm == 'SAC':
    from stable_baselines3 import SAC
    from stable_baselines3.sac import CnnPolicy, MlpPolicy, MultiInputPolicy
elif args.algorithm == 'TD3':
    from stable_baselines3 import TD3
    from stable_baselines3.td3 import CnnPolicy, MlpPolicy, MultiInputPolicy
elif args.algorithm == 'A2C':
    from stable_baselines3 import A2C
    from stable_baselines3.a2c import CnnPolicy, MlpPolicy, MultiInputPolicy
    

model.learn(total_timesteps=total_timesteps, callback=[checkpoint_callback])

model.save(os.path.join(log_dir, file_name))

my_env.close()