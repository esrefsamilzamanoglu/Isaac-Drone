import argparse
import os
import time
import torch
import gymnasium as gym

from isaaclab.app import AppLauncher
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner

import cli_args
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg

# ======================== ARG PARSING ========================
parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--device", type=str, default="cuda:0")
parser.add_argument("--test_trials", type=int, default=10000)
parser.add_argument("--success_thresh", type=float, default=0.01)
parser.add_argument("--approx_thresh", type=float, default=0.30)
AppLauncher.add_app_launcher_args(parser)
cli_args.add_rsl_rl_args(parser)
args = parser.parse_args()
args.enable_cameras = False

# ======================== SIM SETUP ========================
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
agent_cfg = cli_args.parse_rsl_rl_cfg(args.task, args)
resume_path = retrieve_file_path(args.checkpoint)

env = gym.make(args.task, cfg=env_cfg)
if isinstance(env.unwrapped, DirectMARLEnv):
    env = multi_agent_to_single_agent(env)
env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
ppo_runner.load(resume_path)
policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

# ======================== TEST LOOP ========================
success, approx, died = 0, 0, 0
distances = []

for i in range(args.test_trials):
    obs, _ = env.reset()
    died_flag = False
    min_dist = float("inf")

    for _ in range(env.unwrapped.max_episode_length):
        with torch.no_grad():
            actions = policy(obs)
            obs, _, done, _ = env.step(actions)

        robot_pos = env.unwrapped._robot.data.root_pos_w[0, :3]
        goal_pos = env.unwrapped._desired_pos_w[0, :3]
        dist = torch.linalg.norm(goal_pos - robot_pos).item()
        min_dist = min(min_dist, dist)

        if done.any().item():
            if env.unwrapped.reset_terminated[0].item():
                died_flag = True
            break

    distances.append(min_dist)
    if min_dist < args.success_thresh:
        success += 1
    if min_dist < args.approx_thresh:
        approx += 1
    if died_flag:
        died += 1

# ======================== METRİKLER ========================
print("==================== TEST SONUCU ====================")
print(f"Toplam deneme:             {args.test_trials}")
print(f"Final distance ort.:       {sum(distances)/len(distances):.4f} m")
print(f"Success rate (<{args.success_thresh}m):    {success / args.test_trials:.2%}")
print(f"Approx. rate (<{args.approx_thresh}m):     {approx / args.test_trials:.2%}")
print(f"Died rate:                 {died / args.test_trials:.2%}")
print("====================================================")

# ======================== KAPAT ========================
env.close()
simulation_app.close()
