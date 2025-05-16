from __future__ import annotations

"""Quadcopter‑RSSI environment – *logic only* (RL + physics).

All Omniverse UI/görselleştirme kodu **quadopter_vis.py** dosyasına taşındı.
Bu dosya artık yalnızca ortam mantığını ve ödülleri içerir. Başsız (headless)
çalıştırmalarda da güvenle import edilebilir.
"""

from typing import Any

import gymnasium as gym
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import subtract_frame_transforms

# Visualization helpers & custom window ----------------------------

import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from quadcopter_vis import (
    make_single_sphere_vis,
    make_traj_point_vis,
    QuadcopterEnvWindow,
)

# Pre‑defined robot cfg -------------------------------------------
from isaaclab_assets import CRAZYFLIE_CFG  # isort: skip
from isaaclab.markers import VisualizationMarkers
from isaaclab.sim import RenderCfg


# =================================================================
#  Configuration
# =================================================================

@configclass
class QuadcopterRSSIEnvCfg(DirectRLEnvCfg):
    # ---------------- environment basics ----------------
    episode_length_s: float = 10.0
    decimation: int = 2
    action_space: int = 4
    state_space: int = 0
    debug_vis: bool = True
    observation_space: int = 10

    # ---------------- RSSI model ------------------------
    rssi_A1m_dbm: float = -36.0  # mean RSSI @1 m
    rssi_path_exp: float = 2.2   # path‑loss exponent n
    rssi_min_dbm: float = -150.0
    rssi_max_dbm: float = 30.0

    # Attach custom Omniverse window
    ui_window_class_type = QuadcopterEnvWindow

    # ---------------- simulation ------------------------
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 100,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        render=RenderCfg(enable_translucency=True),
    )

    terrain: TerrainImporterCfg = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        debug_vis=False,
    )

    # ---------------- scene -----------------------------
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096, env_spacing=2.5, replicate_physics=True
    )

    # ---------------- robot -----------------------------
    robot: ArticulationCfg = CRAZYFLIE_CFG.replace(
        prim_path="/World/envs/env_.*/Robot"
    )
    thrust_to_weight: float = 1.9
    moment_scale: float = 0.01

    # ---------------- rewards ---------------------------
    lin_vel_reward_scale: float = -0.05
    ang_vel_reward_scale: float = -0.01
    distance_to_goal_reward_scale: float = 50.0
    died_scale: float = -1.0


# =================================================================
#  Environment implementation
# =================================================================


class QuadcopterRSSIEnv(DirectRLEnv):
    """Quadcopter hovering / goal‑seeking task with RSSI observation."""

    cfg: QuadcopterRSSIEnvCfg

    # -----------------------------------------------------------------
    # Init & buffers
    # -----------------------------------------------------------------

    def __init__(self, cfg: QuadcopterRSSIEnvCfg, render_mode: str | None = None, **kwargs: Any):
        super().__init__(cfg, render_mode, **kwargs)

        # trajectory viz ------------------------------------------------
        self.traj_len = 200
        self._traj_buf = torch.zeros((self.num_envs, self.traj_len, 3), device=self.device)
        self._traj_head = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.traj_vis: VisualizationMarkers = make_traj_point_vis(self.traj_len)

        # last RSSI measurement
        self._last_rssi = torch.zeros(self.num_envs, device=self.device)

        # action buffers -------------------------------------------------
        flat_dim = gym.spaces.flatdim(self.single_action_space)
        self._actions = torch.zeros(self.num_envs, flat_dim, device=self.device)
        self._thrust = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._moment = torch.zeros(self.num_envs, 1, 3, device=self.device)

        # goal -----------------------------------------------------------
        self._desired_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        
        # RSSI buffers
        self._rssi_buf = torch.zeros(self.num_envs, 1, device=self.device)
        self._rssi_counter = 0

        # episodic logs --------------------------------------------------
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "lin_vel",
                "ang_vel",
                "distance_to_goal",
                "died",
            ]
        }

        # body references -----------------------------------------------
        self._body_id = self._robot.find_bodies("body")[0]
        self._robot_mass = self._robot.root_physx_view.get_masses()[0].sum()
        self._gravity_magnitude = torch.tensor(self.sim.cfg.gravity, device=self.device).norm()
        self._robot_weight = (self._robot_mass * self._gravity_magnitude).item()

        # optional debug vis handle (set in _set_debug_vis_impl)
        self.set_debug_vis(self.cfg.debug_vis)

    # -----------------------------------------------------------------
    # Scene assembly
    # -----------------------------------------------------------------

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot

        # terrain --------------------------------------------------
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        # replicate envs ------------------------------------------
        self.scene.clone_environments(copy_from_source=False)

        # lighting -------------------------------------------------
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    # -----------------------------------------------------------------
    # Simulation step helpers
    # -----------------------------------------------------------------

    def _pre_physics_step(self, actions: torch.Tensor):
        self._actions = actions.clamp(-1.0, 1.0)
        # thrust: map [-1,1] → [0, 2 * weight * factor]
        self._thrust[:, 0, 2] = (
            self.cfg.thrust_to_weight * self._robot_weight * (self._actions[:, 0] + 1.0) / 2.0
        )
        self._moment[:, 0, :] = self.cfg.moment_scale * self._actions[:, 1:]

        # store trajectory point for viz
        idx = self._traj_head
        self._traj_buf[torch.arange(self.num_envs), idx] = self._robot.data.root_pos_w
        self._traj_head = (idx + 1) % self.traj_len

    def _apply_action(self):
        self._robot.set_external_force_and_torque(self._thrust, self._moment, body_ids=self._body_id)

    # -----------------------------------------------------------------
    # Observations / rewards / termination
    # -----------------------------------------------------------------

    def _get_observations(self) -> dict[str, torch.Tensor]:
        # desired position in body frame
        desired_pos_b, _ = subtract_frame_transforms(
            self._robot.data.root_state_w[:, :3],
            self._robot.data.root_state_w[:, 3:7],
            self._desired_pos_w,
        )
        dist = torch.linalg.norm(desired_pos_b, dim=1, keepdim=True).clamp_min(1e-3)

        # RSSI path‑loss model ------------------------------------
        rssi_dbm = self.cfg.rssi_A1m_dbm - 10.0 * self.cfg.rssi_path_exp * torch.log10(dist)
        rssi_01 = (rssi_dbm - self.cfg.rssi_min_dbm) / (
            self.cfg.rssi_max_dbm - self.cfg.rssi_min_dbm
        )
        rssi = torch.clamp(rssi_01 * 2.0 - 1.0, -1.0, 1.0)
        self._last_rssi = rssi_dbm.flatten()

        obs = torch.cat(
            [
                self._robot.data.root_lin_vel_b,
                self._robot.data.root_ang_vel_b,
                self._robot.data.projected_gravity_b,
                rssi,
            ],
            dim=-1,
        )
        return {"policy": obs}

    # -------------------------------------------------------------

    def _get_rewards(self) -> torch.Tensor:
        lin_vel = torch.sum(torch.square(self._robot.data.root_lin_vel_b), dim=1)
        ang_vel = torch.sum(torch.square(self._robot.data.root_ang_vel_b), dim=1)
        distance_to_goal = torch.linalg.norm(self._desired_pos_w - self._robot.data.root_pos_w, dim=1)
        distance_to_goal_mapped = 1 - torch.tanh(distance_to_goal / 0.8)
        died = torch.logical_or(self._robot.data.root_pos_w[:, 2] < 0.1, self._robot.data.root_pos_w[:, 2] > 2.0)
        rewards = {
            "lin_vel": lin_vel * self.cfg.lin_vel_reward_scale * self.step_dt,
            "ang_vel": ang_vel * self.cfg.ang_vel_reward_scale * self.step_dt,
            "distance_to_goal": distance_to_goal_mapped * self.cfg.distance_to_goal_reward_scale * self.step_dt,
            "died": died * self.cfg.died_scale
        }
        reward_total = torch.sum(torch.stack(list(rewards.values())), dim=0)

        # log episodic sums
        for key, value in rewards.items():
            self._episode_sums[key] += value
        return reward_total

    # -------------------------------------------------------------

    def _get_dones(self):
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        died = torch.logical_or(
            self._robot.data.root_pos_w[:, 2] < 0.1, self._robot.data.root_pos_w[:, 2] > 2.0
        )
        return died, time_out

    # -----------------------------------------------------------------
    # Reset / bookkeeping
    # -----------------------------------------------------------------

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES

        # trajectory buffer reset ---------------------------------
        self._traj_buf[env_ids] = 0.0
        self._traj_head[env_ids] = 0

        # log final distance to goal ------------------------------
        final_dist = torch.linalg.norm(
            self._desired_pos_w[env_ids] - self._robot.data.root_pos_w[env_ids], dim=1
        ).mean()

        extras: dict[str, Any] = {}
        for key in self._episode_sums.keys():
            episodic_avg = torch.mean(self._episode_sums[key][env_ids])
            extras[f"Episode_Reward/{key}"] = episodic_avg / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0

        extras.update(
            {
                "Episode_Termination/died": torch.count_nonzero(self.reset_terminated[env_ids]).item(),
                "Episode_Termination/time_out": torch.count_nonzero(self.reset_time_outs[env_ids]).item(),
                "Metrics/final_distance_to_goal": final_dist.item(),
            }
        )
        self.extras["log"] = extras

        # ---- physics state reset --------------------------------
        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)

        if len(env_ids) == self.num_envs:
            # jitter/reset offset to avoid sync spikes
            self.episode_length_buf = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))

        self._actions[env_ids] = 0.0

        # new goal -------------------------------------------------
        self._desired_pos_w[env_ids, :2] = torch.zeros_like(self._desired_pos_w[env_ids, :2]).uniform_(-2.0, 2.0)
        self._desired_pos_w[env_ids, :2] += self._terrain.env_origins[env_ids, :2]
        self._desired_pos_w[env_ids, 2] = torch.zeros_like(self._desired_pos_w[env_ids, 2]).uniform_(0.5, 1.5)

        # robot start pose ----------------------------------------
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        root_state = self._robot.data.default_root_state[env_ids]
        root_state[:, :3] += self._terrain.env_origins[env_ids]

        self._robot.write_root_pose_to_sim(root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

    # -----------------------------------------------------------------
    # Debug visualisation hooks
    # -----------------------------------------------------------------

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            self._window.set_visibility(True)
            self.traj_vis.set_visibility(True)

            # create concentric goal spheres once -----------------
            if not hasattr(self, "goal_pos_visualizer"):
                layers, min_r, max_r = 12, 0.01, 0.1
                min_opa, color = 0.001, (1.0, 0.0, 0.0)
                self.goal_pos_visualizer: list[VisualizationMarkers] = []
                for i in range(layers):
                    t = i / (layers - 1) if layers > 1 else 0.0
                    r = min_r + (max_r - min_r) * t
                    op = 1.0 - (1.0 - min_opa) * t
                    vis = make_single_sphere_vis(
                        radius=r,
                        color=color,
                        opacity=op,
                        prim_path=f"/Visuals/GoalSphere/layer_{i}",
                    )
                    self.goal_pos_visualizer.append(vis)
            for vis in self.goal_pos_visualizer:
                vis.set_visibility(True)
        else:
            self._window.set_visibility(False)
            self.traj_vis.set_visibility(False)
            if hasattr(self, "goal_pos_visualizer"):
                for vis in self.goal_pos_visualizer:
                    vis.set_visibility(False)

    # -------------------------------------------------------------

    def _debug_vis_callback(self, event):
        # goal marker update
        for vis in self.goal_pos_visualizer:
            vis.visualize(self._desired_pos_w)

        # breadcrumb trajectory
        all_pts = self._traj_buf.reshape(-1, 3)
        self.traj_vis.visualize(all_pts)

        # live RSSI in window (if present)
        val = self._last_rssi[0].item() if self._last_rssi.numel() > 1 else self._last_rssi.item()
        if hasattr(self, "_window") and self._window is not None:
            self._window.set_rssi(val)
