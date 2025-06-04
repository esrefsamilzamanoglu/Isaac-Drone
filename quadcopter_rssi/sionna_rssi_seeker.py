from __future__ import annotations
"""Quadcopter RSSI environment (Isaac Lab + Sionna‑RT)
=====================================================
* RL + physics logic resides here.
* All UI helpers live in **quadcopter_vis.py** (imported below).
* Headless runs remain possible; Omniverse widgets load only when a viewer exists.
"""

# -----------------------------------------------------------------------------
# Standard / third‑party imports
# -----------------------------------------------------------------------------
import os, sys, math, time
from pathlib import Path
from typing import Optional, Any

import gymnasium as gym
import torch

# -----------------------------------------------------------------------------
# Isaac Lab imports
# -----------------------------------------------------------------------------
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import subtract_frame_transforms
from isaaclab.markers import VisualizationMarkers

from isaaclab_assets import CRAZYFLIE_CFG  # pre‑defined robot
from isaaclab.markers import CUBOID_MARKER_CFG

# -----------------------------------------------------------------------------
# Local helper: import sibling `quadcopter_vis.py`
# -----------------------------------------------------------------------------
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from quadcopter_vis import (
    make_single_sphere_vis,
    make_traj_point_vis,
    QuadcopterEnvWindow,
)
# -----------------------------------------------------------------------------
# Sionna‑RT & Mitsuba setup
# -----------------------------------------------------------------------------

import mitsuba as mi
mi.set_variant("cuda_ad_mono_polarized")

from sionna.rt import load_scene, PlanarArray, Transmitter, Receiver, PathSolver

try:  # Sionna ≤ 0.19.x
    from sionna.rt.radio_materials.itu_material import _ITU_MATERIALS as _MATS
except ImportError:  # Sionna ≥ 1.0
    from sionna.rt.radio_materials import itu as _itu
    _MATS = getattr(_itu, "ITU_MATERIALS", getattr(_itu, "ITU_MATERIALS_PROPERTIES"))

# alias → base material mapping ------------------------------------------------
ITU_ALIASES = {"floorboard": "wood", "ceiling_board": "plasterboard", "plywood": "wood"}

# ============================================================================
# Configuration dataclass
# ============================================================================

@configclass
class QuadcopterRSSIEnvCfg(DirectRLEnvCfg):
    # Isaac Lab basics
    episode_length_s: float = 10.0
    decimation: int = 2
    action_space: int = 4
    state_space: int = 0
    observation_space: int = 10  # 9 low‑dim + 1 RSSI
    debug_vis: bool = True

    # Paths
    _ROOT = Path(__file__).resolve().parent
    usd_enable: bool = True
    usd_path: Path = _ROOT / "assets/scenes/simple_room/simple_room.usd"
    sionna_scene_file: Path = _ROOT / "assets/scenes/simple_room/simple_room.xml"

    # Simulation
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
    )
    terrain: Optional[TerrainImporterCfg] = (
        TerrainImporterCfg(
            prim_path="/World/envs/env_0",
            terrain_type="usd",
            usd_path=str(usd_path),
            collision_group=-1,
            debug_vis=False,
        )
        if usd_enable
        else None
    )

    # Scene replication
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=8, env_spacing=0, replicate_physics=True)

    # Robot params
    robot: ArticulationCfg = CRAZYFLIE_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    thrust_to_weight: float = 1.9
    moment_scale: float = 0.01

    # Reward scales
    lin_vel_reward_scale: float = -0.05
    ang_vel_reward_scale: float = -0.01
    distance_to_goal_reward_scale: float = 50.0
    died_scale: float = -1.0
    found_goal_scale: float = 0.0

    # Sionna RF params
    frequency: float = 2.4e9  # Hz
    tx_power_dbm: float = 9.0
    max_depth: int = 2
    samples_per_tx: int = 10
    rssi_min_dbm: float = -150.0
    rssi_max_dbm: float = 30.0
    rssi_update_interval: int = 1

    # UI class
    ui_window_class_type = QuadcopterEnvWindow

# ============================================================================
# Environment implementation
# ============================================================================

class QuadcopterRSSIEnv(DirectRLEnv):
    """Isaac Lab quadcopter with Sionna‑RT‑based RSSI simulation."""

    cfg: QuadcopterRSSIEnvCfg

    # ----------------------------- init ----------------------------------
    def __init__(self, cfg: QuadcopterRSSIEnvCfg, render_mode: Optional[str] = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.traj_len = 200
        self._traj_buf = torch.zeros((self.num_envs, self.traj_len, 3), device=self.device)
        self._traj_head = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.traj_vis: VisualizationMarkers = make_traj_point_vis(self.traj_len)

        # last RSSI measurement
        self._last_rssi = torch.zeros(self.num_envs, device=self.device)

        # action → force/moment buffers
        flat_dim = gym.spaces.flatdim(self.single_action_space)
        self._actions = torch.zeros(self.num_envs, flat_dim, device=self.device)
        self._thrust  = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._moment  = torch.zeros(self.num_envs, 1, 3, device=self.device)

        # goal/TX positions
        self._desired_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        
        # RSSI buffers
        self._rssi_buf = torch.zeros(self.num_envs, 1, device=self.device)
        self._rssi_counter = 0

        # episodic sums
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "lin_vel",
                "ang_vel",
                "distance_to_goal",
                "found_goal",
                "died",
            ]
        }

        # mass/weight
        self._body_id     = self._robot.find_bodies("body")[0]
        self._robot_mass  = self._robot.root_physx_view.get_masses()[0].sum()
        self._robot_weight = (self._robot_mass * torch.tensor(self.sim.cfg.gravity, device=self.device).norm()).item()

        self._env_origins = (
            self._terrain.env_origins
            if cfg.terrain is not None else
            torch.zeros(self.num_envs, 3, device=self.device)
        )

        # Sionna scene
        self._init_sionna()

        self.set_debug_vis(self.cfg.debug_vis)

    # ------------------------- Sionna setup ------------------------------
    def _init_sionna(self):
        # extend material table
        for alias, base in ITU_ALIASES.items():
            if alias not in _MATS:
                _MATS[alias] = _MATS[base].copy()
            else:
                for rng, props in _MATS[base].items():
                    _MATS[alias].setdefault(rng, props)

        self._sionna_scene = load_scene(str(self.cfg.sionna_scene_file))
        self._sionna_scene.frequency = self.cfg.frequency
        self._sionna_scene.bandwidth = 100e6
        arr = PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="V")
        self._sionna_scene.tx_array = arr
        self._sionna_scene.rx_array = arr

        self._tx_devices: list[Transmitter] = []
        self._rx_devices: list[Receiver] = []
        for i in range(self.num_envs):
            tx = Transmitter(f"tx{i}", [0,0,0], [0,0,0], power_dbm=self.cfg.tx_power_dbm)
            rx = Receiver(f"rx{i}", [0,0,0], [0,0,0])
            self._sionna_scene.add(tx)
            self._sionna_scene.add(rx)
            self._tx_devices.append(tx)
            self._rx_devices.append(rx)

        self._ps = PathSolver()
        self._ps.loop_mode = "symbolic"

    # --------------------- Isaac Lab scene build --------------------------
    def _setup_scene(self):
        super()._setup_scene()
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot

        if self.cfg.usd_enable and self.cfg.terrain is not None:
            self.cfg.terrain.num_envs   = self.scene.cfg.num_envs
            self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
            self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        else:
            self._terrain = type("DummyTerrain", (), {"env_origins": torch.zeros(self.num_envs, 3, device=self.device)})()

        self.scene.clone_environments(copy_from_source=True)
        light_cfg = sim_utils.DomeLightCfg(intensity=8000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg) 
    # --------------------------- actions ---------------------------------
    def _pre_physics_step(self, actions: torch.Tensor):
        self._actions = actions.clamp(-1.0, 1.0)
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

    # ------------------------- observations ------------------------------
    def _get_observations(self):
        # update every N steps
        step_time = time.time()
        if self._rssi_counter % self.cfg.rssi_update_interval == 0:
            desired_local = self._desired_pos_w - self._env_origins
            robot_local   = self._robot.data.root_pos_w[:, :3] - self._env_origins
            rssi_list = []
            #print(f"desired_local: {desired_local}, robot_local: {robot_local} ")
            for i in range(self.num_envs):
                # update positions
                self._tx_devices[i].position = desired_local[i].tolist()
                self._rx_devices[i].position = robot_local[i].tolist()
                # solve paths
                paths = self._ps(
                    scene=self._sionna_scene,
                    max_depth=self.cfg.max_depth,
                    samples_per_src=self.cfg.samples_per_tx,
                    synthetic_array=True,
                    los=True,
                    specular_reflection=True,
                    diffuse_reflection=True,
                    refraction=True,
                    seed=i,
                )
                a_t, _ = paths.cir(normalize_delays=True, out_type="torch")
                abs2 = torch.abs(a_t.to(self.device))**2
                power_paths = abs2[0,0,0,:,:].sum()
                d = torch.linalg.norm(desired_local[i] - robot_local[i]).clamp_min(1e-3)
                lam = 3e8 / self.cfg.frequency
                fspl = (lam / (4 * math.pi * d))**2
                power = torch.maximum(power_paths, fspl)
                power_safe = torch.clamp(power, min=1e-6)
                prx_dbm = self.cfg.tx_power_dbm + 10 * torch.log10(power_safe)

                prx_dbm_cl = prx_dbm.clamp(self.cfg.rssi_min_dbm, self.cfg.rssi_max_dbm)
                scale = self.cfg.rssi_max_dbm - self.cfg.rssi_min_dbm
                rssi_val = (((prx_dbm_cl - self.cfg.rssi_min_dbm) / scale) * 2 - 1).clamp(-1.0,1.0)
                rssi_list.append(rssi_val)
                self._last_rssi[i] = prx_dbm

            self._rssi_buf = torch.stack(rssi_list, dim=0).unsqueeze(-1)
        self._rssi_counter += 1

        desired_pos_b, _ = subtract_frame_transforms(
            self._robot.data.root_state_w[:, :3],
            self._robot.data.root_state_w[:, 3:7],
            self._desired_pos_w,
        )
        obs = torch.cat([
            self._robot.data.root_lin_vel_b,
            self._robot.data.root_ang_vel_b,
            self._robot.data.projected_gravity_b,
            self._rssi_buf,
        ], dim=-1)
        #print(f"step_time: {(time.time()-step_time):.3f}")
        return {"policy": obs}

    # ---------------------------- rewards -------------------------------
    def _get_rewards(self):
        lin_vel = torch.sum(torch.square(self._robot.data.root_lin_vel_b), dim=1)
        ang_vel = torch.sum(torch.square(self._robot.data.root_ang_vel_b), dim=1)

        # --- env-yerel vektörler ---
        desired_pos_local = self._desired_pos_w - self._env_origins
        robot_pos_local   = self._robot.data.root_pos_w - self._env_origins
        distance_to_goal  = torch.linalg.norm(desired_pos_local - robot_pos_local, dim=1)

        # debug
        #print("distance_to_goal:", distance_to_goal)

        distance_to_goal_mapped = 1 - torch.tanh(distance_to_goal / 0.8)
        found_goal = (distance_to_goal < 0.01).float()
        died = torch.logical_or(self._robot.data.root_pos_w[:, 2] < 0.1,
                                self._robot.data.root_pos_w[:, 2] > 2.0)

        rewards = {
            "lin_vel": lin_vel * self.cfg.lin_vel_reward_scale * self.step_dt,
            "ang_vel": ang_vel * self.cfg.ang_vel_reward_scale * self.step_dt,
            "distance_to_goal": distance_to_goal_mapped * self.cfg.distance_to_goal_reward_scale * self.step_dt,
            "found_goal": found_goal * self.cfg.found_goal_scale * self.step_dt,
            "died": died * self.cfg.died_scale,
        }
        reward_total = torch.sum(torch.stack(list(rewards.values())), dim=0)

        # episodic log
        for k, v in rewards.items():
            self._episode_sums[k] += v
        return reward_total

    # ----------------------------- dones -------------------------------
    def _get_dones(self):
        timeout = self.episode_length_buf >= self.max_episode_length-1
        died = torch.logical_or(self._robot.data.root_pos_w[:,2] < 0.1,
                                self._robot.data.root_pos_w[:,2] > 2.0)
        #if died: print("died")
        #if timeout: print("time out")
        return died, timeout

    # ------------------------------ reset ------------------------------
    def _reset_idx(self, env_ids: Optional[torch.Tensor]):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES
        self._traj_buf[env_ids] = 0.0
        self._traj_head[env_ids] = 0

        final_dist = torch.linalg.norm(
            self._desired_pos_w[env_ids] - self._robot.data.root_pos_w[env_ids], dim=1
        ).mean()
        dists = torch.norm(
            self._desired_pos_w[env_ids] - self._robot.data.root_pos_w[env_ids], dim=1
        )
        for idx, d in zip(env_ids.tolist(), dists.tolist()):
            print(f"Env {idx} final_dist: {d:.4f}")

        # new random target
        local_goal = torch.zeros(3, device=self.device)
        local_goal[:2] = torch.rand(2, device=self.device) * 4 - 2
        local_goal[2]  = torch.rand(1, device=self.device) * 1 + 0.5
        self._desired_pos_w[:] = self._env_origins + local_goal
        # update each transmitter
        for i, tx in enumerate(self._tx_devices):
            tx.position = local_goal.tolist()

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

        # robot default state
        jp = self._robot.data.default_joint_pos[env_ids]
        jv = self._robot.data.default_joint_vel[env_ids]
        root = self._robot.data.default_root_state[env_ids]
        root[:, :3] += self._terrain.env_origins[env_ids]

        self._robot.write_root_pose_to_sim(root[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(root[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(jp, jv, None, env_ids)

        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)
        #if len(env_ids) == self.num_envs:
        #    self.episode_length_buf = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))

    def _set_debug_vis_impl(self, debug_vis: bool):
        if not getattr(self, "_window", None):
            return
        if debug_vis:
            self._window.set_visibility(True)
            self.traj_vis.set_visibility(True)

            # create concentric goal spheres once -----------------
            if not hasattr(self, "goal_pos_visualizer"):
                layers, min_r, max_r = 5, 0.02, 0.07
                min_opa, color = 0.1, (1.0, 0.0, 0.0)
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
        if not getattr(self, "_window", None):
            return
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

# ------------------------------ standalone test ------------------------------
if __name__ == "__main__":
    cfg = QuadcopterRSSIEnvCfg()
    env = QuadcopterRSSIEnv(cfg, render_mode="human")
    obs, _ = env.reset()
    done = False
    while not done:
        act = torch.zeros(env.single_action_space.shape, device=env.device)
        obs, _, term, trunc, _ = env.step(act)
        done = bool(term or trunc)
    env.close()
