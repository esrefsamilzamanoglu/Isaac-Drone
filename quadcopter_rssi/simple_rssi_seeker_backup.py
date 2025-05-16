from __future__ import annotations

import gymnasium as gym
import torch
import os
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.envs.ui import BaseEnvWindow
from isaaclab.markers import VisualizationMarkers
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import subtract_frame_transforms
from isaaclab.envs.ui.viewport_camera_controller import ViewportCameraController  # kamera controller
from isaaclab.envs import ViewerCfg                                     # config sınıfı

##
# Pre-defined configs
##
from isaaclab_assets import CRAZYFLIE_CFG  # isort: skip
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.sim import SphereCfg, PreviewSurfaceCfg, RenderCfg

from pxr import Gf       
import omni.ui as ui 
import omni.kit.viewport.utility as vp_util


def make_single_sphere_vis(radius: float,
                           color  =(1.0,0.0,0.0),
                           opacity=0.3,
                           prim_path="/Visuals/GoalSphere") -> VisualizationMarkers:
    cfg = VisualizationMarkersCfg(
        prim_path = prim_path,
        markers   = {
            "sphere": SphereCfg(
                radius=radius,
                visual_material=PreviewSurfaceCfg(
                    diffuse_color=color,
                    opacity=opacity
                )
            )
        }
    )
    return VisualizationMarkers(cfg)

def make_traj_point_vis(
    max_points: int,
    radius: float = 0.003,
    color: tuple[float, float, float] = (0.0, 1.0, 0.0),
    prim_path: str = "/Visuals/TrajPoints",
) -> VisualizationMarkers:
    markers = {
        f"pt_{i:04d}": SphereCfg(
            radius=radius,
            visual_material=PreviewSurfaceCfg(diffuse_color=color, opacity=1.0),
        )
        for i in range(max_points)
    }
    cfg = VisualizationMarkersCfg(prim_path=prim_path, markers=markers)
    return VisualizationMarkers(cfg)

class QuadcopterEnvWindow(BaseEnvWindow):
    """Window manager for the Quadcopter environment."""

    def __init__(self, env: QuadcopterRSSIEnv, window_name: str = "IsaacLab"):
        """Initialize the window.

        Args:
            env: The environment object.
            window_name: The name of the window. Defaults to "IsaacLab".
        """
        # initialize base window
        super().__init__(env, window_name)

        viewer_cfg = ViewerCfg(
            eye=(1.5, 1.5, 1.5),            # kamera pozisyonu
            lookat=(0.0, 0.0, 0.0),         # bakılan nokta
            cam_prim_path="/OmniverseKit_Persp",  # default kamera prim
            resolution=(1280, 720),         # pencere çözünürlüğü
            origin_type="asset_root",              # “world” | “env” | “asset_root” | “asset_body”
            env_index=0,                    # hangi ortamı takip edecek
            asset_name="robot",             # asset_root takibi için robot kök adı
            body_name=None                  # asset_body takibi yok
        )
        # ❸ Controller’ı oluşturun:
        self.camera_ctrl = ViewportCameraController(self.env, viewer_cfg)

        # add custom UI elements
        with self.ui_window_elements["main_vstack"]:
            with self.ui_window_elements["debug_frame"]:
                with self.ui_window_elements["debug_vstack"]:
                    # add command manager visualization
                    self._create_debug_vis_ui_element("targets", self.env)
                    



@configclass
class QuadcopterRSSIEnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 15.0
    decimation = 2
    action_space = 4
    state_space = 0
    debug_vis = True
    observation_space = 10          # 12 → 10  (3 yerine 1 değer)
    # --- RSSI modeli ---
    rssi_A1m_dbm: float = -36.0     # 1 m’de ortalama RSSI
    rssi_path_exp: float = 2.2      # n (path-loss exponent)
    rssi_min_dbm: float = -150.0
    rssi_max_dbm: float = 30.0
    
    ui_window_class_type = QuadcopterEnvWindow
    
    # simulation
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
        render = RenderCfg(
            enable_translucency = True,
        ),
    )
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        debug_vis=False,
    )
    

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=2.5, replicate_physics=True)

    # robot
    robot: ArticulationCfg = CRAZYFLIE_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    thrust_to_weight = 1.9
    moment_scale = 0.01

    # reward scales
    lin_vel_reward_scale = -0.05
    ang_vel_reward_scale = -0.01
    distance_to_goal_reward_scale = 50.0
    died_scale = -1.0

class QuadcopterRSSIEnv(DirectRLEnv):
    cfg: QuadcopterRSSIEnvCfg

    
    def __init__(self, cfg: QuadcopterRSSIEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        
        self.traj_len = 200
        self._traj_buf = torch.zeros((self.num_envs, self.traj_len, 3), device=self.device)
        self._traj_head = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.traj_vis = make_traj_point_vis(self.traj_len)  

        self._vp_window = vp_util.get_active_viewport_window()
        with self._vp_window.get_frame("Overlay"):
            self._rssi_label = ui.Label(
                "-150 dBm",
                alignment = ui.Alignment.CENTER,
                size      = 18,
                style     = {"color": 0xffffffff},
            )

        self._last_rssi = torch.zeros(self.num_envs, device=self.device)

        # Total thrust and moment applied to the base of the quadcopter
        self._actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)
        self._thrust = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._moment = torch.zeros(self.num_envs, 1, 3, device=self.device)
        # Goal position
        self._desired_pos_w = torch.zeros(self.num_envs, 3, device=self.device)

        # Logging
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "lin_vel",
                "ang_vel",
                "distance_to_goal",
                "died"
            ]
        }
        # Get specific body indices
        self._body_id = self._robot.find_bodies("body")[0]
        self._robot_mass = self._robot.root_physx_view.get_masses()[0].sum()
        self._gravity_magnitude = torch.tensor(self.sim.cfg.gravity, device=self.device).norm()
        self._robot_weight = (self._robot_mass * self._gravity_magnitude).item()

        # add handle for debug visualization (this is set to a valid handle inside set_debug_vis)
        self.set_debug_vis(self.cfg.debug_vis)

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor):
        self._actions = actions.clone().clamp(-1.0, 1.0)
        self._thrust[:, 0, 2] = self.cfg.thrust_to_weight * self._robot_weight * (self._actions[:, 0] + 1.0) / 2.0
        self._moment[:, 0, :] = self.cfg.moment_scale * self._actions[:, 1:]
        idx = self._traj_head
        self._traj_buf[torch.arange(self.num_envs), idx] = self._robot.data.root_pos_w
        self._traj_head = (idx + 1) % self.traj_len

    def _apply_action(self):
        self._robot.set_external_force_and_torque(self._thrust, self._moment, body_ids=self._body_id)

    def _get_observations(self) -> dict:
        desired_pos_b, _ = subtract_frame_transforms(
            self._robot.data.root_state_w[:, :3],
            self._robot.data.root_state_w[:, 3:7],
            self._desired_pos_w,
        )

        dist = torch.linalg.norm(desired_pos_b, dim=1, keepdim=True).clamp_min(1e-3)

        rssi_dbm = (
            self.cfg.rssi_A1m_dbm
            - 10.0 * self.cfg.rssi_path_exp * torch.log10(dist)
        )
        #print(rssi_dbm)
        
        rssi_01 = (rssi_dbm - self.cfg.rssi_min_dbm) / (self.cfg.rssi_max_dbm - self.cfg.rssi_min_dbm)
        rssi = rssi_01 * 2.0 - 1.0
        rssi = torch.clamp(rssi, -1.0, 1.0)
        obs = torch.cat(
            [
                self._robot.data.root_lin_vel_b,      
                self._robot.data.root_ang_vel_b,      
                self._robot.data.projected_gravity_b, 
                rssi,                                 
            ],
            dim=-1,
        )
        self._last_rssi = rssi_dbm.flatten()
        return {"policy": obs}


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
        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
        # Logging
        for key, value in rewards.items():
            self._episode_sums[key] += value
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        died = torch.logical_or(self._robot.data.root_pos_w[:, 2] < 0.1, self._robot.data.root_pos_w[:, 2] > 2.0)
        #if died: print("died")
        #if time_out: print("time out")
        return died, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES

        self._traj_buf[env_ids] = 0.0
        self._traj_head[env_ids] = 0

        # Logging
        final_distance_to_goal = torch.linalg.norm(
            self._desired_pos_w[env_ids] - self._robot.data.root_pos_w[env_ids], dim=1
        ).mean()
        extras = dict()
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            extras["Episode_Reward/" + key] = episodic_sum_avg / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0
        self.extras["log"] = dict()
        self.extras["log"].update(extras)
        extras = dict()
        extras["Episode_Termination/died"] = torch.count_nonzero(self.reset_terminated[env_ids]).item()
        extras["Episode_Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids]).item()
        extras["Metrics/final_distance_to_goal"] = final_distance_to_goal.item()
        self.extras["log"].update(extras)

        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)
        if len(env_ids) == self.num_envs:
            # Spread out the resets to avoid spikes in training when many environments reset at a similar time
            self.episode_length_buf = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))

        self._actions[env_ids] = 0.0
        # Sample new commands
        self._desired_pos_w[env_ids, :2] = torch.zeros_like(self._desired_pos_w[env_ids, :2]).uniform_(-2.0, 2.0)
        self._desired_pos_w[env_ids, :2] += self._terrain.env_origins[env_ids, :2]
        self._desired_pos_w[env_ids, 2] = torch.zeros_like(self._desired_pos_w[env_ids, 2]).uniform_(0.5, 1.5)
        # Reset robot state
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        default_root_state = self._robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]
        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

    def _set_debug_vis_impl(self, debug_vis: bool):
        """Env bazlı debug görselleştirme (hedef işaretçisi)."""
        if debug_vis:
            self._rssi_label.visible = True
            self.traj_vis.set_visibility(True)
            if not hasattr(self, "goal_pos_visualizer"):   # << isim KORUNDU
                # Parametrik katmanlar
                LAYERS        = 12
                MIN_R, MAX_R  = 0.01, 0.02
                MIN_OPA       = 0.5
                color         = (1.0, 0.0, 0.0)

                self.goal_pos_visualizer: list[VisualizationMarkers] = []
                for i in range(LAYERS):
                    t  = i/(LAYERS-1) if LAYERS>1 else 0.0
                    r  = MIN_R + (MAX_R-MIN_R)*t
                    op = 1.0 - (1.0-MIN_OPA)*t
                    vis = make_single_sphere_vis(
                            radius=r,
                            color=color,
                            opacity=op,
                            prim_path=f"/Visuals/GoalSphere/layer_{i}"
                          )
                    self.goal_pos_visualizer.append(vis)
            # Görünür yap
            for vis in self.goal_pos_visualizer:
                vis.set_visibility(True)
        else:
            self._rssi_label.visible = False
            self.traj_vis.set_visibility(False)
            if hasattr(self, "goal_pos_visualizer"):
                for vis in self.goal_pos_visualizer:
                    vis.set_visibility(False)

    def _debug_vis_callback(self, event):
        # update the markers
        for vis in self.goal_pos_visualizer: vis.visualize(self._desired_pos_w) 

        all_pts = self._traj_buf.reshape(-1, 3)   # (num_envs*TRAJ_LEN, 3)
        self.traj_vis.visualize(all_pts)

        val = self._last_rssi[0].item() if self._last_rssi.numel() > 1 else self._last_rssi.item()
        self._rssi_label.text = f"{val:.1f} dBm"
