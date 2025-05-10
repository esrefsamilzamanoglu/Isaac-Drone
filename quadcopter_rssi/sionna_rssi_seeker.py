from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import gymnasium as gym
import torch
import math

# Isaac Lab
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

# Sionna‑RT & Mitsuba
import mitsuba as mi
mi.set_variant("cuda_ad_mono_polarized")

from sionna.rt import load_scene, PlanarArray, Transmitter, Receiver, PathSolver

# --- Özellik sözlüğünü yakala (sürüm farkı için try/except) ------------------
try:                                    # 0.19.x   (eski)
    from sionna.rt.radio_materials.itu_material import _ITU_MATERIALS as _MATS
except ImportError:                     # 1.x.y    (yeni)
    from sionna.rt.radio_materials import itu as _itu
    _MATS = getattr(_itu, "ITU_MATERIALS", getattr(_itu, "ITU_MATERIALS_PROPERTIES"))


# Isaac Lab hazır konfigürasyonlar
from isaaclab_assets import CRAZYFLIE_CFG  # isort: skip
from isaaclab.markers import CUBOID_MARKER_CFG  # isort: skip

def _try_set_viewport():
    """Call after World built; will defer itself until a viewport exists."""
    try:
        import carb
        import omni.kit.app as kit_app
        import omni.kit.viewport.utility as vpu    # yeni yardımcı arayüz
    except ModuleNotFoundError:
        return  # headless

    # Aktif viewport henüz yoksa bir sonraki frame'e ertele
    vp_iface = vpu.get_active_viewport()
    if vp_iface is None:
        kit_app.get_app().next_update(_try_set_viewport)
        return

    # Kamera konumu
    vpu.set_camera_look_at(
        viewport_api=vp_iface,
        position=(1.0, 1.0, 0.5),
        target=(0.0, 0.0, 0.0),
    )

    # Takip modunu ayarla
    s = carb.settings.get_settings()
    s.set_bool("/app/window/viewport/follow_env_index", True)
    s.set_int ("/app/window/viewport/env_index", 1)
    s.set_string("/app/window/viewport/follow_mode", "Robot")

# ――― malzeme eşleştirmeleri ―――
ITU_ALIASES = {
    "floorboard"    : "wood",
    "ceiling_board" : "plasterboard",
    "plywood"       : "wood",
}

# ══════════════════════════════════════════════════════════
# UI Penceresi
# ══════════════════════════════════════════════════════════
class QuadcopterEnvWindow(BaseEnvWindow):
    def __init__(self, env: "QuadcopterRSSIEnv", window_name: str = "IsaacLab"):
        super().__init__(env, window_name)
        with self.ui_window_elements["main_vstack"]:
            with self.ui_window_elements["debug_frame"]:
                with self.ui_window_elements["debug_vstack"]:
                    self._create_debug_vis_ui_element("targets", self.env)


# ══════════════════════════════════════════════════════════
# Yapılandırma
# ══════════════════════════════════════════════════════════
@configclass
class QuadcopterRSSIEnvCfg(DirectRLEnvCfg):
    # Isaac Lab
    episode_length_s: float = 10.0
    decimation: int = 2
    action_space: int = 4
    state_space: int = 0 
    observation_space: int = 10
    debug_vis: bool = True

    usd_enable: bool = False
    # Dosya yolları
    _ROOT = Path(__file__).resolve().parent
    usd_path: Path = _ROOT / "assets" / "scenes" / "simple_room" / "simple_room.usd"
    sionna_scene_file: Path = _ROOT / "assets" / "scenes" / "simple_room" / "simple_room.xml"

    # Simülasyon
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
        if usd_enable else None
    )

    # Sahne
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1024, env_spacing=8, replicate_physics=True)

    # Robot
    robot: ArticulationCfg = CRAZYFLIE_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    thrust_to_weight: float = 1.9
    moment_scale: float = 0.01

    # Ödül katsayıları
    lin_vel_reward_scale: float = -0.05
    ang_vel_reward_scale: float = -0.01
    distance_to_goal_reward_scale: float = 30.0

    # Sionna
    frequency: float = 2.4e9            # Hz
    tx_power_dbm: float = 9.0           # dBm
    max_depth: int = 3
    samples_per_tx: int = 20_000
    rssi_min_dbm: float = -150.0      
    rssi_max_dbm: float =  +30.0 
    rssi_update_interval: int = 1      # adım

    ui_window_class_type = QuadcopterEnvWindow


# ══════════════════════════════════════════════════════════
# Ortam
# ══════════════════════════════════════════════════════════
class QuadcopterRSSIEnv(DirectRLEnv):
    cfg: QuadcopterRSSIEnvCfg

    def __init__(self, cfg: QuadcopterRSSIEnvCfg, render_mode: Optional[str] = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Isaac Lab tamponları
        self._actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)
        self._thrust = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._moment = torch.zeros(self.num_envs, 1, 3, device=self.device)

        # Hedef / TX pozisyonları
        self._desired_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        self._tx_pos_w = torch.zeros_like(self._desired_pos_w)

        # RSSI buf & sayaç
        self._rssi_buf = torch.zeros(self.num_envs, 1, device=self.device)
        self._rssi_counter = 0

        # Episode log
        self._episode_sums = {k: torch.zeros(self.num_envs, device=self.device) for k in [
            "lin_vel", "ang_vel", "distance_to_goal", "rssi_dbm"]}

        # Gövde & ağırlık
        self._body_id = self._robot.find_bodies("body")[0]
        self._robot_mass = self._robot.root_physx_view.get_masses()[0].sum()
        g = torch.tensor(self.sim.cfg.gravity, device=self.device).norm()
        self._robot_weight = (self._robot_mass * g).item()

        # Sionna kurulumu
        self._init_sionna()

        self.set_debug_vis(self.cfg.debug_vis)

    # ───────────────────────────────────────────────────────
    # Sionna setup
    # ───────────────────────────────────────────────────────
    def _init_sionna(self):
        # alias'ları kaydet (var‑olanı ezmez, yoksa ekler)
        for alias, base in ITU_ALIASES.items():
            # Eğer alias yoksa doğrudan kopyala
            if alias not in _MATS:
                _MATS[alias] = _MATS[base].copy()
            else:
                # Alias mevcut ama aralık eksik → base'teki eksik frekansları ekle
                for rng, props in _MATS[base].items():
                    if rng not in _MATS[alias]:
                        _MATS[alias][rng] = props
        self._sionna_scene = load_scene(str(self.cfg.sionna_scene_file))
        self._sionna_scene.frequency = self.cfg.frequency
        self._sionna_scene.bandwidth = 100e6
        arr = PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="V")
        self._sionna_scene.tx_array = arr
        self._sionna_scene.rx_array = arr

        
        # --- Burası ÖNEMLİ: N adet verici & alıcı ekle ---
        self._tx_devices: list[Transmitter] = []
        self._rx_devices: list[Receiver]    = []
        for i in range(self.num_envs):
            tx = Transmitter(f"tx{i}",
                             position=[0,0,0],
                             orientation=[0,0,0],
                             power_dbm=self.cfg.tx_power_dbm)
            rx = Receiver(f"rx{i}",
                          position=[0,0,0],
                          orientation=[0,0,0])
            self._sionna_scene.add(tx)
            self._sionna_scene.add(rx)
            self._tx_devices.append(tx)
            self._rx_devices.append(rx)

        # PathSolver’ı CUDA Graphs ile başlatmak launch latency’yi düşürür
        self._ps = PathSolver(loop_mode="cuda_graphs")

    # ───────────────────────────────────────────────────────
    # Isaac Lab sahne kurulumu
    # ───────────────────────────────────────────────────────
    def _setup_scene(self):
        super()._setup_scene()

        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot

        if self.cfg.usd_enable and self.cfg.terrain is not None:
            self.cfg.terrain.num_envs   = self.scene.cfg.num_envs
            self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
            self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        else:
            # USD devre dışı → düz zemin kabul edip env_origin’leri sıfırla
            self._terrain = type(
                "DummyTerrain",
                (),
                {"env_origins": torch.zeros(self.num_envs, 3, device=self.device)},
            )()
            
        self.scene.clone_environments(copy_from_source=True)
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)
        _try_set_viewport()

    # ───────────────────────────────────────────────────────
    # Aksiyonlar
    # ───────────────────────────────────────────────────────
    def _pre_physics_step(self, actions: torch.Tensor):
        self._actions = actions.clone().clamp(-1.0, 1.0)
        self._thrust[:, 0, 2] = self.cfg.thrust_to_weight * self._robot_weight * (self._actions[:, 0] + 1.0) / 2.0
        self._moment[:, 0, :] = self.cfg.moment_scale * self._actions[:, 1:]

    def _apply_action(self):
        self._robot.set_external_force_and_torque(self._thrust, self._moment, body_ids=self._body_id)

    # ───────────────────────────────────────────────────────
    # Gözlemler
    # ───────────────────────────────────────────────────────
    def _get_observations(self):
        if self._rssi_counter % self.cfg.rssi_update_interval == 0:
            # 1) Tüm Tx/Rx pozisyonlarını topluca CPU’ya al
            tx_cpu = self._tx_pos_w.cpu().numpy()
            rx_cpu = self._robot.data.root_pos_w[:, :3].cpu().numpy()

            # 2) Her cihaza tek tek ata
            for tx, p in zip(self._tx_devices, tx_cpu):
                tx.position = p.tolist()
            for rx, p in zip(self._rx_devices, rx_cpu):
                rx.position = p.tolist()

            # 3) Tek seferde tüm yolları hesaplat
            paths = self._ps(
                scene=self._sionna_scene,
                max_depth=self.cfg.max_depth,
                samples_per_src=self.cfg.samples_per_tx,
                synthetic_array=True,
                los=True,
                specular_reflection=True,
                diffuse_reflection=True,
                refraction=True,
                seed=0,
            )

            # 4) Karmaşık kazanç katsayılarını Torch tensör olarak çek
            a_torch, _ = paths.cir(normalize_delays=True, out_type="torch")
            a_torch = a_torch.to(self.device)                           # [N_rx, N_tx, N_paths]
            power   = (torch.abs(a_torch)**2).sum(dim=-1)               # [N_rx, N_tx]
            gain    = torch.diagonal(power)                             # [N] — sadece (i,i) çiftleri

            # 5) FSPL garantisi için fallback
            rx_pos = self._robot.data.root_pos_w[:, :3]
            d      = torch.linalg.norm(rx_pos - self._tx_pos_w, dim=1).clamp(min=1e-3)
            lam    = 3e8 / self.cfg.frequency
            fspl   = (lam / (4 * math.pi * d))**2
            gain   = torch.maximum(gain, fspl)

            # 6) dBm’e çevir & normalize et
            prx_dbm     = self.cfg.tx_power_dbm + 10.0 * torch.log10(gain)
            rssi_dbm_t  = prx_dbm.unsqueeze(1)                          # [N,1]
            scale       = self.cfg.rssi_max_dbm - self.cfg.rssi_min_dbm
            clamped     = torch.clamp(rssi_dbm_t,
                                     self.cfg.rssi_min_dbm,
                                     self.cfg.rssi_max_dbm)
            self._rssi_buf = ((clamped - self.cfg.rssi_min_dbm) / scale) * 2.0 - 1.0
            self._episode_sums["rssi_dbm"] += rssi_dbm_t.squeeze() * self.step_dt

        self._rssi_counter += 1

        desired_pos_b, _ = subtract_frame_transforms(
            self._robot.data.root_state_w[:, :3],
            self._robot.data.root_state_w[:, 3:7],
            self._tx_pos_w,  # hedef = TX
        )
        obs = torch.cat([
            self._robot.data.root_lin_vel_b,
            self._robot.data.root_ang_vel_b,
            self._robot.data.projected_gravity_b,
            self._rssi_buf,
        ], dim=-1)
        return {"policy": obs}

    # ───────────────────────────────────────────────────────
    # Ödüller / bitiş
    # ───────────────────────────────────────────────────────
    def _get_rewards(self):
        lin_vel = torch.sum(self._robot.data.root_lin_vel_b ** 2, dim=1)
        ang_vel = torch.sum(self._robot.data.root_ang_vel_b ** 2, dim=1)
        distance_to_goal = torch.linalg.norm(self._tx_pos_w - self._robot.data.root_pos_w, dim=1)
        distance_to_goal_mapped = 1 - torch.tanh(distance_to_goal / 0.8)
        rewards = {
            "lin_vel": lin_vel * self.cfg.lin_vel_reward_scale * self.step_dt,
            "ang_vel": ang_vel * self.cfg.ang_vel_reward_scale * self.step_dt,
            "distance_to_goal": distance_to_goal_mapped * self.cfg.distance_to_goal_reward_scale * self.step_dt,
        }
        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
        for k, v in rewards.items():
            self._episode_sums[k] += v
        return reward

    def _get_dones(self):
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        died = torch.logical_or(self._robot.data.root_pos_w[:, 2] < 0.1, self._robot.data.root_pos_w[:, 2] > 2.0)
        return died, time_out

    # ───────────────────────────────────────────────────────
    # Reset
    # ───────────────────────────────────────────────────────
    def _reset_idx(self, env_ids: Optional[torch.Tensor]):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES

        # Episode log hesaplamaları 
        final_distance_to_goal = torch.linalg.norm(
            self._desired_pos_w[env_ids] - self._robot.data.root_pos_w[env_ids], dim=1
        ).mean()

        extras: dict[str, float] = {}
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            extras[f"Episode_Reward/{key}"] = (episodic_sum_avg / self.max_episode_length_s).item()
            # sıfırla
            self._episode_sums[key][env_ids] = 0.0

        extras["Episode_Termination/died"] = int(torch.count_nonzero(self.reset_terminated[env_ids]))
        extras["Episode_Termination/time_out"] = int(torch.count_nonzero(self.reset_time_outs[env_ids]))
        extras["Metrics/final_distance_to_goal"] = final_distance_to_goal.item()

        self.extras["log"] = extras

        # Isaac Lab reset
        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)
        if len(env_ids) == self.num_envs:
            self.episode_length_buf = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))
        self._actions[env_ids] = 0.0

        # ─── Yeni TX & hedef pozisyonu ───
        self._tx_pos_w[env_ids, :2] = torch.zeros_like(self._tx_pos_w[env_ids, :2]).uniform_(-2.0, 2.0)
        self._tx_pos_w[env_ids, :2] += self._terrain.env_origins[env_ids, :2]
        self._tx_pos_w[env_ids, 2] = torch.zeros_like(self._tx_pos_w[env_ids, 2]).uniform_(0.5, 1.5)
        # Robotun izleyeceği hedef de aynı olsun
        self._desired_pos_w[env_ids] = self._tx_pos_w[env_ids]

        # Robot state (başlangıç)
        jp = self._robot.data.default_joint_pos[env_ids]
        jv = self._robot.data.default_joint_vel[env_ids]
        root = self._robot.data.default_root_state[env_ids]
        root[:, :3] += self._terrain.env_origins[env_ids]
        self._robot.write_root_pose_to_sim(root[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(root[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(jp, jv, None, env_ids)

    # ───────────────────────────────────────────────────────
    # Debug viz
    # ───────────────────────────────────────────────────────
    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "goal_pos_visualizer"):
                cfg = CUBOID_MARKER_CFG.copy()
                cfg.markers["cuboid"].size = (0.05, 0.05, 0.05)
                cfg.prim_path = "/Visuals/Command/goal_position"
                self.goal_pos_visualizer = VisualizationMarkers(cfg)
            self.goal_pos_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_pos_visualizer"):
                self.goal_pos_visualizer.set_visibility(False)

    def _debug_vis_callback(self, _event):
        self.goal_pos_visualizer.visualize(self._tx_pos_w)


# ───────────────────────── main test ─────────────────────────
if __name__ == "__main__":
    cfg = QuadcopterRSSIEnvCfg()
    env = QuadcopterRSSIEnv(cfg, render_mode="human")
    obs, _ = env.reset()
    done = False
    while not done:
        act = env.single_action_space.sample() * 0
        obs, _, term, trunc, _ = env.step(act)
        done = bool(term or trunc)
    env.close()
