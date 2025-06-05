from __future__ import annotations

"""Shared visualization helpers for Quadcopter-RSSI demos.

Everything that is *only* needed for visual feedback lives here so that the
main environment file focuses on physics + RL logic.

Usage
-----
>>> from quadcopter_vis import make_single_sphere_vis, make_traj_point_vis, QuadcopterEnvWindow

Make sure that this module is on *PYTHONPATH* (or lives next to your main
script) before importing.
"""

from typing import TYPE_CHECKING

# -----------------------------------------------------------------------------
# Guard Omni UI imports for headless compatibility
# -----------------------------------------------------------------------------
try:
    import omni.ui as ui
    import omni.kit.viewport.utility as vp_util
    UI_AVAILABLE = True
except ImportError:
    UI_AVAILABLE = False

from isaaclab.envs.ui import BaseEnvWindow  # safe even if headless
from isaaclab.envs import ViewerCfg          
from isaaclab.envs.ui.viewport_camera_controller import ViewportCameraController
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.sim import SphereCfg, CylinderCfg, PreviewSurfaceCfg

import torch

if TYPE_CHECKING:  # avoid circular import at runtime
    from quadcopter_env import QuadcopterRSSIEnv  # adjust to your actual module name

# -----------------------------------------------------------------------------
# Helper factory functions
# -----------------------------------------------------------------------------

def make_single_sphere_vis(
    radius: float,
    color: tuple[float, float, float] = (1.0, 0.0, 0.0),
    opacity: float = 0.2,
    prim_path: str = "/Visuals/GoalSphere",
) -> VisualizationMarkers:
    """Create a single semi-transparent sphere marker."""
    cfg = VisualizationMarkersCfg(
        prim_path=prim_path,
        markers={
            "sphere": SphereCfg(
                radius=radius,
                visual_material=PreviewSurfaceCfg(
                    diffuse_color=color,
                    opacity=opacity,
                    metallic=0.0,            # metalik değil
                    roughness=1.0
                ),
            )
        },
    )
    return VisualizationMarkers(cfg)


def make_traj_point_vis(
    max_points: int,
    radius: float = 0.003,
    color: tuple[float, float, float] = (0.0, 1.0, 0.0),
    prim_path: str = "/Visuals/TrajPoints",
) -> VisualizationMarkers:
    """Pre-allocate a pool of sphere markers for trajectory breadcrumbs."""
    markers = {
        f"pt_{i:04d}": SphereCfg(
            radius=radius,
            visual_material=PreviewSurfaceCfg(diffuse_color=color, opacity=1),
        )
        for i in range(max_points)
    }
    cfg = VisualizationMarkersCfg(prim_path=prim_path, markers=markers)
    return VisualizationMarkers(cfg)


_CYL_R        = 0.005          # 1 cm çap
_MAX_RAYS     = 64             # PathSolver'da samples_per_tx
_MAX_BOUNCES  = 3              # env.cfg.max_depth
_PREFIX       = "/Visuals/Rays"


def make_path_vis(device="cpu"):
    """
    İnce uzun silindir (CylinderCfg) havuzu oluşturur ve eşlik eden
    `visualize(paths)` callback'ini döndürür.
    ------------------------------------------------------------------------
    Dönüş:
        markers   : VisualizationMarkers  (önceden yaratılmış USD silindirleri)
        visualize : callable(paths)       (her kare çağrılıp segmentleri günceller)
    """
    # --- 1) Havuzu kur ------------------------------------------------------
    markers = VisualizationMarkers()
    pool = _MAX_RAYS * _MAX_BOUNCES
    for i in range(pool):
        cyl_cfg = CylinderCfg(
            prim_path=f"{_PREFIX}/cyl_{i:05d}",
            radius=_CYL_R,
            height=1.0,        # gerçek uzunluk scale ile ayarlanacak
            axis="Z",
            visual_material=PreviewSurfaceCfg(
                diffuse_color=(0.0, 1.0, 1.0), opacity=0.6
            ),
        )
        markers.add_visual(cyl_cfg)

    # --- 2) Her kare çağrılacak helper --------------------------------------
    uz = torch.tensor([0.0, 0.0, 1.0], device=device)

    def visualize(paths):
        if paths is None:
            markers.set_visibility(False)
            return

        # (R, V, 3) – 0 dummy köşeleri ayıklıyoruz
        verts = paths.vertices.squeeze(0).to(device)         # (1,R,…) -> (R,…)
        mask  = torch.any(torch.abs(verts) > 1e-6, dim=-1)

        T_all = []
        for r in range(verts.shape[0]):
            pts = verts[r][mask[r]]                          # gerçek köşeler
            for j in range(pts.shape[0] - 1):
                p0, p1 = pts[j], pts[j + 1]
                v  = p1 - p0
                L  = torch.linalg.norm(v)
                if L < 1e-6:
                    continue

                mid = 0.5 * (p0 + p1)
                vn  = v / L

                # Rodrigues: Z eksenini vn vektörüne döndür
                axis = torch.cross(uz, vn)
                c    = torch.clamp(torch.dot(uz, vn), -1.0, 1.0)
                if torch.norm(axis) < 1e-6:                   # paralel (+/-)
                    R = torch.diag(torch.tensor([1., 1., 1.], device=device))
                    if c < 0:                                 # ters yönde
                        R[1, 1] = R[2, 2] = -1
                else:
                    k  = 1.0 / (1.0 + c)
                    K  = torch.tensor([[0, -axis[2], axis[1]],
                                      [axis[2], 0, -axis[0]],
                                      [-axis[1], axis[0], 0]], device=device)
                    R  = torch.eye(3, device=device) + K + k * K @ K

                # 4×4 world transform
                T = torch.eye(4, device=device)
                T[:3, :3] = R
                T[:3,  3] = mid
                T[0, 0]  *= _CYL_R * 2        # X-scale = çap
                T[1, 1]  *= _CYL_R * 2        # Y-scale = çap
                T[2, 2]  *= L                 # Z-scale = segment uzunluğu
                T_all.append(T)

        if T_all:
            markers.set_visibility(True)
            markers.visualize(torch.stack(T_all))
        else:
            markers.set_visibility(False)

    return markers, visualize

# Custom viewport window
# -----------------------------------------------------------------------------

if UI_AVAILABLE:
    class QuadcopterEnvWindow(BaseEnvWindow):
        """Lightweight UI/window wrapper for *QuadcopterRSSIEnv*."""

        def __init__(self, env: "QuadcopterRSSIEnv", window_name: str = "IsaacLab"):
            super().__init__(env, window_name)

            # ------------------ camera ------------------
            viewer_cfg = ViewerCfg(
                eye=(0, 2, 1),
                lookat=(0.0, 0.0, 0.0),
                cam_prim_path="/OmniverseKit_Persp",
                resolution=(3840, 2160),
                origin_type="asset_root",
                env_index=0,
                asset_name="robot",
                body_name=None,
            )
            self.camera_ctrl = ViewportCameraController(self.env, viewer_cfg)

            # ------------------- UI ---------------------
            with self.ui_window_elements["main_vstack"]:
                with self.ui_window_elements["debug_frame"]:
                    with self.ui_window_elements["debug_vstack"]:
                        self._create_debug_vis_ui_element("targets", self.env)

            # live RSSI read-out in the top-left corner
            self._vp_window = vp_util.get_active_viewport_window()
            with self._vp_window.get_frame("Overlay"):
                self._rssi_label = ui.Label(
                    f"{env.cfg.rssi_min_dbm:.1f} dBm",
                    alignment=ui.Alignment.CENTER,
                    size=18,
                    style={"color": 0xFFFFFFFF},
                )

        def set_rssi(self, value_dbm: float):
            """Update RSSI text in UI."""
            self._rssi_label.text = f"{value_dbm:.1f} dBm"

        def set_visibility(self, vis: bool):
            """Show or hide the RSSI overlay label."""
            self._rssi_label.visible = bool(vis)

else:
    # Headless stub class
    class QuadcopterEnvWindow:
        def __init__(self, *args, **kwargs):
            pass
        def set_rssi(self, value_dbm: float):
            pass
        def set_visibility(self, vis: bool):
            pass