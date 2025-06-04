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
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg, LineStripCfg
from isaaclab.sim import SphereCfg, PreviewSurfaceCfg

if TYPE_CHECKING:  # avoid circular import at runtime
    from quadcopter_env import QuadcopterRSSIEnv  # adjust to your actual module name

# -----------------------------------------------------------------------------
# Helper factory functions
# -----------------------------------------------------------------------------


def make_rf_path_vis(
    max_paths: int,
    max_segments: int,
    color: tuple[float, float, float] = (0.2, 0.8, 1.0),
    thickness: float = 0.003,
    prim_path: str = "/Visuals/RFPaths",
) -> VisualizationMarkers:
    """
    Çok-yollu (multipath) RF ışınlarını çizmek için line-strip marker’ları oluşturur.
    * ``max_paths``        : Aynı anda göstereceğin toplam yol (ışın) sayısı
    * ``max_segments``     : Bir yolda olabilecek max kırılım + 1 (TX-RX arası düğüm sayısı-1)
    * ``color`` / ``thickness`` : Görsel ayarlar
    """
    markers = {
        f"path_{i}": LineStripCfg(
            thickness=thickness,
            color=color,
            max_num_points=max_segments + 1,  #   ►  n düğüm  ==  n-1 çizgi
        )
        for i in range(max_paths)
    }
    cfg = VisualizationMarkersCfg(prim_path=prim_path, markers=markers, replicate=False)
    return VisualizationMarkers(cfg)

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

# -----------------------------------------------------------------------------
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