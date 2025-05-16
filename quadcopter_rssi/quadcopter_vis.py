from __future__ import annotations

"""Shared visualization helpers for Quadcopter‑RSSI demos.

Everything that is *only* needed for visual feedback lives here so that the
main environment file focuses on physics + RL logic.

Usage
-----
>>> from quadcopter_vis import make_single_sphere_vis, make_traj_point_vis, QuadcopterEnvWindow

Make sure that this module is on *PYTHONPATH* (or lives next to your main
script) before importing.
"""

from typing import TYPE_CHECKING

import omni.ui as ui
import omni.kit.viewport.utility as vp_util

from isaaclab.envs.ui import BaseEnvWindow
from isaaclab.envs import ViewerCfg          
from isaaclab.envs.ui.viewport_camera_controller import ViewportCameraController
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.sim import SphereCfg, PreviewSurfaceCfg

# -----------------------------------------------------------------------------
# Helper factory functions
# -----------------------------------------------------------------------------

def make_single_sphere_vis(
    radius: float,
    color: tuple[float, float, float] = (1.0, 0.0, 0.0),
    opacity: float = 0.3,
    prim_path: str = "/Visuals/GoalSphere",
) -> VisualizationMarkers:
    """Create a single semi‑transparent sphere marker.

    Parameters
    ----------
    radius : float
        Sphere radius in *scene units*.
    color : tuple
        RGB triple in the range [0, 1]. Default is red.
    opacity : float, optional
        Alpha channel (0 → transparent, 1 → opaque), by default 0.3.
    prim_path : str, optional
        USD prim path under which the marker will be spawned.
    """
    cfg = VisualizationMarkersCfg(
        prim_path=prim_path,
        markers={
            "sphere": SphereCfg(
                radius=radius,
                visual_material=PreviewSurfaceCfg(
                    diffuse_color=color,
                    opacity=opacity,
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
    """Pre‑allocate a pool of sphere markers for trajectory breadcrumbs."""
    markers = {
        f"pt_{i:04d}": SphereCfg(
            radius=radius,
            visual_material=PreviewSurfaceCfg(diffuse_color=color, opacity=1.0),
        )
        for i in range(max_points)
    }
    cfg = VisualizationMarkersCfg(prim_path=prim_path, markers=markers)
    return VisualizationMarkers(cfg)

# -----------------------------------------------------------------------------
# Custom viewport window
# -----------------------------------------------------------------------------

if TYPE_CHECKING:  # avoid circular import at runtime
    from quadcopter_env import QuadcopterRSSIEnv  # adjust to your actual module name


class QuadcopterEnvWindow(BaseEnvWindow):
    """Lightweight UI/window wrapper for *QuadcopterRSSIEnv*.

    This class centralises all Omniverse‑UI code. Nothing here is required for
    the RL loop itself – remove it from *production* or *headless* runs to save
    resources.
    """

    def __init__(self, env: "QuadcopterRSSIEnv", window_name: str = "IsaacLab"):
        super().__init__(env, window_name)

        # ------------------ camera ------------------
        viewer_cfg = ViewerCfg(
            eye=(1.5, 1.5, 1.5),
            lookat=(0.0, 0.0, 0.0),
            cam_prim_path="/OmniverseKit_Persp",
            resolution=(1280, 720),
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

        # live RSSI read‑out in the top‑left corner
        self._vp_window = vp_util.get_active_viewport_window()
        with self._vp_window.get_frame("Overlay"):
            self._rssi_label = ui.Label(
                "-150 dBm",
                alignment=ui.Alignment.CENTER,
                size=18,
                style={"color": 0xFFFFFFFF},
            )

    # ---------------------------------------------------------------------
    # public helper – call this from the env to update text live
    # ---------------------------------------------------------------------
    def set_rssi(self, value_dbm: float):
        self._rssi_label.text = f"{value_dbm:.1f} dBm"
        
    def set_visibility(self, vis):
        self._rssi_label.visible = vis