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
_CYL_R        = 0.005        # Silindir yarıçapı (metre cinsinden, 1 cm)
_MAX_RAYS     = 64           # PathSolver’da samples_per_src
_MAX_BOUNCES  = 3            # cfg.max_depth
_ROOT_PRIM    = "/Visuals/Rays"   # Tüm silindirler bu prefix altında spawn edilecek

def make_path_vis(device: str = "cpu"):
    """
    Sionna-RT paths.vertices → ince silindir marker’larına dönüştüren
    VisualizationMarkers + callback(kullanıcıya draw_fn) ikilisi döner.

    Device:
        "cpu" veya "cuda" (env.device) olarak verilmeli, çünkü
        çizim için world-transform hesaplamalarında torch kullanıyoruz.

    Dönenler:
      markers : VisualizationMarkers  (USD içinde önceden yaratılmış CylinderCfg havuzu)
      draw_fn : Callable(paths)        (her karede çağrılıp, paths argümanına göre silindirleri günceller)
    """

    # ===========================================================
    #  1) En başta “markers” listesine direkt tüm CylinderCfg’leri koyun:
    # ===========================================================
    # Havuzdaki silindir sayısı = _MAX_RAYS * _MAX_BOUNCES
    pool_size = _MAX_RAYS * _MAX_BOUNCES

    # Her silindir için bir CylinderCfg oluşturup listeye ekleyelim:
    cyl_cfgs = []
    for i in range(pool_size):
        cfg = CylinderCfg(
            prim_path=f"{_ROOT_PRIM}/cyl_{i:05d}",
            radius=_CYL_R,
            height=1.0,        # Sonradan world-transform’da z-yönündeki ölçekle gerçek boyu ayarlanacak
            axis="Z",
            visual_material=PreviewSurfaceCfg(
                diffuse_color=(0.0, 1.0, 1.0),  # Camgöbeği renk
                opacity=0.6
            ),
        )
        cyl_cfgs.append(cfg)

    # “markers” listesi boş olamaz; onu doğrudan bunlarla başlatıyoruz:
    vm_cfg = VisualizationMarkersCfg(
        prim_path=_ROOT_PRIM,
        markers=cyl_cfgs,    # ⭐ Doğru: boş değil, tam havuzu tanımladık
    )
    markers = VisualizationMarkers(vm_cfg)

    # ===========================================================
    #  2) Her karede çağrılacak draw(paths) fonksiyonu
    # ===========================================================
    #   - paths.vertices: shape (num_tx, num_rx, num_rays, max_bounces+2, 3)
    #   - Tek Tx/Tek Rx’li tipik kullanımda squeeze ederek (num_rays, max_bounces+2, 3) elde edilir.
    uz = torch.tensor([0.0, 0.0, 1.0], device=device)

    def draw(paths):
        # Eğer henüz paths yoksa veya None geliyorsa marker’ları gizle:
        if paths is None:
            markers.set_visibility(False)
            return

        # 1) paths.vertices’e erişelim ve fazladan boyutları sıkıştırıp GPU/CPU’ya taşıyalım:
        #    → Orijinal: (1, 1, R, B+2, 3)  ya da  (1, R, B+2, 3) (Sionna sürümüne göre)
        #    Biz R-düzeyini öne çıkaracağız:
        verts = paths.vertices.squeeze(0)      # (1,R,B+2,3) → (R,B+2,3)
        verts = verts.to(device)               # CPU/CUDA eşleşmesi için

        # 2) Her ray içindeki “dummy 0” satırlarını maskele:
        valid_mask = torch.any(torch.abs(verts) > 1e-6, dim=-1)  
        # valid_mask.shape = (R, B+2), True olanlar gerçekte var olan köşe noktaları

        # 3) Her ray için tüm segmentleri hesaplayıp 4×4 transform listesine dönüştürelim:
        transforms = []
        for r in range(verts.shape[0]):            # r = 0..(num_rays-1)
            ray_pts = verts[r][valid_mask[r]]      # örn. [(x0,y0,z0), (x1,y1,z1), …]
            # ray_pts.shape = (bounce_count+2, 3)

            for j in range(ray_pts.shape[0] - 1):
                p0 = ray_pts[j]
                p1 = ray_pts[j + 1]
                v  = p1 - p0
                L  = torch.linalg.norm(v)
                if L < 1e-6:
                    continue  # çok kısa, yok say

                # 3A) Orta nokta:
                mid_point = 0.5 * (p0 + p1)
                # 3B) Yön birimi:
                vn = v / L

                # 3C) Rodrigues rotasyonunu hesaplayalım:
                axis = torch.cross(uz, vn)
                c_val = torch.clamp(torch.dot(uz, vn), -1.0, 1.0)

                if torch.norm(axis) < 1e-6:
                    # Z ekseniyle paralel veya tersse:
                    R = torch.eye(3, device=device)
                    if c_val < 0:
                        # Z-ekseni ters yönde: 180° (X ekseni etrafında)
                        R[1, 1] = -1.0
                        R[2, 2] = -1.0
                else:
                    k = 1.0 / (1.0 + c_val)
                    K = torch.tensor([
                        [0.0,        -axis[2],  axis[1]],
                        [axis[2],     0.0,      -axis[0]],
                        [-axis[1],    axis[0],   0.0   ]
                    ], device=device)
                    R = torch.eye(3, device=device) + K + k * (K @ K)

                # 3D → 4×4 dünya dönüşüm matrisi:
                T = torch.eye(4, device=device)
                T[:3, :3] = R
                T[:3,  3] = mid_point
                # Ölçekleri uygulayalım:
                T[0, 0] *= (_CYL_R * 2)    # X ekseni = çap
                T[1, 1] *= (_CYL_R * 2)    # Y ekseni = çap
                T[2, 2] *= L               # Z ekseni = segment uzunluğu

                transforms.append(T)

        # 4) Tek bir tensor haline getir ve marker’lara bildir:
        if transforms:
            markers.set_visibility(True)
            markers.visualize(torch.stack(transforms))
        else:
            markers.set_visibility(False)

    # markers nesnesini ve draw(paths) callback’ini döndür:
    return markers, draw
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