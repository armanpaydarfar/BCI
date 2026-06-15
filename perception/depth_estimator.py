# Vendored from harmony_vlm (https://github.com/vivianchen98/harmony_vlm) @ cfa01b6
# by Vivian Chen. Folded into the BCI repo for WS3 (2026-06-15). Edit here, not
# upstream; see Documents/SoftwareDocs/projects/harmony-bci/vlm-integration/.
"""
depth_estimator.py — Monocular metric depth estimation using Apple Depth Pro.

Wraps the Depth Pro model for use with OpenCV BGR frames from the Pupil Labs
pipeline.  Returns dense depth maps in metres at the original input resolution.

Usage standalone:
    python -m utils.depth_estimator path/to/image.jpg
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch

_DEFAULT_DEVICE = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)


class DepthEstimator:
    """Monocular depth estimation via Apple Depth Pro.

    Parameters
    ----------
    checkpoint : path to ``depth_pro.pt`` weights file.
    device     : inference device (cuda / mps / cpu).
    precision  : torch dtype — use ``torch.float16`` on GPU for ~2x speed.
    """

    def __init__(
        self,
        checkpoint: str | Path = "models/depth_pro.pt",
        device: str = _DEFAULT_DEVICE,
        precision: torch.dtype = torch.float32,
        save_path: str | Path | None = None,
    ) -> None:
        import depth_pro
        from depth_pro.depth_pro import DEFAULT_MONODEPTH_CONFIG_DICT
        from dataclasses import replace

        config = replace(DEFAULT_MONODEPTH_CONFIG_DICT, checkpoint_uri=str(checkpoint))
        self.device = torch.device(device)
        self.precision = precision
        self.save_path: Path | None = Path(save_path) if save_path else None
        self._save_idx = 0
        if self.save_path:
            self.save_path.mkdir(parents=True, exist_ok=True)

        print(f"[DepthEstimator] Loading model from {checkpoint} → {device} ({precision})")
        self.model, self.transform = depth_pro.create_model_and_transforms(
            config=config,
            device=self.device,
            precision=precision,
        )
        self.model.eval()
        print("[DepthEstimator] Model ready.")

    @torch.no_grad()
    def estimate(
        self,
        bgr: np.ndarray,
        f_px: Optional[float] = None,
        gaze_xy: Optional[tuple[float, float]] = None,
    ) -> tuple[np.ndarray, Optional[str]]:
        """Run depth estimation on a single BGR frame.

        Parameters
        ----------
        bgr     : OpenCV BGR image, shape (H, W, 3), dtype uint8.
        f_px    : Optional focal length in pixels.  If provided, the model uses
                  it for metric scale; otherwise it estimates focal length from
                  the image content.
        gaze_xy : Optional (x, y) gaze/fixation pixel.  When provided and
                  save_path is set, the fixation is drawn on the depth image
                  with its metric depth label.

        Returns
        -------
        depth      : np.ndarray of shape (H, W), dtype float32, values in metres.
        saved_path : path to the saved .png (relative to cwd), or None.
        """
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        tensor = self.transform(rgb)
        f_px_t = torch.tensor(f_px, device=self.device, dtype=self.precision) if f_px is not None else None
        prediction = self.model.infer(tensor, f_px=f_px_t)
        depth = prediction["depth"].squeeze().cpu().numpy().astype(np.float32)

        saved_path: Optional[str] = None
        if self.save_path is not None:
            prefix = self.save_path / f"depth_{self._save_idx:04d}"
            np.save(f"{prefix}.npy", depth)
            d_vis = self._render_depth_with_colorbar(depth, gaze_xy=gaze_xy)
            cv2.imwrite(f"{prefix}.png", d_vis)
            saved_path = f"{prefix}.png"
            self._save_idx += 1

        return depth, saved_path

    @staticmethod
    def _render_depth_with_colorbar(
        depth: np.ndarray,
        bar_width: int = 60,
        gaze_xy: Optional[tuple[float, float]] = None,
    ) -> np.ndarray:
        """Render a depth map with an INFERNO colourmap, metric colorbar, and
        optional fixation marker labeled with the depth value."""
        d_min, d_max = float(depth.min()), float(depth.max())
        d_norm = (depth - d_min) / (d_max - d_min + 1e-8)
        d_colour = cv2.applyColorMap((d_norm * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)

        h, w = d_colour.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = max(0.4, h / 2000)
        thickness = max(1, int(h / 1500))

        # ── draw fixation marker with depth label ────────────────────────
        if gaze_xy is not None:
            gx = max(0, min(int(round(gaze_xy[0])), w - 1))
            gy = max(0, min(int(round(gaze_xy[1])), h - 1))
            depth_val = float(depth[gy, gx])
            radius = max(8, int(h / 80))
            # crosshair
            cv2.drawMarker(d_colour, (gx, gy), (0, 255, 0),
                           cv2.MARKER_CROSS, radius * 3, thickness + 1, cv2.LINE_AA)
            cv2.circle(d_colour, (gx, gy), radius, (0, 255, 0), thickness, cv2.LINE_AA)
            # label with background
            label = f"{depth_val:.2f}m"
            lbl_scale = font_scale * 1.4
            lbl_thick = thickness + 1
            (tw, th), baseline = cv2.getTextSize(label, font, lbl_scale, lbl_thick)
            lx = gx + radius + 6
            ly = gy - radius
            # keep label on screen
            if lx + tw + 4 > w:
                lx = gx - radius - tw - 6
            if ly - th - 4 < 0:
                ly = gy + radius + th + 6
            cv2.rectangle(d_colour, (lx - 2, ly - th - 4), (lx + tw + 4, ly + baseline + 2),
                          (0, 0, 0), cv2.FILLED)
            cv2.putText(d_colour, label, (lx, ly),
                        font, lbl_scale, (0, 255, 0), lbl_thick, cv2.LINE_AA)

        # ── colorbar panel ───────────────────────────────────────────────
        bar = np.linspace(255, 0, h, dtype=np.uint8).reshape(h, 1)
        bar = np.repeat(bar, bar_width, axis=1)
        bar_colour = cv2.applyColorMap(bar, cv2.COLORMAP_INFERNO)

        text_w = 80 + int(font_scale * 60)
        panel = np.zeros((h, bar_width + text_w, 3), dtype=np.uint8)
        panel[:, :bar_width] = bar_colour[:, :bar_width]

        n_ticks = 5
        label_margin = 8
        for i in range(n_ticks):
            frac = i / (n_ticks - 1)
            y = int(frac * (h - 1))
            val = d_max - frac * (d_max - d_min)
            label = f"{val:.2f}m"
            (tw, th_t), _ = cv2.getTextSize(label, font, font_scale, thickness)
            ty = max(th_t + 2, min(y + th_t // 2, h - 2))
            cv2.putText(panel, label, (bar_width + label_margin, ty),
                        font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

        return np.hstack([d_colour, panel])


# ── standalone smoke test ────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Depth Pro single-image inference")
    p.add_argument("image", help="Path to input image")
    p.add_argument("--checkpoint", default="models/depth_pro.pt", help="Depth Pro weights")
    p.add_argument("--save_path", default=None, help="Directory to save .npy + .png results")
    args = p.parse_args()

    bgr = cv2.imread(args.image)
    if bgr is None:
        print(f"ERROR: cannot read {args.image}")
        raise SystemExit(1)

    estimator = DepthEstimator(checkpoint=args.checkpoint, save_path=args.save_path)
    depth, saved = estimator.estimate(bgr)

    print(f"Input : {args.image}  shape={bgr.shape}")
    print(f"Depth : shape={depth.shape}  min={depth.min():.3f}m  max={depth.max():.3f}m  mean={depth.mean():.3f}m")

    if saved:
        print(f"Saved to: {saved}")
    else:
        d_vis = DepthEstimator._render_depth_with_colorbar(depth)
        cv2.imshow("Depth Pro", d_vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
