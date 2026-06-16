# Vendored from harmony_vlm (https://github.com/vivianchen98/harmony_vlm) @ cfa01b6
# by Vivian Chen. Folded into the BCI repo for WS3 (2026-06-15). Edit here, not
# upstream; see Documents/SoftwareDocs/projects/harmony-bci/vlm-integration/.
"""
object_detector.py — Class-agnostic segmentation with gaze hit-test.

Supports FastSAM (default), EfficientSAM, or YOLO-seg models.
Run standalone for a quick smoke-test:
    python object_detector.py neon/QuickShare_2603111547
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch

# Ultralytics
try:
    from ultralytics import FastSAM, YOLO
except ImportError as e:
    raise ImportError("pip install ultralytics") from e

# EfficientSAM (optional — only needed when using --model efficient_sam_*.pt)
_HAS_EFFICIENT_SAM = False
try:
    from efficient_sam.build_efficient_sam import build_efficient_sam_vitt, build_efficient_sam_vits
    from segment_anything.automatic_mask_generator import SamAutomaticMaskGenerator
    _HAS_EFFICIENT_SAM = True
except ImportError:
    pass

# Auto-detect best available device
_DEFAULT_DEVICE = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)


# ── detection result ──────────────────────────────────────────────────────────

@dataclass
class Detection:
    label:        str
    confidence:   float
    box_xyxy:     tuple[float, float, float, float]   # (x1,y1,x2,y2) in full-res coords
    box_center:   tuple[float, float]                  # (cx, cy) in full-res coords
    mask_polygon: Optional[np.ndarray] = None          # shape (N,1,2) int32, or None


@dataclass
class GazeHit:
    """Result of probabilistic gaze-to-object inference."""
    detection:   Detection
    probability: float
    candidates:  list[tuple[Detection, float]]   # sorted descending by probability


@dataclass
class Waypoint3D:
    """3D waypoint for a detected object, in camera frame."""
    label:          str
    position_cam:   tuple[float, float, float]   # (X, Y, Z) in camera frame, metres
    pixel_center:   tuple[float, float]           # (cx, cy) mask centroid in pixels
    depth_median_m: float                          # median depth within mask, metres


def compute_3d_waypoints(
    detections: list[Detection],
    depth_map: np.ndarray,
    camera_matrix: np.ndarray,
) -> list[Waypoint3D]:
    """Backproject each detection's centroid into 3D camera-frame coordinates.

    Parameters
    ----------
    detections    : list of Detection objects (with optional mask_polygon).
    depth_map     : dense depth map (H, W), float32, values in metres.
    camera_matrix : 3x3 intrinsic matrix [[fx,0,cx],[0,fy,cy],[0,0,1]].

    Returns
    -------
    List of Waypoint3D, one per valid detection.
    """
    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]
    cx0 = camera_matrix[0, 2]
    cy0 = camera_matrix[1, 2]
    h, w = depth_map.shape[:2]
    waypoints: list[Waypoint3D] = []

    for det in detections:
        # Rasterize mask or fall back to bbox
        if det.mask_polygon is not None and len(det.mask_polygon) >= 3:
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(mask, [det.mask_polygon], 1)
            depths = depth_map[mask == 1]
            # Centroid from mask moments
            M = cv2.moments(mask, binaryImage=True)
            if M["m00"] > 0:
                cx = M["m10"] / M["m00"]
                cy = M["m01"] / M["m00"]
            else:
                cx, cy = det.box_center
        else:
            # Fallback: bbox center + depth at that pixel
            cx, cy = det.box_center
            ix = max(0, min(int(round(cx)), w - 1))
            iy = max(0, min(int(round(cy)), h - 1))
            depths = np.array([depth_map[iy, ix]])

        # Filter invalid depths
        depths = depths[(depths > 0) & np.isfinite(depths) & (depths < 10.0)]
        if len(depths) == 0:
            continue

        Z = float(np.median(depths))
        X = float((cx - cx0) * Z / fx)
        Y = float((cy - cy0) * Z / fy)

        waypoints.append(Waypoint3D(
            label=det.label,
            position_cam=(round(X, 4), round(Y, 4), round(Z, 4)),
            pixel_center=(round(cx, 1), round(cy, 1)),
            depth_median_m=round(Z, 4),
        ))

    return waypoints


# ── detector ─────────────────────────────────────────────────────────────────

class ObjectDetector:
    """
    Segmentation detector using FastSAM (default), EfficientSAM, or YOLO-seg.

    FastSAM / EfficientSAM produce class-agnostic masks ("segment_0", etc.).
    YOLO-seg produces COCO class labels.

    Parameters
    ----------
    model_size     : model path, e.g. "models/FastSAM-s.pt",
                     "models/efficient_sam_vitt.pt", or "models/yolo26s-seg.pt"
    conf_threshold : minimum detection confidence (0-1)
    device         : "cuda", "mps", or "cpu" (auto-detected)
    """

    def __init__(
        self,
        model_size: str = "models/FastSAM-s.pt",
        conf_threshold: float = 0.4,
        device: str = _DEFAULT_DEVICE,
    ) -> None:
        self.conf_threshold = conf_threshold
        self.device = device
        model_lower = model_size.lower()
        self._is_fastsam = "fastsam" in model_lower
        self._is_efficientsam = "efficient_sam" in model_lower or "efficientsam" in model_lower

        print(f"[ObjectDetector] Loading {model_size} on {device}…")

        if self._is_efficientsam:
            if not _HAS_EFFICIENT_SAM:
                raise ImportError(
                    "EfficientSAM requires: pip install "
                    "git+https://github.com/yformer/EfficientSAM.git "
                    "git+https://github.com/facebookresearch/segment-anything.git"
                )
            from efficient_sam.build_efficient_sam import build_efficient_sam
            if "vits" in model_lower:
                sam_model = build_efficient_sam(
                    encoder_patch_embed_dim=384, encoder_num_heads=6,
                    checkpoint=model_size,
                ).eval()
            else:
                sam_model = build_efficient_sam(
                    encoder_patch_embed_dim=192, encoder_num_heads=3,
                    checkpoint=model_size,
                ).eval()
            sam_model = sam_model.to(device)
            self._sam_model = sam_model
            self._sam_device = torch.device(device)
            self._points_per_side = 16
            self._iou_threshold = 0.7
            self.model = None
            # Warm-up
            dummy = np.zeros((640, 640, 3), dtype=np.uint8)
            self._detect_efficientsam(dummy)
            print("[ObjectDetector] EfficientSAM warm-up done.")
        elif self._is_fastsam:
            self.model = FastSAM(model_size)
            dummy = np.zeros((640, 640, 3), dtype=np.uint8)
            self.model.predict(dummy, device=device, verbose=False)
            print("[ObjectDetector] Warm-up done.")
        else:
            self.model = YOLO(model_size)
            dummy = np.zeros((640, 640, 3), dtype=np.uint8)
            self.model.predict(dummy, device=device, verbose=False)
            print("[ObjectDetector] Warm-up done.")

    # ── public ────────────────────────────────────────────────────────────────

    def detect(self, frame_bgr: np.ndarray) -> list[Detection]:
        """
        Run segmentation on frame_bgr and return detections.
        FastSAM/EfficientSAM: class-agnostic segments. YOLO: COCO-labeled detections.
        """
        if self._is_efficientsam:
            return self._detect_efficientsam(frame_bgr)
        return self._detect_ultralytics(frame_bgr)

    def _detect_ultralytics(self, frame_bgr: np.ndarray) -> list[Detection]:
        """FastSAM / YOLO-seg detection via ultralytics."""
        results = self.model.predict(
            frame_bgr,
            device=self.device,
            conf=self.conf_threshold,
            verbose=False,
            imgsz=640,
        )

        detections: list[Detection] = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for idx, box in enumerate(boxes):
                conf = float(box.conf[0])
                if conf < self.conf_threshold:
                    continue

                if self._is_fastsam:
                    label = f"segment_{idx}"
                else:
                    cls_id = int(box.cls[0])
                    label = result.names[cls_id]

                x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0

                # Extract segmentation mask polygon
                mask_poly = None
                if result.masks is not None and idx < len(result.masks.xy):
                    pts = result.masks.xy[idx]
                    if len(pts) >= 3:
                        mask_poly = pts.astype(np.int32).reshape((-1, 1, 2))

                detections.append(Detection(
                    label=label,
                    confidence=conf,
                    box_xyxy=(x1, y1, x2, y2),
                    box_center=(cx, cy),
                    mask_polygon=mask_poly,
                ))

        return detections

    def _detect_efficientsam(self, frame_bgr: np.ndarray) -> list[Detection]:
        """EfficientSAM segment-everything via grid point prompts."""
        from torchvision import transforms

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w = frame_rgb.shape[:2]

        # Prepare image tensor
        img_tensor = transforms.ToTensor()(frame_rgb).to(self._sam_device)
        img_batch = img_tensor.unsqueeze(0)  # [1, 3, H, W]

        # Generate grid of point prompts
        pts_per_side = self._points_per_side
        xs = np.linspace(0, w, pts_per_side + 2)[1:-1]
        ys = np.linspace(0, h, pts_per_side + 2)[1:-1]
        grid_points = np.array([[x, y] for y in ys for x in xs], dtype=np.float32)

        # Run inference in batches of points (one point per query)
        all_masks = []
        all_ious = []
        batch_size = 64  # points per forward pass
        for i in range(0, len(grid_points), batch_size):
            batch_pts = grid_points[i:i + batch_size]
            n = len(batch_pts)
            # Shape: [1, n, 1, 2] — 1 batch, n queries, 1 point each, xy
            pts_tensor = torch.tensor(batch_pts, dtype=torch.float32, device=self._sam_device)
            pts_tensor = pts_tensor.unsqueeze(0).unsqueeze(2)  # [1, n, 1, 2]
            labels = torch.ones(1, n, 1, dtype=torch.long, device=self._sam_device)

            with torch.no_grad():
                predicted_logits, predicted_iou = self._sam_model(
                    img_batch, pts_tensor, labels,
                )
            # predicted_logits: [1, n, num_masks, H, W]
            # predicted_iou: [1, n, num_masks]
            # Take the best mask per point (highest IoU)
            best_idx = predicted_iou[0].argmax(dim=-1)  # [n]
            for j in range(n):
                iou_val = float(predicted_iou[0, j, best_idx[j]])
                if iou_val < self._iou_threshold:
                    continue
                mask = (predicted_logits[0, j, best_idx[j]] > 0).cpu().numpy()
                all_masks.append(mask)
                all_ious.append(iou_val)

        if not all_masks:
            return []

        # NMS: deduplicate overlapping masks by IoU on binary masks
        detections = self._nms_masks(all_masks, all_ious, h, w)
        return detections

    def _nms_masks(
        self,
        masks: list[np.ndarray],
        ious: list[float],
        h: int, w: int,
        overlap_thresh: float = 0.5,
    ) -> list[Detection]:
        """Non-maximum suppression on binary masks, sorted by IoU score."""
        # Sort by IoU descending
        order = sorted(range(len(masks)), key=lambda i: ious[i], reverse=True)
        keep: list[int] = []
        suppressed = set()

        for i in order:
            if i in suppressed:
                continue
            keep.append(i)
            mask_i = masks[i]
            area_i = mask_i.sum()
            for j in order:
                if j <= i or j in suppressed:
                    continue
                intersection = (mask_i & masks[j]).sum()
                min_area = min(area_i, masks[j].sum())
                if min_area > 0 and intersection / min_area > overlap_thresh:
                    suppressed.add(j)

        detections: list[Detection] = []
        for idx, ki in enumerate(keep):
            mask = masks[ki]
            iou = ious[ki]
            if iou < self.conf_threshold:
                continue

            # Extract contour → polygon
            contours, _ = cv2.findContours(
                mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
            )
            mask_poly = None
            if contours:
                largest = max(contours, key=cv2.contourArea)
                if len(largest) >= 3:
                    mask_poly = largest.astype(np.int32).reshape((-1, 1, 2))

            # Bounding box from mask
            ys_nz, xs_nz = np.where(mask)
            if len(xs_nz) == 0:
                continue
            x1, y1 = float(xs_nz.min()), float(ys_nz.min())
            x2, y2 = float(xs_nz.max()), float(ys_nz.max())
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0

            detections.append(Detection(
                label=f"segment_{idx}",
                confidence=iou,
                box_xyxy=(x1, y1, x2, y2),
                box_center=(cx, cy),
                mask_polygon=mask_poly,
            ))

        return detections

    # ── probabilistic hit-test ───────────────────────────────────────────────

    MIN_SIGMA = 12.0
    MAX_SIGMA = 60.0
    MIN_HIT_PROBABILITY = 0.05

    @staticmethod
    def probabilistic_hit_test(
        detections: list[Detection],
        gx: float,
        gy: float,
        sigma: float = 30.0,
        frame_shape: tuple[int, int] = (1200, 1600),
    ) -> Optional[GazeHit]:
        """
        Bayesian gaze-to-object inference using a 2D Gaussian gaze model.

        Models gaze uncertainty as a Gaussian centered at (gx, gy) with the
        given sigma (derived from fixation drift).  Computes the overlap of
        each detection mask with the Gaussian, weighted by an inverse-sqrt-area
        prior that favours small objects near the gaze centre.

        Returns the best-matching GazeHit, or None if no detection exceeds
        the minimum probability threshold.
        """
        if not detections:
            return None

        sigma = max(ObjectDetector.MIN_SIGMA, min(sigma, ObjectDetector.MAX_SIGMA))
        h, w = frame_shape[:2]

        # Build local patch around gaze (3*sigma captures 99.7% of mass)
        radius = int(math.ceil(3.0 * sigma))
        x0 = max(int(gx) - radius, 0)
        y0 = max(int(gy) - radius, 0)
        x1 = min(int(gx) + radius, w)
        y1 = min(int(gy) + radius, h)
        ph, pw = y1 - y0, x1 - x0
        if ph <= 0 or pw <= 0:
            return None

        # 2D Gaussian weight array over the patch
        yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
        gauss = np.exp(-((xx - gx) ** 2 + (yy - gy) ** 2) / (2.0 * sigma * sigma))
        gauss_sum = gauss.sum()
        if gauss_sum < 1e-9:
            return None

        scored: list[tuple[Detection, float]] = []

        for det in detections:
            # Quick-reject: bbox doesn't overlap the patch
            bx1, by1, bx2, by2 = det.box_xyxy
            if bx2 < x0 or bx1 > x1 or by2 < y0 or by1 > y1:
                continue

            # Rasterize mask into patch-local buffer
            mask_buf = np.zeros((ph, pw), dtype=np.uint8)
            if det.mask_polygon is not None:
                shifted = det.mask_polygon.copy()
                shifted[:, :, 0] -= x0
                shifted[:, :, 1] -= y0
                cv2.fillPoly(mask_buf, [shifted], 1)
            else:
                # Fallback: rasterize bounding box
                lx1 = max(int(bx1) - x0, 0)
                ly1 = max(int(by1) - y0, 0)
                lx2 = min(int(bx2) - x0, pw)
                ly2 = min(int(by2) - y0, ph)
                mask_buf[ly1:ly2, lx1:lx2] = 1

            # Likelihood: fraction of Gaussian mass captured by this mask
            likelihood = float(np.sum(gauss * mask_buf)) / gauss_sum
            if likelihood < 1e-6:
                continue

            # Prior: inverse sqrt of bbox area (favours smaller objects)
            area = max((bx2 - bx1) * (by2 - by1), 1.0)
            prior = 1.0 / math.sqrt(area)

            scored.append((det, likelihood * prior))

        if not scored:
            return None

        # Normalize to probabilities
        total = sum(s for _, s in scored)
        if total < 1e-9:
            return None
        scored = [(d, s / total) for d, s in scored]
        scored.sort(key=lambda x: x[1], reverse=True)

        best_det, best_prob = scored[0]
        if best_prob < ObjectDetector.MIN_HIT_PROBABILITY:
            return None

        return GazeHit(
            detection=best_det,
            probability=best_prob,
            candidates=scored,
        )

    # ── legacy point hit-test ────────────────────────────────────────────────

    @staticmethod
    def hit_test(
        detections: list[Detection],
        gx: float,
        gy: float,
    ) -> Optional[Detection]:
        """
        Return the Detection whose mask or bounding box contains (gx, gy).

        Prefers mask polygon test (accurate for irregular shapes).
        Falls back to bounding box if no mask is available.
        """
        point = (float(gx), float(gy))
        candidates = []

        for d in detections:
            # Prefer mask polygon test
            if d.mask_polygon is not None:
                if cv2.pointPolygonTest(d.mask_polygon, point, False) >= 0:
                    candidates.append(d)
            else:
                # Fallback to bounding box
                if (d.box_xyxy[0] <= gx <= d.box_xyxy[2]
                        and d.box_xyxy[1] <= gy <= d.box_xyxy[3]):
                    candidates.append(d)

        if not candidates:
            return None

        # Return smallest area segment (most specific)
        return min(
            candidates,
            key=lambda d: (d.box_xyxy[2] - d.box_xyxy[0]) * (d.box_xyxy[3] - d.box_xyxy[1]),
        )
