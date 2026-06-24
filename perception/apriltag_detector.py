# Vendored from harmony_vlm (https://github.com/vivianchen98/harmony_vlm) @ cfa01b6
# by Vivian Chen. Folded into the BCI repo for WS3 (2026-06-15). Edit here, not
# upstream; see Documents/SoftwareDocs/projects/harmony-bci/vlm-integration/.
# STAGED — not import-safe in this env (deps deliberately excluded); see the
# live-vs-staged list in perception/__init__.py before importing.
"""
apriltag_detector.py — AprilTag pose estimation using dt-apriltags.

Camera-agnostic: accepts (fx, fy, cx, cy) directly so it works with both
Pupil Core and RealSense cameras.  Uses 4x4 homogeneous transforms internally.
"""

from __future__ import annotations

import cv2
import dt_apriltags
import numpy as np


def pose_to_matrix(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    """Convert rotation (3x3) + translation (3x1 or 3,) to 4x4 homogeneous matrix."""
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = rotation
    T[:3, 3] = np.asarray(translation).ravel()
    return T


class AprilTagDetector:
    """AprilTag detector with 4x4 homogeneous pose output."""

    def __init__(
        self,
        camera_params: tuple[float, float, float, float],
        tag_size: float = 0.08,
        family: str = "tag36h11",
    ):
        """
        Parameters
        ----------
        camera_params : (fx, fy, cx, cy)
        tag_size      : physical tag size in meters
        family        : AprilTag family (default: tag36h11)
        """
        self._detector = dt_apriltags.Detector(families=family)
        self.camera_params = camera_params
        self.tag_size = tag_size

    def detect(self, image: np.ndarray) -> list:
        """Run detection with pose estimation.  Accepts BGR or grayscale."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        return self._detector.detect(
            gray,
            estimate_tag_pose=True,
            camera_params=self.camera_params,
            tag_size=self.tag_size,
        )

    def get_T_cam_from_tag(self, image: np.ndarray) -> np.ndarray | None:
        """Detect first tag and return T_camera_from_tag (4x4).

        Returns None if no tag found.
        """
        results = self.detect(image)
        if not results:
            return None
        r = results[0]
        return pose_to_matrix(r.pose_R, r.pose_t)

    def get_T_world_from_camera(
        self,
        image: np.ndarray,
        T_world_from_tag: np.ndarray = np.eye(4),
    ) -> np.ndarray | None:
        """Compute camera-to-world transform via an AprilTag.

        Parameters
        ----------
        image            : camera image (BGR or grayscale)
        T_world_from_tag : 4x4 known pose of the tag in world frame

        Returns
        -------
        T_world_from_camera (4x4) or None if tag not visible
        """
        T_cam_from_tag = self.get_T_cam_from_tag(image)
        if T_cam_from_tag is None:
            return None
        # T_world_from_cam = T_world_from_tag @ inv(T_cam_from_tag)
        T_tag_from_cam = np.linalg.inv(T_cam_from_tag)
        return T_world_from_tag @ T_tag_from_cam
