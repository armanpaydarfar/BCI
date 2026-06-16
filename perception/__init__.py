# Vendored from harmony_vlm (https://github.com/vivianchen98/harmony_vlm) @ cfa01b6
# by Vivian Chen. Folded into the BCI repo for WS3 (2026-06-15). Edit here, not
# upstream; see Documents/SoftwareDocs/projects/harmony-bci/vlm-integration/.
"""BCI perception subsystem — segmentation, metric depth, gaze fixation, and
VLM intent reasoning, vendored from harmony_vlm.

This package intentionally performs no eager submodule imports: import the
specific module you need (e.g. ``from perception.object_detector import
ObjectDetector``) so that importing ``perception`` never pulls in a module
whose dependency is not installed.

Module status
-------------
**Live** — wired into the running perception service (``vlm_service.py`` /
``Utils/frame_relay.py``) and importable in the unified ``environment.yml``:

    object_detector, depth_estimator, fixation_detector, intent_reasoner,
    pupil_reader, visualize_neon, neon/

**Staged** — vendored as a unit for the imminent WS4/WS5 work, but **not yet
importable** in the current env because their dependencies are deliberately
excluded from ``environment.yml`` (see the WS3 decision doc, §2.1 "Env
consequence"). Do not import these on the live path until their workstream
adds the dependency:

    apriltag_detector   needs an AprilTag lib — WS5 will replace dt_apriltags
                        (dropped: no win/py3.12 wheel) with pupil-apriltags
    realsense_camera    needs pyrealsense2 (no RealSense on the current hosts)
    gaze_grounder       needs apriltag_detector + realsense_camera
    overlay_renderer    cv2 segmentation/state overlay — a WS4 candidate
    exo_controller      upstream's monolithic demo loop, superseded here by the
                        vlm_service.py UDP architecture; kept for reference
    core/               Pupil Labs *Core* reader (we use Neon, not Core)
"""
