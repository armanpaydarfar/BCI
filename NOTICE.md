# NOTICE — Third-party attribution

## perception/ — vendored from harmony_vlm

The `perception/` package in this repository is vendored from **harmony_vlm**, a
codebase authored by **Vivian Chen** (https://github.com/vivianchen98/harmony_vlm).
It provides the gaze/vision perception stack — class-agnostic segmentation
(FastSAM), metric depth (Depth Pro), gaze-fixation detection, Gemini-backed VLM
intent reasoning, and the Pupil Labs Neon readers.

- **Source commit:** `cfa01b6` ("feat: add configurable thinking budget for Gemini
  models and implement latency tracking", 2026-05-25).
- **Folded in:** WS3, 2026-06-15. The upstream `utils/` tree was copied into
  `perception/` essentially verbatim; the only local changes are import-path
  rewrites (`utils.` → `perception.`) and the per-file provenance headers. Edit the
  code here, not upstream — see the header on each `perception/*.py` file and
  `Documents/SoftwareDocs/projects/harmony-bci/vlm-integration/`.
- **Permission:** used with the author's permission. harmony_vlm carries no license
  of its own, so this attribution and the author's consent are the basis for reuse;
  the per-file headers preserve authorship.

`perception/realsense_camera.py` is, in turn, adapted upstream (by harmony_vlm) from
`realsense-utils-dev-main/realsense/camera.py`; that note is carried verbatim in the
file.

All other code in this repository is part of the Harmony BCI project unless a file
header states otherwise.
