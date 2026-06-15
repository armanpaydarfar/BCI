# Vendored from harmony_vlm (https://github.com/vivianchen98/harmony_vlm) @ cfa01b6
# by Vivian Chen. Folded into the BCI repo for WS3 (2026-06-15). Edit here, not
# upstream; see Documents/SoftwareDocs/projects/harmony-bci/vlm-integration/.
"""
Neon Eye Tracker - Gaze & Eye State Visualization
Parses gaze + eye state data, syncs to scene camera video, renders overlay.
Usage:
    python visualize_neon.py [recording_dir] [--out output.mp4]
"""

import argparse
import json
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm

# ── dtype definitions ─────────────────────────────────────────────────────────

GAZE_DTYPE = np.dtype([("x", "<f4"), ("y", "<f4")])

EYE_STATE_DTYPE = np.dtype([
    ("pupil_diameter_left_mm",   "<f4"),
    ("eyeball_center_left_x",    "<f4"),
    ("eyeball_center_left_y",    "<f4"),
    ("eyeball_center_left_z",    "<f4"),
    ("optical_axis_left_x",      "<f4"),
    ("optical_axis_left_y",      "<f4"),
    ("optical_axis_left_z",      "<f4"),
    ("pupil_diameter_right_mm",  "<f4"),
    ("eyeball_center_right_x",   "<f4"),
    ("eyeball_center_right_y",   "<f4"),
    ("eyeball_center_right_z",   "<f4"),
    ("optical_axis_right_x",     "<f4"),
    ("optical_axis_right_y",     "<f4"),
    ("optical_axis_right_z",     "<f4"),
    ("eyelid_angle_top_left",    "<f4"),
    ("eyelid_angle_bottom_left", "<f4"),
    ("eyelid_aperture_left_mm",  "<f4"),
    ("eyelid_angle_top_right",   "<f4"),
    ("eyelid_angle_bottom_right","<f4"),
    ("eyelid_aperture_right_mm", "<f4"),
])

# ── loaders ───────────────────────────────────────────────────────────────────

def load_stream(rec: Path, name: str, dtype: np.dtype):
    """Load a .raw binary stream and its paired .time file (int64 ns)."""
    raw_path  = next(rec.glob(f"{name} ps1.raw"))
    time_path = next(rec.glob(f"{name} ps1.time"))
    data = np.frombuffer(raw_path.read_bytes(), dtype=dtype)
    ts   = np.frombuffer(time_path.read_bytes(), dtype="<i8")
    assert len(data) == len(ts), f"Length mismatch for {name}"
    return data, ts


def load_video_timestamps(rec: Path):
    """Load scene camera frame timestamps (int64 ns)."""
    time_path = rec / "Neon Scene Camera v1 ps1.time"
    return np.frombuffer(time_path.read_bytes(), dtype="<i8")


def nearest_idx(timestamps: np.ndarray, query_ts: int) -> int:
    """Return index of nearest timestamp to query_ts."""
    idx = np.searchsorted(timestamps, query_ts)
    if idx == 0:
        return 0
    if idx == len(timestamps):
        return len(timestamps) - 1
    before = timestamps[idx - 1]
    after  = timestamps[idx]
    return idx - 1 if (query_ts - before) <= (after - query_ts) else idx

# ── drawing helpers ───────────────────────────────────────────────────────────

FONT      = cv2.FONT_HERSHEY_SIMPLEX
COL_GAZE  = (0, 255, 80)      # bright green
COL_LEFT  = (80, 160, 255)    # blue-ish  (left eye)
COL_RIGHT = (255, 160, 80)    # orange    (right eye)
COL_TEXT  = (240, 240, 240)
COL_SHADOW= (0, 0, 0)

def shadow_text(img, text, pos, scale=0.55, thickness=1, color=COL_TEXT):
    x, y = pos
    cv2.putText(img, text, (x+1, y+1), FONT, scale, COL_SHADOW, thickness+1, cv2.LINE_AA)
    cv2.putText(img, text, (x, y),     FONT, scale, color,      thickness,   cv2.LINE_AA)


def draw_gaze_cursor(img, gx: float, gy: float):
    """Draw crosshair + concentric circles at gaze point."""
    cx, cy = int(round(gx)), int(round(gy))
    h, w   = img.shape[:2]
    if not (0 <= cx < w and 0 <= cy < h):
        return
    # outer ring
    cv2.circle(img, (cx, cy), 28, COL_GAZE, 2, cv2.LINE_AA)
    cv2.circle(img, (cx, cy), 12, COL_GAZE, 2, cv2.LINE_AA)
    # centre dot
    cv2.circle(img, (cx, cy),  3, COL_GAZE, -1, cv2.LINE_AA)
    # crosshair lines
    cv2.line(img, (cx - 40, cy), (cx - 14, cy), COL_GAZE, 1, cv2.LINE_AA)
    cv2.line(img, (cx + 14, cy), (cx + 40, cy), COL_GAZE, 1, cv2.LINE_AA)
    cv2.line(img, (cx, cy - 40), (cx, cy - 14), COL_GAZE, 1, cv2.LINE_AA)
    cv2.line(img, (cx, cy + 14), (cx, cy + 40), COL_GAZE, 1, cv2.LINE_AA)


def draw_hud(img, gaze_sample, eye_sample, frame_idx: int,
             total_frames: int, rec_duration_s: float):
    h, w = img.shape[:2]
    pad  = 14

    # ── semi-transparent HUD panel (bottom-left) ──────────────────────────────
    panel_h = 170
    panel_w = 340
    overlay = img.copy()
    cv2.rectangle(overlay, (0, h - panel_h), (panel_w, h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)

    gx, gy = float(gaze_sample["x"]), float(gaze_sample["y"])
    pd_l   = float(eye_sample["pupil_diameter_left_mm"])
    pd_r   = float(eye_sample["pupil_diameter_right_mm"])
    ea_l   = float(eye_sample["eyelid_aperture_left_mm"])
    ea_r   = float(eye_sample["eyelid_aperture_right_mm"])

    t_elapsed = (frame_idx / total_frames) * rec_duration_s

    base_y = h - panel_h + 22
    dy     = 24

    shadow_text(img, "NEON EYE TRACKER",        (pad, base_y),        0.52, 1, (180, 220, 255))
    shadow_text(img, f"Gaze  ({gx:6.1f}, {gy:6.1f}) px", (pad, base_y + dy),     0.50)
    shadow_text(img, f"Pupil  L {pd_l:.2f} mm   R {pd_r:.2f} mm",
                                                 (pad, base_y + 2*dy), 0.50)
    shadow_text(img, f"Eyelid L {ea_l:.2f} mm   R {ea_r:.2f} mm",
                                                 (pad, base_y + 3*dy), 0.50)
    shadow_text(img, f"t = {t_elapsed:5.2f} s / {rec_duration_s:.2f} s",
                                                 (pad, base_y + 4*dy), 0.50)
    shadow_text(img, f"Frame {frame_idx+1}/{total_frames}",
                                                 (pad, base_y + 5*dy), 0.48, 1, (160, 160, 160))

    # ── pupil diameter bar chart (bottom-right) ───────────────────────────────
    bar_x   = w - 180
    bar_top = h - panel_h + 14
    bar_h   = 100
    max_pd  = 9.0   # mm — typical max
    for label, val, col, bx in [
        ("L pupil", pd_l, COL_LEFT,  bar_x),
        ("R pupil", pd_r, COL_RIGHT, bar_x + 75),
    ]:
        filled = int(bar_h * min(val / max_pd, 1.0))
        cv2.rectangle(img, (bx, bar_top),             (bx+55, bar_top+bar_h), (50,50,50), -1)
        cv2.rectangle(img, (bx, bar_top+bar_h-filled),(bx+55, bar_top+bar_h), col,        -1)
        cv2.rectangle(img, (bx, bar_top),             (bx+55, bar_top+bar_h), (120,120,120), 1)
        shadow_text(img, label,         (bx, bar_top + bar_h + 16), 0.40)
        shadow_text(img, f"{val:.1f}mm",(bx, bar_top + bar_h + 30), 0.40)

    # ── timeline scrubber (very bottom) ───────────────────────────────────────
    tl_y   = h - 6
    tl_x0  = 0
    tl_x1  = w
    tl_fill= int(w * frame_idx / max(total_frames - 1, 1))
    cv2.line(img, (tl_x0, tl_y), (tl_x1, tl_y), (60, 60, 60), 4)
    cv2.line(img, (tl_x0, tl_y), (tl_fill, tl_y), COL_GAZE,   4)

# ── main ──────────────────────────────────────────────────────────────────────

def process_recording(rec_dir: Path, out_path: Path):
    print(f"\n{'='*60}")
    print(f"Recording: {rec_dir.name}")

    # load metadata
    info = json.loads((rec_dir / "info.json").read_text())
    duration_s = info["duration"] / 1e9
    print(f"Duration:  {duration_s:.2f} s")

    # load gaze & eye state
    gaze,      gaze_ts = load_stream(rec_dir, "gaze",      GAZE_DTYPE)
    eye_state, eye_ts  = load_stream(rec_dir, "eye_state", EYE_STATE_DTYPE)
    # both share the same timestamps; verify
    assert np.array_equal(gaze_ts, eye_ts), "gaze/eye_state timestamps differ"
    data_ts = gaze_ts
    print(f"Gaze samples:      {len(gaze)}")
    print(f"Eye-state samples: {len(eye_state)}")

    # load scene camera
    video_path = rec_dir / "Neon Scene Camera v1 ps1.mp4"
    frame_ts   = load_video_timestamps(rec_dir)
    cap        = cv2.VideoCapture(str(video_path))
    fps        = cap.get(cv2.CAP_PROP_FPS)
    n_frames   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W          = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H          = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video: {n_frames} frames @ {fps:.2f} fps, {W}x{H}")
    print(f"Frame timestamps: {len(frame_ts)} entries")

    # output writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (W, H))

    for fi in tqdm(range(n_frames), desc="Rendering"):
        ok, frame = cap.read()
        if not ok:
            break

        # sync: find nearest gaze/eye-state sample for this frame
        fts  = frame_ts[fi] if fi < len(frame_ts) else frame_ts[-1]
        didx = nearest_idx(data_ts, fts)

        draw_gaze_cursor(frame, gaze[didx]["x"], gaze[didx]["y"])
        draw_hud(frame, gaze[didx], eye_state[didx], fi, n_frames, duration_s)

        writer.write(frame)

    cap.release()
    writer.release()
    print(f"Saved → {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Neon gaze visualization")
    parser.add_argument("recordings", nargs="*",
                        default=[
                            "neon/QuickShare_2603111547",
                            "neon/QuickShare_2603111549",
                        ],
                        help="Recording directory/ies (default: both demos)")
    parser.add_argument("--out-dir", default="neon_output",
                        help="Output directory for rendered videos")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for rec_str in args.recordings:
        rec_dir  = Path(rec_str)
        out_path = out_dir / f"{rec_dir.name}_gaze.mp4"
        process_recording(rec_dir, out_path)

    print(f"\nDone. Videos saved to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
