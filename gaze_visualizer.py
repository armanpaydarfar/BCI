#!/usr/bin/env python3
# gaze_visualizer.py
#
# Simple real-time gaze visualizer:
#   - Connects to Neon LSL gaze stream (type='Gaze')
#   - Shows a blank window with a red dot at the current gaze location
#   - Uses the same normalization as harmony_gaze_calibration.py:
#       x_norm = sample[0] / 1600
#       y_norm = sample[1] / 1200
#   - Press ESC or close window to exit.

import sys
import time
import pygame
import numpy as np
from pylsl import StreamInlet, resolve_stream

# Screen where we draw the red dot
SCREEN_W, SCREEN_H = 800, 600

# These should match the calibration assumptions
GAZE_SAMPLE_WIDTH  = 1600.0
GAZE_SAMPLE_HEIGHT = 1200.0
GAZE_CONFIDENCE_THRESHOLD = 0.7

# Simple smoothing so the dot doesn’t jitter too much
SMOOTH_ALPHA = 0.5  # 0 = no update, 1 = no smoothing

def resolve_gaze_stream():
    print("[GAZE] Resolving LSL stream (type='Gaze')...")
    streams = resolve_stream('type', 'Gaze')
    if not streams:
        print("[GAZE] ERROR: No gaze stream found.")
        sys.exit(1)
    inlet = StreamInlet(streams[0])
    print(f"[GAZE] Connected to stream: {streams[0].name()}")
    return inlet

def main():
    inlet = resolve_gaze_stream()

    pygame.init()
    screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
    pygame.display.set_caption("Gaze Visualizer")
    clock = pygame.time.Clock()

    # Start dot in the center
    x_vis = SCREEN_W / 2.0
    y_vis = SCREEN_H / 2.0

    running = True
    while running:
        # Handle quit / ESC
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False

        # Non-blocking pull from LSL (most recent sample)
        sample, _ = inlet.pull_sample(timeout=0.0)

        if sample:
            try:
                x_raw = float(sample[0])
                y_raw = float(sample[1])
                conf  = float(sample[15])
            except Exception:
                x_raw = y_raw = conf = None

            if conf is not None and conf >= GAZE_CONFIDENCE_THRESHOLD:
                # Normalize to [0,1] using same scheme as calibration
                x_norm = x_raw / GAZE_SAMPLE_WIDTH
                y_norm = y_raw / GAZE_SAMPLE_HEIGHT

                # Clamp in case of slight overshoots
                x_norm = max(0.0, min(1.0, x_norm))
                y_norm = max(0.0, min(1.0, y_norm))

                # Map to screen coordinates
                x_target = x_norm * SCREEN_W
                y_target = y_norm * SCREEN_H  # 0 = top, 1 = bottom

                # Exponential smoothing
                x_vis = SMOOTH_ALPHA * x_target + (1.0 - SMOOTH_ALPHA) * x_vis
                y_vis = SMOOTH_ALPHA * y_target + (1.0 - SMOOTH_ALPHA) * y_vis

        # Draw
        screen.fill((0, 0, 0))  # black background
        pygame.draw.circle(screen, (255, 0, 0), (int(x_vis), int(y_vis)), 10)
        pygame.display.flip()

        # Limit to ~60 FPS
        clock.tick(60)

    pygame.quit()
    print("Gaze visualizer closed.")

if __name__ == "__main__":
    main()
