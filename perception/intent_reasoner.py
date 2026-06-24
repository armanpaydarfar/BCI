# Vendored from harmony_vlm (https://github.com/vivianchen98/harmony_vlm) @ cfa01b6
# by Vivian Chen. Folded into the BCI repo for WS3 (2026-06-15). Edit here, not
# upstream; see Documents/SoftwareDocs/projects/harmony-bci/vlm-integration/.
"""
intent_reasoner.py — Async VLM intent reasoning for the exoskeleton.

Always uses vision: sends annotated scene frames with gaze markers to the VLM.
The VLM identifies objects (open vocabulary) and infers user intent.
"""

from __future__ import annotations

import base64
import json
import sys
import traceback
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Optional

import cv2
import numpy as np

from perception.fixation_detector import FixationState
from perception.pupil_reader import GazeSample

# ── constants ─────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are an assistive intent reasoner for an exoskeleton worn by a person "
    "with limited mobility. The user is looking at something in the scene. "
    "A bright magenta crosshair marks where the user is gazing. "
    "Colored translucent overlays show detected object segments in the scene. "
    "Identify the object at the gaze point. "
    "Propose exactly 5 possible interactions the user might want, "
    "ranked from most to least likely. Each candidate should have an intent "
    "description and a confidence score (0-1, summing to 1.0). "
    "Respond ONLY with valid JSON matching the provided schema. "
    "Never issue motion commands. "
    "Provide a clarification_question based on the top candidate — "
    "natural, short, and specific."
)

PAIR_SYSTEM_PROMPT = (
    "You are an assistive intent reasoner for an exoskeleton worn by a person "
    "with limited mobility. The user looked at two locations sequentially. "
    "Marker '1' (magenta) shows where they looked first. "
    "Marker '2' (cyan) shows where they looked second. "
    "Colored translucent overlays show detected object segments. "
    "Identify both objects. "
    "Propose exactly 5 possible interactions involving both objects, "
    "ranked from most to least likely (e.g., pick-and-place, use one as a tool "
    "on the other, move one near the other). Each candidate should have an intent "
    "description and a confidence score (0-1, summing to 1.0). "
    "Respond ONLY with valid JSON matching the provided schema. "
    "Never issue motion commands. "
    "Provide a clarification_question based on the top candidate — "
    "natural, short, and specific."
)

JSON_SCHEMA_DESCRIPTION = """\
Respond with a JSON object with these fields:
{
  "object": <string — name of the first/primary target object>,
  "second_object": <string — name of the second object, or null if single-object>,
  "candidates": [
    {"intent": <string — action description>, "confidence": <float 0-1>},
    {"intent": <string>, "confidence": <float>},
    {"intent": <string>, "confidence": <float>},
    {"intent": <string>, "confidence": <float>},
    {"intent": <string>, "confidence": <float>}
  ],
  "clarification_question": <string — question based on top candidate>,
  "reasoning": <string — brief explanation of ranking>
}
The candidates array must have exactly 5 entries, ranked from most to least likely.
Confidence scores should sum to approximately 1.0."""

FALLBACK_RESPONSE: dict = {
    "object": None,
    "second_object": None,
    "candidates": [
        {"intent": "unknown", "confidence": 0.2},
        {"intent": "unknown", "confidence": 0.2},
        {"intent": "unknown", "confidence": 0.2},
        {"intent": "unknown", "confidence": 0.2},
        {"intent": "unknown", "confidence": 0.2},
    ],
    "clarification_question": "I couldn't determine your intent. What would you like to do?",
    "reasoning": "API call failed or returned unparseable output.",
}


# ── frame annotation ─────────────────────────────────────────────────────────

def _annotate_gaze(
    frame_bgr: np.ndarray,
    gx: float,
    gy: float,
    label: str | None = None,
    color: tuple = (255, 0, 255),
) -> np.ndarray:
    """Draw a bright crosshair + circle at the gaze point. Returns a copy."""
    img = frame_bgr.copy()
    gxi, gyi = int(round(gx)), int(round(gy))

    # Crosshair
    cv2.drawMarker(img, (gxi, gyi), color, cv2.MARKER_CROSS, 50, 3)
    cv2.circle(img, (gxi, gyi), 25, color, 3)

    # Label
    if label:
        cv2.putText(
            img, label, (gxi + 30, gyi - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 5,
        )
        cv2.putText(
            img, label, (gxi + 30, gyi - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3,
        )

    return img


def _draw_segments(frame_bgr: np.ndarray, detections: list) -> np.ndarray:
    """Draw SoM-style segmentation: translucent masks + numbered labels at centroids."""
    img = frame_bgr.copy()
    # Distinct colors for each segment (BGR)
    COLORS = [
        (255, 100, 100), (100, 255, 100), (100, 100, 255),
        (255, 255, 100), (255, 100, 255), (100, 255, 255),
        (200, 150, 50), (50, 200, 150), (150, 50, 200),
        (200, 200, 200),
    ]
    overlay = img.copy()
    for i, det in enumerate(detections):
        color = COLORS[i % len(COLORS)]
        if det.mask_polygon is not None:
            cv2.fillPoly(overlay, [det.mask_polygon], color)
            cv2.polylines(img, [det.mask_polygon], True, color, 2, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.3, img, 0.7, 0, img)

    # SoM numbered labels at each segment centroid
    for i, det in enumerate(detections):
        cx, cy = int(det.box_center[0]), int(det.box_center[1])
        color = COLORS[i % len(COLORS)]
        label = str(i)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        # Dark background pill
        cv2.rectangle(
            img,
            (cx - tw // 2 - 4, cy - th // 2 - 4),
            (cx + tw // 2 + 4, cy + th // 2 + 4),
            (0, 0, 0), -1,
        )
        # Colored number
        cv2.putText(
            img, label,
            (cx - tw // 2, cy + th // 2),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA,
        )

    return img


def _crop_around_gaze(
    frame_bgr: np.ndarray,
    gx: float,
    gy: float,
    crop_size: int = 200,
) -> np.ndarray:
    """Extract a square crop from the raw frame centered on the gaze point.

    No segmentation overlays — gives the VLM a clean close-up of what's
    actually at the gaze location.  Returns an image with a small crosshair.
    """
    h, w = frame_bgr.shape[:2]
    half = crop_size // 2
    cx, cy = int(round(gx)), int(round(gy))

    x1 = max(cx - half, 0)
    y1 = max(cy - half, 0)
    x2 = min(cx + half, w)
    y2 = min(cy + half, h)

    crop = frame_bgr[y1:y2, x1:x2].copy()

    # Draw a small crosshair at the gaze point relative to the crop
    lx, ly = cx - x1, cy - y1
    cv2.drawMarker(crop, (lx, ly), (255, 0, 255), cv2.MARKER_CROSS, 20, 2)
    cv2.circle(crop, (lx, ly), 10, (255, 0, 255), 2)

    return crop


def _encode_jpeg(bgr: np.ndarray, quality: int = 80) -> str:
    """Encode a BGR image as JPEG and return base64 string."""
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("JPEG encode failed")
    return base64.b64encode(buf.tobytes()).decode("ascii")


# ── reasoner ─────────────────────────────────────────────────────────────────

class IntentReasoner:
    """
    Wraps a VLM (Gemini or GPT-4o) to reason about gaze-based user intent.
    Always uses vision — sends annotated scene frames.

    Model selection:
      - "gemini-2.5-pro" → Google Gemini via google-genai (uses GOOGLE_API_KEY)
      - "gpt-4o" etc.    → OpenAI (uses OPENAI_API_KEY)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o",
        max_tokens: int = 1024,
        thinking_budget: Optional[int] = None,
    ) -> None:
        self.model = model
        self._backend = "gemini" if "gemini" in model.lower() else "openai"
        self.thinking_budget = thinking_budget
        if self._backend == "gemini" and max_tokens == 1024:
            if thinking_budget != 0:
                max_tokens = 2048  # Increase to avoid truncation due to thinking tokens
        self.max_tokens = max_tokens

        if self._backend == "gemini":
            from google import genai
            kwargs = {}
            if api_key:
                kwargs["api_key"] = api_key
            self._gemini = genai.Client(**kwargs)
            print(f"[IntentReasoner] Using Gemini backend: {model}")
        else:
            from openai import OpenAI
            kwargs = {}
            if api_key:
                kwargs["api_key"] = api_key
            self._openai = OpenAI(**kwargs)
            print(f"[IntentReasoner] Using OpenAI backend: {model}")

        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="intent")
        self._session_vlm_log: Path | None = None  # set by ExoController
        self._session_dir: Path | None = None  # set by ExoController
        self._vlm_call_count: int = 0

    # ── public ────────────────────────────────────────────────────────────────

    def reason_async(
        self,
        gaze: GazeSample,
        fixation: FixationState,
        frame_bgr: np.ndarray,
        detections: list | None = None,
        gaze_hit_info: str = "",
    ) -> Future:
        """Single-object reasoning: annotated frame with gaze marker + segments."""
        return self._executor.submit(
            self._reason_single, gaze, fixation, frame_bgr, list(detections or []),
            gaze_hit_info,
        )

    def reason_async_pair(
        self,
        first_gaze: tuple[float, float],
        second_gaze: tuple[float, float],
        first_fix: FixationState,
        second_fix: FixationState,
        first_frame_bgr: np.ndarray,
        second_frame_bgr: np.ndarray,
        first_detections: list | None = None,
        second_detections: list | None = None,
        gaze_hit_info: str = "",
    ) -> Future:
        """Two-object reasoning: two frames with segments + gaze markers."""
        return self._executor.submit(
            self._reason_pair, first_gaze, second_gaze, first_fix, second_fix,
            first_frame_bgr, second_frame_bgr,
            list(first_detections or []), list(second_detections or []),
            gaze_hit_info,
        )

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False)

    # ── single-object path ────────────────────────────────────────────────────

    def _reason_single(
        self,
        gaze: GazeSample,
        fixation: FixationState,
        frame_bgr: np.ndarray,
        detections: list = [],
        gaze_hit_info: str = "",
    ) -> dict:
        try:
            duration_ms = fixation.duration_ns / 1_000_000
            annotated = _draw_segments(frame_bgr, detections)
            annotated = _annotate_gaze(annotated, gaze.x, gaze.y)
            full_small = cv2.resize(annotated, (800, 600))
            full_b64 = _encode_jpeg(full_small, quality=75)

            # Close-up crop from the raw frame (no overlays)
            crop = _crop_around_gaze(frame_bgr, gaze.x, gaze.y)
            crop_b64 = _encode_jpeg(crop, quality=90)

            user_content = [
                {
                    "type": "text",
                    "text": (
                        f"The user has been fixating at the magenta crosshair "
                        f"for {duration_ms:.0f} ms. "
                        f"Image 1 is the full scene with segmentation overlays. "
                        f"Image 2 is a close-up crop around the gaze point (no overlays) "
                        f"for precise object identification. "
                        f"Identify the object at the crosshair and determine intent.\n\n"
                        f"{JSON_SCHEMA_DESCRIPTION}"
                    ),
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{full_b64}", "detail": "low"},
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{crop_b64}", "detail": "high"},
                },
            ]

            return self._call_api(
                [{"role": "user", "content": user_content}],
                system_prompt=SYSTEM_PROMPT,
                gaze_hit_info=gaze_hit_info,
            )
        except Exception:
            traceback.print_exc()
            return dict(FALLBACK_RESPONSE)

    # ── two-object path ───────────────────────────────────────────────────────

    def _reason_pair(
        self,
        first_gaze: tuple[float, float],
        second_gaze: tuple[float, float],
        first_fix: FixationState,
        second_fix: FixationState,
        first_frame_bgr: np.ndarray,
        second_frame_bgr: np.ndarray,
        first_detections: list = [],
        second_detections: list = [],
        gaze_hit_info: str = "",
    ) -> dict:
        try:
            dur1_ms = first_fix.duration_ns / 1_000_000
            dur2_ms = second_fix.duration_ns / 1_000_000

            # Draw segments then gaze marker on each frame
            frame1 = _draw_segments(first_frame_bgr, first_detections)
            frame1 = _annotate_gaze(
                frame1, first_gaze[0], first_gaze[1],
                label="1", color=(255, 0, 255),  # magenta
            )
            frame2 = _draw_segments(second_frame_bgr, second_detections)
            frame2 = _annotate_gaze(
                frame2, second_gaze[0], second_gaze[1],
                label="2", color=(255, 255, 0),  # cyan (BGR)
            )

            img1_small = cv2.resize(frame1, (800, 600))
            img2_small = cv2.resize(frame2, (800, 600))
            img1_b64 = _encode_jpeg(img1_small, quality=75)
            img2_b64 = _encode_jpeg(img2_small, quality=75)

            # Close-up crops from raw frames (no overlays)
            crop1 = _crop_around_gaze(first_frame_bgr, first_gaze[0], first_gaze[1])
            crop2 = _crop_around_gaze(second_frame_bgr, second_gaze[0], second_gaze[1])
            crop1_b64 = _encode_jpeg(crop1, quality=90)
            crop2_b64 = _encode_jpeg(crop2, quality=90)

            user_content = [
                {
                    "type": "text",
                    "text": (
                        f"The user looked at two objects sequentially. "
                        f"Image 1 shows the full scene where they looked first (magenta marker '1') "
                        f"for {dur1_ms:.0f} ms. "
                        f"Image 2 shows the full scene where they looked second (cyan marker '2') "
                        f"for {dur2_ms:.0f} ms. "
                        f"Image 3 is a close-up crop around gaze point 1 (no overlays). "
                        f"Image 4 is a close-up crop around gaze point 2 (no overlays). "
                        f"Use the close-ups to precisely identify each object — "
                        f"the segmentation overlays in the full scenes may cover small objects. "
                        f"Identify both objects and infer intent.\n\n"
                        f"{JSON_SCHEMA_DESCRIPTION}"
                    ),
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{img1_b64}", "detail": "low"},
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{img2_b64}", "detail": "low"},
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{crop1_b64}", "detail": "high"},
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{crop2_b64}", "detail": "high"},
                },
            ]

            return self._call_api(
                [{"role": "user", "content": user_content}],
                system_prompt=PAIR_SYSTEM_PROMPT,
                gaze_hit_info=gaze_hit_info,
            )
        except Exception:
            traceback.print_exc()
            return dict(FALLBACK_RESPONSE)

    # ── shared API call ───────────────────────────────────────────────────────

    def _call_api(self, messages: list[dict], system_prompt: str = SYSTEM_PROMPT, gaze_hit_info: str = "") -> dict:
        import sys
        from pathlib import Path

        # ── Log prompt ────────────────────────────────────────────────
        import time as _time
        if self._session_dir is not None:
            out_dir = self._session_dir
        else:
            out_dir = Path("output/vlm_prompts")
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = int(_time.time() * 1000)  # ms precision to avoid collisions

        log_lines = []
        log_lines.append(f"# VLM Prompt — {_time.strftime('%Y-%m-%d %H:%M:%S')}")
        log_lines.append(f"Model: {self.model}")
        log_lines.append(f"\n## System Prompt\n```\n{system_prompt}\n```")

        if gaze_hit_info:
            log_lines.append(f"\n## Bayesian Gaze Inference\n```\n{gaze_hit_info}\n```")

        print(f"\n{'='*60}", file=sys.stderr)
        print(f"[VLM] Model: {self.model}", file=sys.stderr)
        print(f"[VLM] System prompt:", file=sys.stderr)
        print(f"  {system_prompt[:200]}...", file=sys.stderr)

        img_idx = 0
        for msg in messages:
            content = msg.get("content")
            if isinstance(content, str):
                print(f"[VLM] User (text): {content[:300]}", file=sys.stderr)
                log_lines.append(f"\n## User Message\n```\n{content}\n```")
            elif isinstance(content, list):
                log_lines.append("\n## User Message (multipart)")
                for item in content:
                    if item.get("type") == "text":
                        print(f"[VLM] User (text): {item['text'][:300]}", file=sys.stderr)
                        log_lines.append(f"\n### Text\n```\n{item['text']}\n```")
                    elif item.get("type") == "image_url":
                        url = item["image_url"]["url"]
                        detail = item["image_url"].get("detail", "auto")
                        size_kb = len(url) * 3 // 4 // 1024
                        print(f"[VLM] User (image): ~{size_kb}KB, detail={detail}", file=sys.stderr)

                        # Save image
                        try:
                            b64_data = url.split(",", 1)[1]
                            img_bytes = base64.b64decode(b64_data)
                            img_name = f"prompt_{ts}_img{img_idx}.jpg"
                            img_path = out_dir / img_name
                            img_path.write_bytes(img_bytes)
                            print(f"[VLM] Saved prompt image: {img_path}", file=sys.stderr)
                            log_lines.append(f"\n### Image (detail={detail}, ~{size_kb}KB)")
                            log_lines.append(f"![prompt image]({img_name})")
                            img_idx += 1
                        except Exception as exc:
                            # Best-effort DEBUG prompt-image dump; never let a disk/
                            # decode failure here break the VLM call below.
                            print(f"[VLM] WARN: prompt-image save failed: {exc!r}",
                                  file=sys.stderr)

        print(f"{'='*60}", file=sys.stderr, flush=True)

        # ── API call ──────────────────────────────────────────────────
        api_start_time = _time.perf_counter()
        if self._backend == "gemini":
            raw = self._call_gemini(messages, system_prompt)
        else:
            raw = self._call_openai(messages, system_prompt)
        api_latency = _time.perf_counter() - api_start_time

        # Log response
        print(f"[VLM] Response: {raw[:500]}", file=sys.stderr, flush=True)
        print(f"[VLM] API latency ({self._backend}): {api_latency:.3f}s", file=sys.stderr, flush=True)

        # Save full prompt + response log
        try:
            log_lines.append(f"\n## VLM Response (Latency: {api_latency:.3f}s)\n```json\n{raw}\n```")
            log_path = out_dir / f"prompt_{ts}.md"
            log_path.write_text("\n".join(log_lines))
            print(f"[VLM] Saved prompt log: {log_path}", file=sys.stderr, flush=True)
        except Exception as exc:
            # Best-effort prompt/response log; a write failure must not fail the call.
            print(f"[VLM] WARN: prompt-log write failed: {exc!r}", file=sys.stderr)

        # Append to session log (combined markdown)
        # All images and session.md are in the same directory, so bare filenames work
        if self._session_vlm_log is not None:
            try:
                self._vlm_call_count += 1
                separator = f"\n\n{'='*60}\n"
                header = f"# VLM Call #{self._vlm_call_count}\n"
                with open(self._session_vlm_log, "a") as f:
                    f.write(separator + header + "\n".join(log_lines) + "\n")
            except Exception as exc:
                # Best-effort combined session log; a write failure must not fail the call.
                print(f"[VLM] WARN: session-log append failed: {exc!r}", file=sys.stderr)

        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            # Attempt to repair truncated JSON (e.g. Gemini ran out of tokens)
            import re
            try:
                # Strip trailing incomplete key/value and close brackets
                repaired = re.sub(r',\s*"[^"]*$', '', raw)  # remove trailing incomplete key
                repaired = re.sub(r',\s*$', '', repaired)    # remove trailing comma
                # Count and close unclosed brackets/braces
                open_braces = repaired.count('{') - repaired.count('}')
                open_brackets = repaired.count('[') - repaired.count(']')
                repaired += ']' * open_brackets + '}' * open_braces
                result = json.loads(repaired)
                print(f"[VLM] Repaired truncated JSON", file=sys.stderr)
                return result
            except Exception:
                print(f"[VLM] Failed to parse response, using fallback", file=sys.stderr)
                return dict(FALLBACK_RESPONSE)

    # ── backend implementations ───────────────────────────────────────────

    def _call_openai(self, messages: list[dict], system_prompt: str) -> str:
        response = self._openai.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                *messages,
            ],
            max_tokens=self.max_tokens,
            response_format={"type": "json_object"},
        )
        return response.choices[0].message.content or "{}"

    def _call_gemini(self, messages: list[dict], system_prompt: str) -> str:
        from google.genai import types as gtypes

        # Build contents list for Gemini
        parts = []
        for msg in messages:
            content = msg.get("content")
            if isinstance(content, str):
                parts.append(content)
            elif isinstance(content, list):
                for item in content:
                    if item.get("type") == "text":
                        parts.append(item["text"])
                    elif item.get("type") == "image_url":
                        url = item["image_url"]["url"]
                        b64_data = url.split(",", 1)[1]
                        img_bytes = base64.b64decode(b64_data)
                        parts.append(gtypes.Part.from_bytes(
                            data=img_bytes,
                            mime_type="image/jpeg",
                        ))

        import time as _time
        for attempt in range(3):
            try:
                config_args = {
                    "system_instruction": system_prompt,
                    "max_output_tokens": self.max_tokens,
                    "response_mime_type": "application/json",
                }
                if self.thinking_budget is not None:
                    config_args["thinking_config"] = gtypes.ThinkingConfig(
                        thinking_budget=self.thinking_budget
                    )
                response = self._gemini.models.generate_content(
                    model=self.model,
                    contents=parts,
                    config=gtypes.GenerateContentConfig(**config_args),
                )
                return response.text or "{}"
            except Exception as e:
                err_str = str(e)
                if ("503" in err_str or "UNAVAILABLE" in err_str or "429" in err_str) and attempt < 2:
                    wait = (attempt + 1) * 2  # 2s, 4s
                    print(f"[VLM] Gemini {err_str[:80]}… retrying in {wait}s ({attempt+1}/3)", file=sys.stderr)
                    _time.sleep(wait)
                else:
                    raise
