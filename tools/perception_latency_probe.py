#!/usr/bin/env python3
"""
perception_latency_probe.py — measure round-trip latency of the perception
service stack (gaze + VLM + frame relay).

Per SoftwareDocs/GPU_Service_Host_Architecture_Plan.md §4.11. Targets on
LAN/Ethernet:

    - VLM   `status`         p99 < 5 ms
    - VLM   `segment`        p99 < 200 ms (CUDA fp16 FastSAM)
    - Gaze  `status` (5588)  p99 < 10 ms
    - Frame relay one-way    p99 < 30 ms at 10 Hz

Run from the operator side (Linux panel host in production, Windows in
single-machine dev). Defaults read from BCI/config.py.

Examples:

    python tools/perception_latency_probe.py
    python tools/perception_latency_probe.py --segment-iters 20

Output:
    All results print as a small table per probe; full JSON dump optional
    via ``--json``. Numbers are appended to <session_dir>/latency.jsonl
    when ``--out-dir`` is provided.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import struct
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Utils.perception_clients import udp_request  # noqa: E402  (path setup above)


def _percentiles(samples: List[float]) -> Dict[str, float]:
    if not samples:
        return {"n": 0, "p50": float("nan"), "p95": float("nan"),
                "p99": float("nan"), "min": float("nan"), "max": float("nan"),
                "mean": float("nan")}
    s = sorted(samples)
    n = len(s)

    def _pct(p: float) -> float:
        idx = max(0, min(n - 1, int(round(p * (n - 1)))))
        return s[idx]

    return {
        "n": n,
        "min": s[0],
        "max": s[-1],
        "mean": sum(s) / n,
        "p50": _pct(0.50),
        "p95": _pct(0.95),
        "p99": _pct(0.99),
    }


def _print_pct(label: str, target_ms: float, pct: Dict[str, float]) -> None:
    if pct["n"] == 0:
        print(f"  {label:30s} no samples")
        return
    p99_ms = pct["p99"] * 1000.0
    pass_str = "PASS" if p99_ms < target_ms else "FAIL"
    print(
        f"  {label:30s} n={pct['n']:3d} "
        f"p50={pct['p50']*1000:.1f}ms p95={pct['p95']*1000:.1f}ms "
        f"p99={p99_ms:.1f}ms  target<{target_ms}ms [{pass_str}]"
    )


# ── individual probes ──────────────────────────────────────────────────────


def probe_vlm_status(host: str, port: int, n: int) -> Dict[str, Any]:
    samples: List[float] = []
    errors = 0
    for _ in range(n):
        t0 = time.perf_counter()
        try:
            udp_request(host, port, {"cmd": "status"}, timeout_s=2.0)
            samples.append(time.perf_counter() - t0)
        except Exception:
            errors += 1
    return {"label": "vlm_status", "errors": errors, **_percentiles(samples)}


def probe_vlm_segment(host: str, port: int, n: int) -> Dict[str, Any]:
    samples: List[float] = []
    errors = 0
    for _ in range(n):
        t0 = time.perf_counter()
        try:
            udp_request(host, port, {"cmd": "segment"}, timeout_s=10.0)
            samples.append(time.perf_counter() - t0)
        except Exception:
            errors += 1
    return {"label": "vlm_segment", "errors": errors, **_percentiles(samples)}


def probe_gaze_status(host: str, port: int, n: int) -> Dict[str, Any]:
    samples: List[float] = []
    errors = 0
    for _ in range(n):
        t0 = time.perf_counter()
        try:
            udp_request(host, port, {"cmd": "status"}, timeout_s=2.0)
            samples.append(time.perf_counter() - t0)
        except Exception:
            errors += 1
    return {"label": "gaze_status", "errors": errors, **_percentiles(samples)}


def probe_frame_relay_oneway(host: str, port: int, n: int) -> Dict[str, Any]:
    """Connect, read N frame envelopes, compute (now - ts_send_ns) per
    frame. Tests the relay's one-way latency from the moment the relay
    serialised the envelope to the moment we finished receiving it.

    NOTE: ts_send_ns is monotonic_ns on the relay side. If the relay and
    probe run on different machines, monotonic clocks differ; subtracting
    them gives an offset-dominated number, not true one-way latency. For
    a meaningful one-way figure run the probe on the same machine as the
    relay; cross-host probes are best interpreted as relative jitter
    (their spread is meaningful even if the offset is not)."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(5.0)
    samples: List[float] = []
    try:
        sock.connect((host, int(port)))
    except OSError as e:
        return {"label": "relay_oneway", "errors": n, "error_msg": str(e), **_percentiles([])}

    try:
        # Skip handshake.
        prefix = _recv_exact(sock, 8)
        if prefix is None:
            return {"label": "relay_oneway", "errors": n, **_percentiles([])}
        json_len, jpeg_len = struct.unpack(">II", prefix)
        _ = _recv_exact(sock, json_len)
        if jpeg_len:
            _ = _recv_exact(sock, jpeg_len)

        for _ in range(n):
            prefix = _recv_exact(sock, 8)
            if prefix is None:
                break
            json_len, jpeg_len = struct.unpack(">II", prefix)
            json_buf = _recv_exact(sock, json_len)
            recv_t_ns = time.monotonic_ns()
            if jpeg_len:
                _ = _recv_exact(sock, jpeg_len)
            if json_buf is None:
                break
            try:
                hdr = json.loads(json_buf.decode("utf-8", errors="replace"))
            except json.JSONDecodeError:
                continue
            if hdr.get("type") != "frame":
                continue
            ts_send = hdr.get("ts_send_ns")
            if isinstance(ts_send, int):
                samples.append((recv_t_ns - ts_send) / 1e9)
    finally:
        try:
            sock.close()
        except OSError:
            pass

    return {"label": "relay_oneway", "errors": 0, **_percentiles(samples)}


def _recv_exact(sock: socket.socket, n: int) -> Optional[bytes]:
    chunks = []
    remaining = n
    while remaining > 0:
        try:
            chunk = sock.recv(remaining)
        except OSError:
            return None
        if not chunk:
            return None
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


# ── CLI ────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Perception service latency probe")
    p.add_argument("--vlm-host", default=None, help="default: config.VLM_SERVICE_HOST")
    p.add_argument("--vlm-port", type=int, default=None, help="default: config.VLM_SERVICE_PORT")
    p.add_argument("--gaze-host", default=None, help="default: config.GAZE_UDP_IP")
    p.add_argument("--gaze-port", type=int, default=None, help="default: config.GAZE_UDP_PORT")
    p.add_argument("--relay-host", default=None, help="default: config.FRAME_RELAY_DIAL_HOST")
    p.add_argument("--relay-port", type=int, default=None, help="default: config.FRAME_RELAY_PORT")
    p.add_argument("--status-iters", type=int, default=100)
    p.add_argument("--segment-iters", type=int, default=10,
                   help="lower default than the doc spec (50) so a quick run "
                        "doesn't hammer the GPU; bump for a serious sweep")
    p.add_argument("--relay-iters", type=int, default=20)
    p.add_argument("--skip-vlm", action="store_true")
    p.add_argument("--skip-gaze", action="store_true")
    p.add_argument("--skip-relay", action="store_true")
    p.add_argument("--out-dir", default=None,
                   help="If set, append a JSON line to <out-dir>/latency.jsonl")
    p.add_argument("--json", action="store_true",
                   help="Print full JSON results dict at the end")
    return p.parse_args()


def main() -> int:
    import config as cfg

    args = parse_args()
    vlm_host = args.vlm_host or getattr(cfg, "VLM_SERVICE_HOST", "127.0.0.1")
    vlm_port = args.vlm_port or int(getattr(cfg, "VLM_SERVICE_PORT", 5589))
    gaze_host = args.gaze_host or getattr(cfg, "GAZE_UDP_IP", "127.0.0.1")
    gaze_port = args.gaze_port or int(getattr(cfg, "GAZE_UDP_PORT", 5588))
    relay_host = args.relay_host or getattr(cfg, "FRAME_RELAY_DIAL_HOST", "127.0.0.1")
    relay_port = args.relay_port or int(getattr(cfg, "FRAME_RELAY_PORT", 5591))

    print(f"perception latency probe @ {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  vlm   = {vlm_host}:{vlm_port}")
    print(f"  gaze  = {gaze_host}:{gaze_port}")
    print(f"  relay = {relay_host}:{relay_port}")

    results: Dict[str, Any] = {"timestamp": time.time(), "endpoints": {
        "vlm": [vlm_host, vlm_port],
        "gaze": [gaze_host, gaze_port],
        "relay": [relay_host, relay_port],
    }}

    if not args.skip_vlm:
        print("--- VLM ---")
        r1 = probe_vlm_status(vlm_host, vlm_port, args.status_iters)
        _print_pct("status", target_ms=5.0, pct=r1)
        results["vlm_status"] = r1
        r2 = probe_vlm_segment(vlm_host, vlm_port, args.segment_iters)
        _print_pct("segment", target_ms=200.0, pct=r2)
        results["vlm_segment"] = r2

    if not args.skip_gaze:
        print("--- Gaze ---")
        r3 = probe_gaze_status(gaze_host, gaze_port, args.status_iters)
        _print_pct("status", target_ms=10.0, pct=r3)
        results["gaze_status"] = r3

    if not args.skip_relay:
        print("--- Frame relay ---")
        r4 = probe_frame_relay_oneway(relay_host, relay_port, args.relay_iters)
        _print_pct("oneway", target_ms=30.0, pct=r4)
        results["relay_oneway"] = r4

    if args.json:
        print(json.dumps(results, indent=2, sort_keys=True))

    if args.out_dir:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        with (out_dir / "latency.jsonl").open("a") as fh:
            fh.write(json.dumps(results) + "\n")
        print(f"appended -> {out_dir / 'latency.jsonl'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
