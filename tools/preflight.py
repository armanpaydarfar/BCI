#!/usr/bin/env python3
"""
preflight.py — deployment-readiness doctor for the Harmony BCI hosts.

WHY THIS EXISTS
---------------
The pytest suite is fast, hardware-free, and contract-level — by design. That
leaves a structural blind spot: a host can have a fully green suite and still be
unable to run the system, because the heavy perception deps are *lazy-imported*
inside constructors (`perception/depth_estimator.py:47` does `import depth_pro`;
`perception/intent_reasoner.py:231` does `from google import genai`) and no test
ever constructs those objects. Both the Windows GPU host and the Linux control
host hit exactly this (missing `depth_pro`, drifted env pins, mis-placed weights,
un-run firewall script) the day before the 2026-06-22 HIL.

This doctor closes that gap. It is NOT a pytest test and is deliberately kept out
of the fast pre-commit suite (see the proposal at
SoftwareDocs/projects/harmony-bci/test-suite/preflight-and-env-drift-proposal.md).
Run it at bring-up / deploy:

    python tools/preflight.py --role server      # GPU/perception host
    python tools/preflight.py --role control     # Linux operator host
    python tools/preflight.py --role server --with-hardware   # day-of strict

Cross-platform on purpose (runs on native Windows, unlike tools/bootstrap_machine.sh
which `:16` notes a Windows server "can't run").

SEMANTICS
---------
Each check yields a CheckResult with status PASS / FAIL / SKIP.
  - PASS  — verified good.
  - FAIL  — broken; would block a real session.
  - SKIP  — not applicable / intentionally not run (e.g. a device-tier check on a
            desk run without `--with-hardware`).
Exit code: 0 if no FAIL (PASS/SKIP only), 1 if any FAIL, 2 on usage/internal error.

In strict mode (`--with-hardware`) the hardware/device tier must PASS — a SKIP
there is promoted to FAIL, so a day-of run can't quietly pass with hardware absent.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata as _md
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable, Iterable, Optional

# Windows OMP guard. This doctor imports the whole perception stack (torch +
# ultralytics + timm + depth_pro) in one process to fire the lazy imports — that
# flat import order trips Intel's duplicate-libiomp5md check (OMP #15) on the
# conda+pip torch combo. vlm_service's own import path does NOT hit this (verified
# 2026-06-21), so production is unaffected; this guard is diagnostic-tool-only.
# Must be set before torch is first imported (torch is imported lazily in checks).
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PASS, FAIL, SKIP = "PASS", "FAIL", "SKIP"


# ─────────────────────────────────────────────────────────────────────────────
# Result type + runner  (STABLE INTERFACE — the control branch builds on this)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CheckResult:
    """One check outcome. `name` is a short stable id (dotted); `detail` is a
    human one-liner; `remedy` is an optional fix hint shown on FAIL."""
    name: str
    status: str               # PASS | FAIL | SKIP
    detail: str = ""
    remedy: str = ""


@dataclass
class Context:
    """Shared state passed to every check function. Role branches read from here
    rather than re-importing/re-resolving."""
    role: str
    strict: bool
    models_dir: str = ""
    config: object = None     # the imported `config` module (or None if it failed)
    results: list = field(default_factory=list)


# A check is `Callable[[Context], CheckResult | Iterable[CheckResult]]`.
Check = Callable[[Context], "CheckResult | Iterable[CheckResult]"]


def run_checks(ctx: Context, checks: Iterable[Check]) -> list[CheckResult]:
    """Run each check, capturing exceptions as FAILs (a check that raises is a
    bug in the check, not a pass). This runner does NOT do strict-mode SKIP→FAIL
    promotion — that happens at construction time in `device_skip()`; here we just
    collect whatever each check returns."""
    out: list[CheckResult] = []
    for chk in checks:
        try:
            r = chk(ctx)
        except Exception as e:  # a check itself blew up — surface it, don't hide
            out.append(CheckResult(getattr(chk, "__name__", "check"), FAIL,
                                   f"check raised {type(e).__name__}: {e}"))
            continue
        rs = [r] if isinstance(r, CheckResult) else list(r)
        out.extend(rs)
    ctx.results = out
    return out


def device_skip(name: str, detail: str, ctx: Context) -> CheckResult:
    """Helper for hardware/device-tier checks: SKIP on a desk run, but in strict
    (`--with-hardware`) mode a missing device is a FAIL so day-of can't pass with
    hardware absent."""
    if ctx.strict:
        return CheckResult(name, FAIL, detail + " (strict: hardware required)",
                           remedy="connect/enable the device or drop --with-hardware")
    return CheckResult(name, SKIP, detail + " (pass --with-hardware to require)")


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers  (control branch reuses these — keep signatures stable)
# ─────────────────────────────────────────────────────────────────────────────

def load_config() -> object:
    """Import the repo `config` (which pulls `config_local` machine values).
    Returns the module, or raises — caller wraps."""
    return importlib.import_module("config")


def validate_api_key(key: Optional[str]) -> tuple[bool, str]:
    """Google API key sanity (format only, never echo the key). Google keys are
    `AIzaSy`-prefixed, ~39 chars. Returns (ok, detail)."""
    if not key or not str(key).strip():
        return False, "empty / unset"
    k = str(key).strip()
    ok = k.startswith("AIzaSy") and 35 <= len(k) <= 45
    return ok, f"prefix={'AIzaSy' if k.startswith('AIzaSy') else k[:6]+'…'}, len={len(k)}"


def resolve_weight(models_dir: str, name_or_path: str) -> str:
    """Mirror vlm_service.py:1943-1949: a bare filename joins onto
    PERCEPTION_MODELS_DIR; an absolute path is taken as-is. Returns the resolved
    path (not checked for existence — caller stats it)."""
    if not name_or_path:
        return ""
    if os.path.isabs(name_or_path):
        return name_or_path
    return os.path.join(models_dir or "", name_or_path)


# environment.yml pin parsing ------------------------------------------------

_IMPORT_NAME = {
    # distribution name -> import name, where they diverge
    "scikit-learn": "sklearn",
    "opencv-python": "cv2",
    "google-genai": "google.genai",
    "pupil-labs-realtime-api": "pupil_labs.realtime_api",
    "depth-pro": "depth_pro",
    "pyside6": "PySide6",
    "pyyaml": "yaml",
    "pillow": "PIL",
    "pyserial": "serial",
}


@dataclass
class Pin:
    name: str
    op: Optional[str]        # "==" (exact, incl. conda "="), "<"/"<=" etc (range), or None
    version: Optional[str]


def parse_env_pins(environment_yml: Path) -> list[Pin]:
    """Light parser for environment.yml dependency lines. conda single-`=` is an
    exact pin; pip uses `==`. Anything with no version is import-only. We avoid a
    PyYAML dependency (it may not be installed) — the dep lines are a simple,
    known `- name<spec>` shape."""
    pins: list[Pin] = []
    line_re = re.compile(r"^\s*-\s+([A-Za-z0-9_.\-]+)\s*(==|<=|>=|<|>|=)?\s*([0-9][^\s#;]*)?")
    for raw in environment_yml.read_text(encoding="utf-8").splitlines():
        stripped = raw.strip()
        if not stripped.startswith("- "):
            continue
        # Skip the `- pip:` section header and obvious non-deps.
        if stripped.rstrip().endswith(":"):
            continue
        m = line_re.match(raw)
        if not m:
            continue
        name, op, ver = m.group(1), m.group(2), m.group(3)
        if name in ("pip",):
            continue
        # Normalize conda exact "=" to "==".
        if op == "=":
            op = "=="
        pins.append(Pin(name=name, op=op, version=ver))
    return pins


def _release_tuple(v: Optional[str]) -> tuple[int, ...]:
    if not v:
        return ()
    out = []
    for part in re.split(r"[.\-+]", v):
        if part.isdigit():
            out.append(int(part))
        else:
            break
    return tuple(out)


def _installed_version(dist: str) -> Optional[str]:
    """Best-effort installed version. Prefer dist metadata; fall back to importing
    the module and reading __version__ (some conda builds ship no Version metadata
    — e.g. numpy here — which would otherwise hide a real pin drift)."""
    try:
        v = _md.version(dist)
        if v:
            return v
    except _md.PackageNotFoundError:
        pass
    mod = _IMPORT_NAME.get(dist, dist.replace("-", "_"))
    try:
        return getattr(importlib.import_module(mod), "__version__", None)
    except Exception:
        return None


def check_env_drift(ctx: Context) -> Iterable[CheckResult]:
    """Pin-operator-aware env-vs-spec check (P2). EXACT for any `==`/conda-`=`
    pin (numpy 1.26.4, pylsl 1.16.2, scipy/sklearn/pyriemann; python at
    major.minor), RANGE for `<`/`<=` (pandas<3), IMPORT-ONLY for unpinned. The
    operator from environment.yml drives it, so the intentionally-floating set is
    never version-asserted and Linux/Windows solves don't false-fail."""
    env_yml = ROOT / "environment.yml"
    if not env_yml.is_file():
        yield CheckResult("env.environment_yml", FAIL, "environment.yml not found")
        return
    pins = parse_env_pins(env_yml)
    for pin in pins:
        dist = pin.name
        # python is the interpreter, not a distribution — handle separately.
        if dist == "python":
            if pin.op == "==" and pin.version:
                # major.minor only — conda solves the patch, and a patch bump
                # (3.12.7 → 3.12.13) is benign. A 3.11/3.13 mismatch still FAILs.
                want = _release_tuple(pin.version)[:2]
                got = tuple(sys.version_info[:2])
                ok = got == want
                cur = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
                yield CheckResult("env.pin.python", PASS if ok else FAIL,
                                  f"want {'.'.join(map(str, want))}.x, have {cur}")
            continue
        have = _installed_version(dist)
        if not have:
            # Exactly-pinned but absent / version unreadable → FAIL (a real
            # readiness problem). Unpinned absent is left to the import-only check.
            if pin.op == "==":
                yield CheckResult(f"env.pin.{dist}", FAIL,
                                  "not installed / version unreadable",
                                  remedy="rebuild the env from environment.yml")
            continue
        if pin.op == "==" and pin.version:
            want, got = _release_tuple(pin.version), _release_tuple(have)
            ok = got[:len(want)] == want
            yield CheckResult(f"env.pin.{dist}", PASS if ok else FAIL,
                              f"want =={pin.version}, have {have}",
                              remedy="" if ok else "rebuild env from environment.yml (drifted off pin)")
        elif pin.op in ("<", "<=") and pin.version:
            want, got = _release_tuple(pin.version), _release_tuple(have)
            ok = got < want if pin.op == "<" else got <= want
            yield CheckResult(f"env.pin.{dist}", PASS if ok else FAIL,
                              f"want {pin.op}{pin.version}, have {have}")
        # else: floating dep — version not asserted (by design).


def check_declared_imports(ctx: Context, names: list[str]) -> Iterable[CheckResult]:
    """Import-only check for the declared perception deps that the lazy
    constructors need (depth_pro, google.genai, timm, ultralytics, …). This is
    what makes the lazy-import gap visible without making the imports eager."""
    for dist in names:
        modname = _IMPORT_NAME.get(dist, dist.replace("-", "_"))
        try:
            importlib.import_module(modname)
            yield CheckResult(f"dep.{dist}", PASS, f"import {modname} ok")
        except Exception as e:
            yield CheckResult(f"dep.{dist}", FAIL, f"import {modname} -> {type(e).__name__}: {e}",
                              remedy=f"pip/conda install {dist} into the active env")


# ─────────────────────────────────────────────────────────────────────────────
# SERVER role checks  (owned here)
# ─────────────────────────────────────────────────────────────────────────────

def _server_config(ctx: Context) -> Iterable[CheckResult]:
    cfg = ctx.config
    if cfg is None:
        yield CheckResult("config.import", FAIL, "could not import config/config_local")
        return
    key_ok, key_detail = validate_api_key(getattr(cfg, "GOOGLE_API_KEY", ""))
    yield CheckResult("config.GOOGLE_API_KEY", PASS if key_ok else FAIL, key_detail,
                      remedy="" if key_ok else "set GOOGLE_API_KEY in config_local.py")
    md = getattr(cfg, "PERCEPTION_MODELS_DIR", "") or ""
    ctx.models_dir = md
    md_ok = bool(md) and os.path.isdir(md)
    yield CheckResult("config.PERCEPTION_MODELS_DIR", PASS if md_ok else FAIL,
                      f"{md or '(unset)'}",
                      remedy="" if md_ok else "set PERCEPTION_MODELS_DIR to an existing dir in config_local.py")


def _server_weights(ctx: Context) -> Iterable[CheckResult]:
    md = ctx.models_dir
    # seg + depth resolve under the models dir (bare-name → join).
    for label, fname in (("FastSAM-s.pt", "FastSAM-s.pt"), ("depth_pro.pt", "depth_pro.pt")):
        p = resolve_weight(md, fname)
        ok = bool(p) and os.path.isfile(p)
        yield CheckResult(f"weights.{label}", PASS if ok else FAIL,
                          p or "(no models dir)",
                          remedy="" if ok else f"place {fname} under PERCEPTION_MODELS_DIR")
    # recognizer weight lives at the repo root and is passed by absolute path.
    yolo = ROOT / "yolo26n.pt"
    yolo_ok = yolo.is_file()
    yield CheckResult("weights.yolo26n.pt", PASS if yolo_ok else FAIL,
                      str(yolo),
                      remedy="" if yolo_ok else "yolo26n.pt missing from repo root (should be git-tracked)")


def _server_cuda(ctx: Context) -> Iterable[CheckResult]:
    try:
        import torch
    except Exception as e:
        yield CheckResult("torch.import", FAIL, f"{type(e).__name__}: {e}")
        return
    yield CheckResult("torch.import", PASS, torch.__version__)
    if torch.cuda.is_available():
        yield CheckResult("torch.cuda", PASS, torch.cuda.get_device_name(0))
    else:
        # CUDA absence is fine on a CPU desk box; on a GPU server day-of it isn't.
        yield device_skip("torch.cuda", "cuda.is_available() is False", ctx)


def _server_model_construct(ctx: Context) -> Iterable[CheckResult]:
    """The gap-catcher: actually construct the perception objects so the lazy
    depth_pro / genai imports fire. Skips gracefully if weights are missing
    (already FAILed above) to avoid a confusing cascade."""
    md = ctx.models_dir
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        device = "cpu"

    depth_ckpt = resolve_weight(md, "depth_pro.pt")
    if depth_ckpt and os.path.isfile(depth_ckpt):
        try:
            import numpy as np
            from perception.depth_estimator import DepthEstimator
            est = DepthEstimator(checkpoint=depth_ckpt, device=device)
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            depth, _ = est.estimate(frame)
            ok = depth.shape == (480, 640)
            yield CheckResult("construct.DepthEstimator", PASS if ok else FAIL,
                              f"loaded on {device}, estimate -> {depth.shape}")
        except Exception as e:
            yield CheckResult("construct.DepthEstimator", FAIL,
                              f"{type(e).__name__}: {e}",
                              remedy="install depth-pro (--no-deps) into the env")
    else:
        yield CheckResult("construct.DepthEstimator", SKIP, "depth_pro.pt missing (see weights)")

    seg = resolve_weight(md, "FastSAM-s.pt")
    if seg and os.path.isfile(seg):
        try:
            from perception.object_detector import ObjectDetector
            ObjectDetector(model_size=seg, device=device)
            yield CheckResult("construct.ObjectDetector", PASS, f"FastSAM loaded on {device}")
        except Exception as e:
            yield CheckResult("construct.ObjectDetector", FAIL, f"{type(e).__name__}: {e}")
    else:
        yield CheckResult("construct.ObjectDetector", SKIP, "FastSAM-s.pt missing (see weights)")

    # IntentReasoner(gemini) — fires the lazy `from google import genai`. The
    # client constructor does no network I/O, so this is hardware-free.
    cfg = ctx.config
    key = getattr(cfg, "GOOGLE_API_KEY", "") if cfg else ""
    try:
        from perception.intent_reasoner import IntentReasoner
        IntentReasoner(api_key=key or "preflight-probe", model="gemini-2.5-flash")
        yield CheckResult("construct.IntentReasoner", PASS, "gemini backend constructed (no network)")
    except Exception as e:
        yield CheckResult("construct.IntentReasoner", FAIL, f"{type(e).__name__}: {e}",
                          remedy="install google-genai into the env")


def _server_network(ctx: Context) -> Iterable[CheckResult]:
    """Device/cross-host tier — the relay/Neon side needs the control host up.
    Hardware-free desk run SKIPs; strict day-of requires it. (Stub for the
    bind/relay reachability probe; the control host owns the round-trip side.)"""
    yield device_skip("net.frame_relay", "relay reachability needs the control host", ctx)


def run_server_checks(ctx: Context) -> list[CheckResult]:
    deps = ["depth-pro", "timm", "ultralytics", "google-genai", "openai"]
    checks: list[Check] = [
        _server_config,
        _server_weights,
        check_env_drift,
        lambda c: check_declared_imports(c, deps),
        _server_cuda,
        _server_model_construct,
        _server_network,
    ]
    return run_checks(ctx, checks)


# ─────────────────────────────────────────────────────────────────────────────
# CONTROL role checks  (owned by the control-host agent — linux-primary)
# Implements the Tier-1/Tier-2 spec from the proposal's "Control-host response".
# Control is Linux-only (CLAUDE.md realtime policy), so the device helpers use
# Linux `ping`/`ip` — reached only on the control path.
# ─────────────────────────────────────────────────────────────────────────────

def _ping(host: str, timeout_s: int = 2) -> bool:
    if not host:
        return False
    try:
        return subprocess.run(["ping", "-c", "1", "-W", str(timeout_s), host],
                              capture_output=True).returncode == 0
    except Exception:
        return False


def _host_has_addr(addr: str) -> bool:
    try:
        out = subprocess.run(["ip", "-4", "addr"], capture_output=True, text=True)
        return addr in out.stdout
    except Exception:
        return False


def _control_config(ctx: Context) -> Iterable[CheckResult]:
    """Tier 1 — config_local *values* the control host must have set."""
    cfg = ctx.config
    if cfg is None:
        yield CheckResult("config.import", FAIL, "could not import config/config_local")
        return
    dd = getattr(cfg, "DATA_DIR", "") or ""
    dd_ok = bool(dd) and os.path.isdir(dd) and os.access(dd, os.W_OK)
    yield CheckResult("config.DATA_DIR", PASS if dd_ok else FAIL, dd or "(unset)",
                      remedy="" if dd_ok else "set DATA_DIR to an existing writable dir in config_local.py")
    pl = getattr(cfg, "POSE_LIBRARY_PATH", "") or ""
    pl_ok = bool(pl) and os.path.isfile(pl)
    yield CheckResult("config.POSE_LIBRARY_PATH", PASS if pl_ok else FAIL, pl or "(unset)",
                      remedy="" if pl_ok else "resolves to a missing file — config.py:510 re-assigns it AFTER `from config_local import *` (config_local override is clobbered; hardcoded filename is stale)")
    for key in ("NEON_COMPANION_HOST", "VLM_SERVICE_HOST", "ARDUINO_PORT"):
        v = getattr(cfg, key, "") or ""
        ok = bool(str(v).strip())
        yield CheckResult(f"config.{key}", PASS if ok else FAIL, f"{v or '(unset)'}",
                          remedy="" if ok else f"set {key} in config_local.py")


def _control_assets(ctx: Context) -> Iterable[CheckResult]:
    """Tier 1 — assets / model-construct: the control analogue of the depth gap-catcher."""
    # Gaze recognizer weight must be tracked at the repo root. gaze_system.py does
    # YOLO("yolo26n.pt") (bare name → cwd); if the file is absent ultralytics
    # silently auto-downloads a *different* weight — so assert FOUND, then load by
    # absolute path (no download) to confirm it's a valid detector.
    yolo = ROOT / "yolo26n.pt"
    if not yolo.is_file():
        yield CheckResult("asset.yolo26n", FAIL, str(yolo),
                          remedy="yolo26n.pt missing from repo root (must be git-tracked; see !/yolo26n.pt)")
    else:
        try:
            from ultralytics import YOLO
            m = YOLO(str(yolo))
            yield CheckResult("asset.yolo26n", PASS,
                              f"repo-root weight loads (task={m.task}, {len(m.names)} classes)")
        except Exception as e:
            yield CheckResult("asset.yolo26n", FAIL, f"{type(e).__name__}: {e}")
    # Active gaze pose library loads.
    cfg = ctx.config
    pl = (getattr(cfg, "POSE_LIBRARY_PATH", "") if cfg else "") or ""
    if pl and os.path.isfile(pl):
        try:
            import numpy as np
            d = np.load(pl, allow_pickle=True)
            keys = list(getattr(d, "files", []) or [])
            yield CheckResult("asset.pose_library", PASS if keys else FAIL,
                              f"{os.path.basename(pl)}: {len(keys)} arrays")
        except Exception as e:
            yield CheckResult("asset.pose_library", FAIL, f"{type(e).__name__}: {e}")
    else:
        yield CheckResult("asset.pose_library", SKIP,
                          "POSE_LIBRARY_PATH missing (see config.POSE_LIBRARY_PATH)")
    # WS5 apriltag stack — pupil-apriltags is pip-installed (NOT in environment.yml),
    # so env-drift can't see it; check it + the Tier-1 apriltag modules explicitly.
    for dist, mod in (("pupil-apriltags", "pupil_apriltags"),
                      ("apriltag_calib", "Utils.gaze.apriltag_calib"),
                      ("apriltag_detect", "Utils.gaze.apriltag_detect"),
                      ("apriltag_world", "Utils.gaze.apriltag_world"),
                      ("harmony_link", "Utils.gaze.harmony_link")):
        try:
            importlib.import_module(mod)
            yield CheckResult(f"asset.{dist}", PASS, f"import {mod} ok")
        except Exception as e:
            yield CheckResult(f"asset.{dist}", FAIL, f"import {mod} -> {type(e).__name__}: {e}",
                              remedy=("conda run -n lsl pip install pupil-apriltags (machine-sync task 004)"
                                      if dist == "pupil-apriltags" else ""))


def _control_devices(ctx: Context) -> Iterable[CheckResult]:
    """Tier 2 — device reachability. SKIP on a desk run; FAIL under --with-hardware."""
    cfg = ctx.config
    # EEG LSL stream (eegoSports) — every realtime driver needs it at startup.
    try:
        from pylsl import resolve_streams
        streams = resolve_streams(2.0)
        eeg = [s for s in streams if (s.type() or "").upper() == "EEG"]
        if eeg:
            yield CheckResult("device.eeg_lsl", PASS, f"{len(eeg)} EEG stream(s) (e.g. {eeg[0].name()})")
        else:
            yield device_skip("device.eeg_lsl", f"no EEG LSL stream ({len(streams)} total seen)", ctx)
    except Exception as e:
        yield CheckResult("device.eeg_lsl", FAIL, f"pylsl error: {type(e).__name__}: {e}")
    # Neon Companion reachable.
    neon = (getattr(cfg, "NEON_COMPANION_HOST", "") if cfg else "") or ""
    if _ping(neon):
        yield CheckResult("device.neon", PASS, f"{neon} reachable")
    else:
        yield device_skip("device.neon", f"Neon host {neon or '(unset)'} unreachable", ctx)
    # Robot LAN (Part B): this host holds 192.168.2.2 + robot 192.168.2.1 pings.
    holds, robot = _host_has_addr("192.168.2.2"), _ping("192.168.2.1")
    if holds and robot:
        yield CheckResult("device.robot_lan", PASS, "192.168.2.2 held; robot 192.168.2.1 pings")
    else:
        yield device_skip("device.robot_lan", f"192.168.2.2 held={holds}, robot ping={robot}", ctx)
    # FES/Arduino serial port present.
    ard = (getattr(cfg, "ARDUINO_PORT", "") if cfg else "") or ""
    if ard and os.path.exists(ard):
        yield CheckResult("device.arduino", PASS, ard)
    else:
        yield device_skip("device.arduino", f"ARDUINO_PORT {ard or '(unset)'} not present", ctx)
    # Cross-host: vlm_service status round-trip (the perception server).
    try:
        from Utils.perception_clients import VLMClient
        r = VLMClient(cfg).status(timeout_s=3.0)
        if isinstance(r, dict) and r.get("ok"):
            yield CheckResult("device.vlm_status", PASS,
                              f"depth_enabled={r.get('depth_enabled')}, model={r.get('model')}")
        else:
            yield CheckResult("device.vlm_status", FAIL, f"unexpected reply: {str(r)[:80]}")
    except Exception as e:
        yield device_skip("device.vlm_status", f"vlm_service not answering ({type(e).__name__})", ctx)


def run_control_checks(ctx: Context) -> list[CheckResult]:
    deps = ["ultralytics", "pyserial", "pygame"]   # control realtime/gaze deps not in EXACT_CORE
    checks: list[Check] = [
        _control_config,
        check_env_drift,                                   # shared (both roles)
        lambda c: check_declared_imports(c, deps),
        _control_assets,
        _control_devices,
    ]
    return run_checks(ctx, checks)


# ─────────────────────────────────────────────────────────────────────────────
# Output + CLI
# ─────────────────────────────────────────────────────────────────────────────

def _print_text(role: str, strict: bool, results: list[CheckResult]) -> None:
    icon = {PASS: "[PASS]", FAIL: "[FAIL]", SKIP: "[SKIP]"}
    for r in results:
        print(f"{icon[r.status]} {r.name:34s} {r.detail}")
        if r.status == FAIL and r.remedy:
            print(f"        └ remedy: {r.remedy}")
    n = {s: sum(1 for r in results if r.status == s) for s in (PASS, FAIL, SKIP)}
    verdict = FAIL if n[FAIL] else PASS
    mode = " (strict)" if strict else ""
    print(f"\npreflight({role}){mode}: {n[PASS]} PASS, {n[FAIL]} FAIL, {n[SKIP]} SKIP  → {verdict}")


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Harmony BCI deployment-readiness doctor.")
    p.add_argument("--role", required=True, choices=["server", "control"])
    p.add_argument("--with-hardware", action="store_true",
                   help="strict day-of mode: device-tier checks must PASS (a SKIP becomes FAIL).")
    p.add_argument("--json", action="store_true", help="emit machine-readable JSON.")
    p.add_argument("--models-dir", default=None,
                   help="override PERCEPTION_MODELS_DIR (default: from config).")
    args = p.parse_args(argv)

    try:
        cfg = load_config()
    except Exception as e:
        cfg = None
        if not args.json:
            print(f"[warn] config import failed: {type(e).__name__}: {e}", file=sys.stderr)

    ctx = Context(role=args.role, strict=args.with_hardware, config=cfg,
                  models_dir=(args.models_dir or
                              (getattr(cfg, "PERCEPTION_MODELS_DIR", "") if cfg else "")))

    results = run_server_checks(ctx) if args.role == "server" else run_control_checks(ctx)

    if args.json:
        n = {s: sum(1 for r in results if r.status == s) for s in (PASS, FAIL, SKIP)}
        print(json.dumps({
            "role": args.role, "strict": args.with_hardware,
            "results": [asdict(r) for r in results],
            "summary": {"pass": n[PASS], "fail": n[FAIL], "skip": n[SKIP]},
            "ok": n[FAIL] == 0,
        }, indent=2))
    else:
        _print_text(args.role, args.with_hardware, results)

    return 1 if any(r.status == FAIL for r in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
