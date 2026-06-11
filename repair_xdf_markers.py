"""
repair_xdf_markers.py

One-off repair utility: reconstructs the missing experiment marker stream for
  sub-CLIN_SUBJ_007_ses-S002ONLINE_task-Default_run-001_eeg_old1.xdf

Context:
  The EEG XDF was recorded without UTIL_marker_stream.py running, so the
  experiment MarkerStream (name='MarkerStream', type='Markers', 4 channels
  float32) is absent from the file. The marker log produced by
  UTIL_marker_stream.py does exist and contains exact marker values and
  timestamps in the same LSL time base as the EEG stream, because that utility
  pulled its timestamps directly from the EEG LSL inlet.

Marker subset selection:
  The log covers the entire session, which spans two separate XDF recordings.
  To isolate markers that belong to this specific EEG file, we use the EEG
  stream's actual timestamp span [eeg_t_start, eeg_t_end] as the inclusion
  window. A 1-second trailing tolerance is added to eeg_t_end to capture any
  markers sent between the last EEG sample and the LabRecorder stop. All
  markers whose 'Sent marker' timestamp falls within this window are included;
  all others are discarded.

XDF write strategy:
  pyxdf is a read-only library. Rather than reimplementing a full XDF writer,
  this utility copies the original file verbatim and appends three binary XDF
  chunks for the reconstructed marker stream: StreamHeader (tag 2), Samples
  (tag 3), StreamFooter (tag 6). pyxdf processes chunks in file order; appended
  chunks are valid as long as the StreamHeader precedes its Samples, which this
  approach guarantees.

Output:
  Input:  ...eeg_old1.xdf   (original, untouched)
  Output: ...eeg_repaired.xdf  (copy of original with marker stream appended)

Marker stream format (matches UTIL_marker_stream.py exactly):
  name='MarkerStream', type='Markers', channel_count=4, nominal_srate=0,
  channel_format='float32', source_id='marker_stream_id'
  Channels per sample: [marker_value, lsl_timestamp, prob_mi, prob_rest]
"""

import re
import struct
import shutil
import sys
import numpy as np
import pyxdf
from pathlib import Path


# ── Paths (hard-coded for this one-off repair) ────────────────────────────────

XDF_DIR = Path("/home/arman-admin/Documents/CurrentStudy"
               "/sub-CLIN_SUBJ_007/ses-S002ONLINE/eeg")
XDF_IN  = XDF_DIR / ("sub-CLIN_SUBJ_007_ses-S002ONLINE"
                     "_task-Default_run-001_eeg_old1.xdf")
XDF_OUT = XDF_DIR / ("sub-CLIN_SUBJ_007_ses-S002ONLINE"
                     "_task-Default_run-001_eeg_repaired.xdf")

LOG_DIR  = Path("/home/arman-admin/Documents/CurrentStudy"
                "/sub-CLIN_SUBJ_007/marker_logs")
LOG_FILE = LOG_DIR / "marker_utility_2026-04-09_15-49-33.log"


# ── Marker stream parameters (must match UTIL_marker_stream.py exactly) ───────

MARKER_STREAM_NAME   = "MarkerStream"
MARKER_STREAM_TYPE   = "Markers"
MARKER_SOURCE_ID     = "marker_stream_id"
MARKER_N_CHANNELS    = 4
MARKER_SRATE         = 0        # irregular (event-based)
MARKER_FORMAT        = "float32"

# Stream ID to use for the appended marker stream; must not collide with
# existing IDs in the file.  The original file uses IDs 1 and 2.
NEW_STREAM_ID = 3

# Trailing tolerance added to eeg_t_end when filtering log markers.
# Captures any marker sent after the last EEG sample but before recording stopped.
T_END_TOLERANCE_SEC = 1.0


# ── XDF binary helpers ────────────────────────────────────────────────────────

def _write_varlen_int(value: int) -> bytes:
    """
    Encode a non-negative integer as an XDF variable-length integer.

    XDF varlen encoding (read by pyxdf._read_varlen_int):
      first byte = number of value bytes that follow (1, 4, or 8)
    """
    if value <= 0xFF:
        return struct.pack("BB", 1, value)
    if value <= 0xFFFFFFFF:
        return struct.pack("<BI", 4, value)
    return struct.pack("<BQ", 8, value)


def _build_chunk(tag: int, stream_id: int, content: bytes) -> bytes:
    """
    Assemble a binary XDF chunk.

    Layout:
      varlen_int(len(tag_bytes + id_bytes + content))
      uint16 tag
      uint32 stream_id
      content
    """
    tag_bytes = struct.pack("<H", tag)
    id_bytes  = struct.pack("<I", stream_id)
    payload   = tag_bytes + id_bytes + content
    return _write_varlen_int(len(payload)) + payload


def _stream_header_xml(name, stype, n_channels, srate, fmt, source_id) -> bytes:
    return (
        f'<?xml version="1.0"?>'
        f'<info>'
        f'<name>{name}</name>'
        f'<type>{stype}</type>'
        f'<channel_count>{n_channels}</channel_count>'
        f'<nominal_srate>{srate}</nominal_srate>'
        f'<channel_format>{fmt}</channel_format>'
        f'<source_id>{source_id}</source_id>'
        f'<desc/>'
        f'</info>'
    ).encode("utf-8")


def _stream_footer_xml(first_ts, last_ts, sample_count) -> bytes:
    return (
        f'<?xml version="1.0"?>'
        f'<info>'
        f'<first_timestamp>{first_ts:.10f}</first_timestamp>'
        f'<last_timestamp>{last_ts:.10f}</last_timestamp>'
        f'<sample_count>{sample_count}</sample_count>'
        f'</info>'
    ).encode("utf-8")


def _build_samples_chunk(stream_id: int,
                         timestamps: np.ndarray,
                         values: np.ndarray) -> bytes:
    """
    Build a single Samples chunk (tag 3) containing all markers.

    Each sample is encoded as:
      0x01                  — explicit-timestamp marker (non-zero)
      double (8 bytes LE)   — LSL timestamp
      float32 * 4 (16 B LE) — channel values

    This matches the encoding expected by pyxdf._read_chunk3 for float32
    irregular streams.
    """
    sample_count_bytes = _write_varlen_int(len(timestamps))
    parts = [sample_count_bytes]
    for i in range(len(timestamps)):
        ts_bytes  = b"\x01" + struct.pack("<d", float(timestamps[i]))
        val_bytes = values[i].astype("<f4").tobytes()
        parts.append(ts_bytes + val_bytes)
    content = b"".join(parts)
    return _build_chunk(3, stream_id, content)


# ── Marker log parsing ────────────────────────────────────────────────────────

# Matches lines like:
#   "Sent marker: 100 at timestamp: 115004.582350946 | P(MI): -1.000, P(REST): -1.000"
_SENT_RE = re.compile(
    r"Sent marker:\s+(\d+)\s+at timestamp:\s+([\d.]+)"
    r"\s+\|\s+P\(MI\):\s+([-\d.]+),\s+P\(REST\):\s+([-\d.]+)"
)


def _parse_marker_log(log_path: Path):
    """
    Parse every 'Sent marker' line in the UTIL_marker_stream log.

    Returns list of (marker_value: int, lsl_timestamp: float,
                      prob_mi: float, prob_rest: float).
    """
    entries = []
    with open(log_path, "r") as fh:
        for line in fh:
            m = _SENT_RE.search(line)
            if m:
                entries.append((
                    int(m.group(1)),
                    float(m.group(2)),
                    float(m.group(3)),
                    float(m.group(4)),
                ))
    return entries


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 64)
    print("XDF Marker Stream Repair Utility")
    print("=" * 64)

    # ── 1. Verify inputs ──────────────────────────────────────────────────────
    for p in (XDF_IN, LOG_FILE):
        if not p.exists():
            print(f"ERROR: required input not found: {p}")
            sys.exit(1)

    # ── 2. Load EEG XDF — extract time range ─────────────────────────────────
    print(f"\nLoading EEG XDF: {XDF_IN.name}")
    streams, _ = pyxdf.load_xdf(
        str(XDF_IN),
        dejitter_timestamps=False,
        synchronize_clocks=False,
    )

    eeg_stream = None
    for s in streams:
        typ   = s["info"].get("type", [""])[0].lower()
        srate = float(s["info"].get("nominal_srate", ["0"])[0])
        if typ == "eeg" and srate > 0:
            eeg_stream = s
            break

    if eeg_stream is None:
        print("ERROR: no continuous EEG stream found in input XDF.")
        sys.exit(1)

    eeg_t_start = float(eeg_stream["time_stamps"][0])
    eeg_t_end   = float(eeg_stream["time_stamps"][-1])
    eeg_name    = eeg_stream["info"]["name"][0]
    eeg_type    = eeg_stream["info"]["type"][0]

    print(f"  EEG stream : {eeg_name!r}  (type={eeg_type!r})")
    print(f"  EEG t_start: {eeg_t_start:.6f} s")
    print(f"  EEG t_end  : {eeg_t_end:.6f} s")

    # ── 3. Parse marker log ───────────────────────────────────────────────────
    print(f"\nParsing marker log: {LOG_FILE.name}")
    all_markers = _parse_marker_log(LOG_FILE)
    print(f"  Total 'Sent marker' entries in log: {len(all_markers)}")

    if not all_markers:
        print("ERROR: no marker entries found in log.")
        sys.exit(1)

    # ── 4. Select markers that belong to this XDF file ────────────────────────
    # Primary criterion: marker's LSL timestamp must fall within the EEG
    # stream's timestamp span, plus T_END_TOLERANCE_SEC.
    #
    # Rationale: the log covers both XDF recordings. The two EEG files have
    # non-overlapping time spans (old1: ~114825–116211, file 2: ~116222–117749).
    # The gap (~11 s) between them clearly separates the two recording segments.
    # Using the EEG time range is therefore both sufficient and unambiguous.
    selection_end = eeg_t_end + T_END_TOLERANCE_SEC
    matched = [
        m for m in all_markers
        if eeg_t_start <= m[1] <= selection_end
    ]

    print(f"\nMarker selection:")
    print(f"  Window: [{eeg_t_start:.6f}, {selection_end:.6f}]  "
          f"(EEG span + {T_END_TOLERANCE_SEC:.1f} s tolerance)")
    print(f"  Markers selected for this XDF: {len(matched)}")
    print(f"  Markers discarded (belong to other file or outside range): "
          f"{len(all_markers) - len(matched)}")

    if not matched:
        print("ERROR: no markers fell within the EEG time range. "
              "Check that the XDF and log are from the same session.")
        sys.exit(1)

    # ── 5. Build marker arrays ────────────────────────────────────────────────
    # timestamps: used as the XDF sample timestamps (the `<d>` field in each sample)
    # values: 4 float32 channels — [marker_val, lsl_ts, prob_mi, prob_rest]
    timestamps = np.array([m[1] for m in matched], dtype=np.float64)
    values     = np.array(
        [[float(m[0]), m[1], m[2], m[3]] for m in matched],
        dtype=np.float32,
    )

    # ── 6. Copy original XDF to output path ──────────────────────────────────
    if XDF_OUT.exists():
        print(f"\nWarning: overwriting existing output file: {XDF_OUT.name}")
    print(f"\nCopying original XDF → {XDF_OUT.name}")
    shutil.copy2(str(XDF_IN), str(XDF_OUT))

    # ── 7. Append marker stream to the copied file ────────────────────────────
    # Appending new chunks at the end of a valid XDF file is legal per the
    # spec: pyxdf processes chunks sequentially, so a StreamHeader that appears
    # after the existing stream footers is still correctly parsed as long as it
    # precedes its own Samples and Footer chunks.
    print("Appending reconstructed marker stream chunks...")
    with open(str(XDF_OUT), "ab") as fh:
        fh.write(_build_chunk(
            2, NEW_STREAM_ID,
            _stream_header_xml(
                MARKER_STREAM_NAME, MARKER_STREAM_TYPE,
                MARKER_N_CHANNELS, MARKER_SRATE,
                MARKER_FORMAT, MARKER_SOURCE_ID,
            ),
        ))
        fh.write(_build_samples_chunk(NEW_STREAM_ID, timestamps, values))
        fh.write(_build_chunk(
            6, NEW_STREAM_ID,
            _stream_footer_xml(
                float(timestamps[0]), float(timestamps[-1]), len(timestamps)
            ),
        ))

    # ── 8. Validate the repaired file ─────────────────────────────────────────
    print("\nValidating repaired file (re-loading with pyxdf)...")
    repaired_streams, _ = pyxdf.load_xdf(
        str(XDF_OUT),
        dejitter_timestamps=False,
        synchronize_clocks=False,
    )

    r_eeg    = None
    r_marker = None
    for s in repaired_streams:
        typ   = s["info"].get("type", [""])[0].lower()
        srate = float(s["info"].get("nominal_srate", ["0"])[0])
        name  = s["info"].get("name", [""])[0]
        if typ == "eeg" and srate > 0:
            r_eeg = s
        if typ in ("markers", "marker") and name == MARKER_STREAM_NAME:
            r_marker = s

    if r_eeg is None:
        print("ERROR: EEG stream missing from repaired file!")
        sys.exit(1)
    if r_marker is None:
        print("ERROR: reconstructed MarkerStream missing from repaired file!")
        sys.exit(1)

    recovered_vals = [int(row[0]) for row in r_marker["time_series"]]
    recovered_ts   = [float(row[1]) for row in r_marker["time_series"]]
    unique_vals    = sorted(set(recovered_vals))

    # ── 9. Validation summary ─────────────────────────────────────────────────
    print("\n" + "=" * 64)
    print("VALIDATION SUMMARY")
    print("=" * 64)
    print(f"  Input file              : {XDF_IN.name}")
    print(f"  Output file             : {XDF_OUT.name}")
    print(f"  EEG stream name / type  : {r_eeg['info']['name'][0]!r}"
          f" / {r_eeg['info']['type'][0]!r}")
    print(f"  Marker stream name/type : {r_marker['info']['name'][0]!r}"
          f" / {r_marker['info']['type'][0]!r}")
    print(f"  Recovered marker count  : {len(recovered_vals)}")
    print(f"  First marker timestamp  : {recovered_ts[0]:.6f}")
    print(f"  Last marker timestamp   : {recovered_ts[-1]:.6f}")
    print(f"  Unique marker values    : {unique_vals}")
    print("=" * 64)
    print("\nRepair complete.\n")


if __name__ == "__main__":
    main()
