"""
test_eeg_stream_state.py

Guards the numerical / state-machine contract of `EEGStreamState`, on
which every realtime BCI driver depends (Plan §6 #5).

This test uses a `FakeInlet` that exposes:
  - `pull_chunk(timeout=0.0) -> (chunk_list, timestamps_list)`
  - `info()` → the nested `desc().child("channels").child("channel")`
    chain that `_get_channel_names` walks
    (`Utils/EEGStreamState.py:409-425`).

Citations under test (verified 2026-05-18):

  - Utils/EEGStreamState.py:14    class EEGStreamState
  - Utils/EEGStreamState.py:119   update()
  - Utils/EEGStreamState.py:222   compute_baseline
  - Utils/EEGStreamState.py:242   get_baseline_corrected_window
  - Utils/EEGStreamState.py:284   get_event_baseline_window
  - Utils/EEGStreamState.py:342   start_ea_accumulation
  - Utils/EEGStreamState.py:357   fit_ea_bootstrap

Each test feeds synthetic data; nothing requires a real LSL stream
or eegoSports device.
"""

from __future__ import annotations

import types
from itertools import count

import numpy as np
import pytest

import config
from Utils.EEGStreamState import EEGStreamState


# ─── fake LSL inlet machinery ─────────────────────────────────────────────

# Same 32-channel CA-209-style montage used by the rest of the repo.
# M1 / M2 are dropped by get_valid_channel_mask_and_metadata.
_FULL_CHANNELS = [
    "Fp1", "Fpz", "Fp2", "F7", "F3", "Fz", "F4", "F8",
    "FC5", "FC1", "FC2", "FC6", "M1", "T7", "C3", "Cz",
    "C4", "T8", "M2", "CP5", "CP1", "CP2", "CP6", "P7",
    "P3", "Pz", "P4", "P8", "POz", "O1", "Oz", "O2",
]


class _Node:
    """Minimal stand-in for an LSL XML node. Exposes the subset of methods
    EEGStreamState._get_channel_names walks (file:414-423)."""

    def __init__(self, tag, value="", children=None, next_=None):
        self._tag = tag
        self._value = value
        self._children = dict(children or {})
        self._next = next_

    def name(self):
        return self._tag

    def value(self):
        return self._value

    def child(self, tag):
        # Returning a leaf node for missing tags (rather than raising) mirrors
        # pylsl's behaviour where missing children are empty-name nodes.
        return self._children.get(tag, _Node(""))

    def first_child(self):
        # Used by _get_channel_names to get the label string node.
        return next(iter(self._children.values()), _Node(""))

    def next_sibling(self):
        return self._next or _Node("")


def _build_channel_chain(names):
    """Build the linked-list channel chain: each channel exposes
    `.child('label').first_child().value() == name` and `.next_sibling()`
    points to the next channel (terminating in an empty-name node)."""
    nxt = None
    for name in reversed(names):
        label_value = _Node("text", value=name)
        label = _Node("label", children={"text": label_value})
        ch = _Node("channel", children={"label": label}, next_=nxt)
        nxt = ch
    return nxt or _Node("")


def _build_info(names):
    """Build the info().desc().child('channels').child('channel') chain."""
    first_ch = _build_channel_chain(names)
    channels = _Node("channels", children={"channel": first_ch})
    desc = _Node("desc", children={"channels": channels})
    info = types.SimpleNamespace(desc=lambda: desc)
    return info


class FakeInlet:
    """Drive EEGStreamState without an LSL connection. Each call to
    pull_chunk consumes the next item from `chunks`; once exhausted,
    returns (empty list, empty list) — the inlet's "no new data" signal."""

    def __init__(self, names, chunks_with_ts):
        self._names = list(names)
        self._iter = iter(chunks_with_ts)
        self._info = _build_info(self._names)

    def info(self):
        return self._info

    def pull_chunk(self, timeout=0.0):
        try:
            chunk, ts = next(self._iter)
        except StopIteration:
            return [], []
        # EEGStreamState.update converts the chunk via np.array(chunk).T,
        # so input shape is (n_samples, n_channels) — matches LSL convention.
        return chunk, ts


def _make_chunk(n_samples, n_channels, *, seed=0, offset=0.0, scale=1.0):
    """Return a (samples-list-of-channel-lists, timestamps) pair."""
    rng = np.random.default_rng(seed)
    data = rng.standard_normal((n_samples, n_channels)) * scale + offset
    return data.tolist(), [float(i) / float(config.FS) for i in range(n_samples)]


# ─── fixtures ────────────────────────────────────────────────────────────

@pytest.fixture
def motor_state():
    """EEGStreamState in motor mode with a single 200-sample chunk pre-loaded.
    Caller drives `update()` to ingest the chunk."""
    chunk, ts = _make_chunk(200, len(_FULL_CHANNELS), seed=42, scale=10.0)
    inlet = FakeInlet(_FULL_CHANNELS, [(chunk, ts)])
    state = EEGStreamState(inlet, config, mode="motor")
    return state


# ─── tests ───────────────────────────────────────────────────────────────

class TestChannelDiscovery:
    def test_get_channel_names_walks_xml_chain(self):
        inlet = FakeInlet(_FULL_CHANNELS, [])
        state = EEGStreamState(inlet, config, mode="motor")
        names = state._get_channel_names()
        assert names == _FULL_CHANNELS

    def test_first_update_selects_motor_channels(self, motor_state):
        """update() should run the one-time channel selection pass and
        narrow `final_indices` down to the configured MOTOR_CHANNEL_NAMES
        (file:135-162)."""
        motor_state.update()
        assert motor_state.first_chunk_processed is True
        # MOTOR_CHANNEL_NAMES has 15 entries (config.py:26). All of them
        # are present in _FULL_CHANNELS minus the mastoids.
        assert len(motor_state.channel_names) == 15
        assert set(motor_state.channel_names) == set(config.MOTOR_CHANNEL_NAMES)
        assert len(motor_state.final_indices) == 15

    def test_buffer_fills_after_first_update(self, motor_state):
        motor_state.update()
        # After one 200-sample chunk, the rolling buffer holds 200 samples
        # regardless of selection (one entry per sample post-filter).
        assert len(motor_state.filtered_buffer) == 200
        assert len(motor_state.timestamps) == 200

    def test_second_update_no_data_is_noop(self, motor_state):
        motor_state.update()
        size_before = len(motor_state.filtered_buffer)
        # FakeInlet has no more chunks → second call returns ([],[]) and
        # update() early-exits.
        motor_state.update()
        assert len(motor_state.filtered_buffer) == size_before


class TestBaseline:
    def test_compute_baseline_then_corrected_window_has_zero_mean(self):
        """If the only data we see is constant + Gaussian noise, the
        baseline mean should subtract the constant offset, leaving a
        near-zero mean per channel."""
        OFFSET = 5.0
        chunk, ts = _make_chunk(2048, len(_FULL_CHANNELS), seed=7,
                                 offset=OFFSET, scale=0.1)
        inlet = FakeInlet(_FULL_CHANNELS, [(chunk, ts)])
        state = EEGStreamState(inlet, config, mode="motor")
        state.update()

        # 1 sec baseline at FS=512 → 512-sample mean.
        state.compute_baseline(duration_sec=1.0)
        assert state.baseline_mean is not None
        assert state.baseline_mean.shape == (15, 1)  # n_motor_channels, 1

        window, ts_out = state.get_baseline_corrected_window(window_size_samples=512)
        assert window.shape == (15, 512)
        assert len(ts_out) == 512
        # The streaming bandpass kills the DC offset, so the *post-filter*
        # signal has near-zero per-channel mean regardless of the input
        # offset. Baseline correction must therefore leave the mean small.
        per_channel_mean = window.mean(axis=1)
        assert np.all(np.abs(per_channel_mean) < 1.0)

    def test_compute_baseline_insufficient_data_raises(self):
        chunk, ts = _make_chunk(100, len(_FULL_CHANNELS), seed=0)
        inlet = FakeInlet(_FULL_CHANNELS, [(chunk, ts)])
        state = EEGStreamState(inlet, config, mode="motor")
        state.update()
        with pytest.raises(ValueError, match="Not enough data"):
            state.compute_baseline(duration_sec=1.0)

    def test_get_window_insufficient_data_raises(self):
        chunk, ts = _make_chunk(100, len(_FULL_CHANNELS), seed=0)
        inlet = FakeInlet(_FULL_CHANNELS, [(chunk, ts)])
        state = EEGStreamState(inlet, config, mode="motor")
        state.update()
        with pytest.raises(ValueError, match="Not enough data"):
            state.get_baseline_corrected_window(window_size_samples=512)


class TestEventBaselineWindow:
    def test_event_window_subtracts_pre_event_mean(self):
        """get_event_baseline_window should return a post-event slice with
        the pre-event mean subtracted (file:336-339), positioned at the
        sample closest to `event_timestamp`."""
        # Use deterministic values per channel: column c has value (c+1).
        # Filtering removes the DC, but the *relative* offset between the
        # pre- and post-event windows should be preserved. To make the
        # arithmetic easy we'll seed the buffer with a flat pattern, run it
        # through update(), and then compare against the on-buffer values.
        N = 1500
        chunk_values = np.tile(np.arange(len(_FULL_CHANNELS), dtype=float),
                               (N, 1))
        chunk = chunk_values.tolist()
        ts = [float(i) / config.FS for i in range(N)]
        inlet = FakeInlet(_FULL_CHANNELS, [(chunk, ts)])
        state = EEGStreamState(inlet, config, mode="motor")
        state.update()

        # Pick an event halfway through the buffer.
        n_buffered = len(state.timestamps)
        target_ts = state.timestamps[n_buffered // 2]
        baseline_samples = 256
        post_samples = 256

        window, ts_out = state.get_event_baseline_window(
            event_timestamp=target_ts,
            post_event_samples=post_samples,
            baseline_samples=baseline_samples,
        )
        assert window.shape == (15, post_samples)
        assert len(ts_out) == post_samples
        # After pre-event baseline subtraction the residual per-channel
        # mean should be tiny — the filtered signal is locally stationary.
        per_channel_mean = window.mean(axis=1)
        assert np.all(np.abs(per_channel_mean) < 1.0)

    def test_event_window_raises_when_not_enough_pre_event_samples(self):
        N = 600
        chunk, ts = _make_chunk(N, len(_FULL_CHANNELS), seed=1)
        inlet = FakeInlet(_FULL_CHANNELS, [(chunk, ts)])
        state = EEGStreamState(inlet, config, mode="motor")
        state.update()

        # Event right at the start → not enough pre-event samples.
        early_ts = state.timestamps[10]
        with pytest.raises(ValueError, match="pre-event"):
            state.get_event_baseline_window(
                event_timestamp=early_ts,
                post_event_samples=100,
                baseline_samples=256,
            )

    def test_event_window_raises_on_empty_buffer(self):
        inlet = FakeInlet(_FULL_CHANNELS, [])
        state = EEGStreamState(inlet, config, mode="motor")
        with pytest.raises(ValueError, match="Buffer is empty"):
            state.get_event_baseline_window(
                event_timestamp=0.0,
                post_event_samples=10,
                baseline_samples=10,
            )


class TestEAAccumulation:
    def test_start_then_fit_produces_symmetric_psd_reference(self):
        """Feed many epochs while accumulation is on; fit_ea_bootstrap
        should produce an n_ch x n_ch reference matrix (the matrix
        square-root inverse of the running mean covariance,
        file:369-374)."""
        # 30 pseudo-epochs of 256 samples each → 7680 samples total.
        # Stream them across multiple chunks so the incremental path is
        # actually exercised.
        n_ch = len(_FULL_CHANNELS)
        epoch_samples = 256
        n_epochs = 30
        total = epoch_samples * n_epochs
        chunk_size = 512
        chunks = []
        rng = np.random.default_rng(99)
        ts_counter = count()
        for start in range(0, total, chunk_size):
            stop = min(start + chunk_size, total)
            n = stop - start
            data = rng.standard_normal((n, n_ch)).tolist()
            ts = [float(next(ts_counter)) / config.FS for _ in range(n)]
            chunks.append((data, ts))

        inlet = FakeInlet(_FULL_CHANNELS, chunks)
        state = EEGStreamState(inlet, config, mode="motor")

        state.start_ea_accumulation(epoch_samples=epoch_samples)
        # Drain all chunks.
        for _ in range(len(chunks)):
            state.update()

        assert state._ea_cov_count > 0, "Accumulator never observed an epoch"
        ref = state.fit_ea_bootstrap(epoch_samples=epoch_samples,
                                     n_pseudo_epochs=n_epochs)
        # After motor-mode selection, the matrix is 15x15.
        n_motor = 15
        assert ref.shape == (n_motor, n_motor)
        # _inv_sqrt_psd returns a symmetric matrix.
        assert np.allclose(ref, ref.T, atol=1e-8)
        # Stored on the state too.
        assert state.ea_reference is ref

    def test_apply_ea_without_reference_raises(self):
        inlet = FakeInlet(_FULL_CHANNELS, [])
        state = EEGStreamState(inlet, config, mode="motor")
        dummy = np.zeros((15, 100))
        with pytest.raises(ValueError, match="No EA reference"):
            state.apply_ea_to_window(dummy)
