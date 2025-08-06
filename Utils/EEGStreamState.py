from collections import deque
import numpy as np
import mne
from pylsl import StreamInlet
from Utils.preprocessing import (
    initialize_filter_bank,
    apply_streaming_filters,
    get_valid_channel_mask_and_metadata,
    select_motor_channels,
)

class EEGStreamState:
    def __init__(self, inlet: StreamInlet, config, logger=None):
        self.inlet = inlet
        self.config = config
        self.logger = logger

        # Filtering
        self.filter_bank = initialize_filter_bank(
            fs=config.FS,
            lowcut=config.LOWCUT,
            highcut=config.HIGHCUT,
            notch_freqs=[60],
            notch_q=30
        )
        self.filter_state = {}

        # Buffers and state
        self.filtered_buffer = deque(maxlen=config.FILTER_BUFFER_SIZE)
        self.timestamps = deque(maxlen=config.FILTER_BUFFER_SIZE)
        self.baseline_mean = None

        # Channel selection
        self.channel_names = None
        self.valid_channel_indices = None
        self.motor_indices = None
        self.final_indices = None  # NEW: final indices used for slicing in real-time

        self.first_chunk_processed = False

    def update(self):
        try:
            # === Pull new chunk from LSL stream ===
            chunk, timestamps = self.inlet.pull_chunk(timeout=0.0)
            if not chunk or not timestamps:
                return  # No new data

            raw_chunk = np.array(chunk).T  # shape: (n_channels, n_samples)

            # === One-time channel selection logic ===
            if not self.first_chunk_processed:
                all_ch_names = self._get_channel_names()

                # Get valid EEG data, channel names, and MNE Raw object
                filtered_data, valid_channel_names, valid_raw = get_valid_channel_mask_and_metadata(
                    raw_chunk, all_ch_names
                )

                # Store indices of valid EEG channels in original stream
                self.valid_channel_indices = [all_ch_names.index(ch) for ch in valid_channel_names]
                self.channel_names = valid_channel_names

                # Optional: apply surface Laplacian
                if self.config.SURFACE_LAPLACIAN_TOGGLE:
                    valid_raw = mne.preprocessing.compute_current_source_density(valid_raw)

                # Optional: select only motor-related EEG channels
                if self.config.SELECT_MOTOR_CHANNELS:
                    motor_raw = select_motor_channels(valid_raw)
                    self.motor_indices = [self.channel_names.index(ch) for ch in motor_raw.ch_names]
                    self.channel_names = motor_raw.ch_names
                    self.final_indices = [self.valid_channel_indices[i] for i in self.motor_indices]
                else:
                    self.final_indices = self.valid_channel_indices

                self.first_chunk_processed = True

            # === Fast real-time slicing using precomputed indices ===
            if self.final_indices is not None:
                raw_chunk = raw_chunk[self.final_indices]

            # === Apply streaming filters ===
            filtered_chunk, self.filter_state = apply_streaming_filters(
                raw_chunk, self.filter_bank, self.filter_state
            )

            # === Append filtered samples to buffer ===
            for i in range(filtered_chunk.shape[1]):
                self.filtered_buffer.append(filtered_chunk[:, i])
                self.timestamps.append(timestamps[i])

        except Exception as e:
            if self.logger:
                self.logger.log_event(f"⚠️ Failed to update EEG stream: {e}")


    def compute_baseline(self, duration_sec=1.0):
        samples_needed = int(duration_sec * self.config.FS)
        if len(self.filtered_buffer) < samples_needed:
            raise ValueError("Not enough data in buffer to compute baseline.")

        buffer_array = np.array(self.filtered_buffer)[-samples_needed:]
        self.baseline_mean = buffer_array.mean(axis=0, keepdims=True).T  # shape: (n_channels, 1)

    def get_baseline_corrected_window(self, window_size_samples):
        if len(self.filtered_buffer) < window_size_samples:
            raise ValueError("Not enough data in buffer for window.")

        window = np.array(self.filtered_buffer)[-window_size_samples:].T  # shape: (n_channels, samples)
        if self.baseline_mean is not None:
            window -= self.baseline_mean
        return window, list(self.timestamps)[-window_size_samples:]
    
    def _get_channel_names(self):
        info = self.inlet.info()
        ch = info.desc().child("channels").child("channel")
        names = []
        rename_dict = {
            "FP1": "Fp1", "FPZ": "Fpz", "FP2": "Fp2",
            "FZ": "Fz", "CZ": "Cz", "PZ": "Pz", "POZ": "POz", "OZ": "Oz"
        }
        while ch.name():
            label_node = ch.child("label").first_child()
            if not label_node:
                raise RuntimeError("Channel label missing")
            raw_label = label_node.value()
            names.append(rename_dict.get(raw_label, raw_label))  # <--- unify here
            ch = ch.next_sibling()
        return names



    def _make_dummy_info(self):
        ch_names = self._get_channel_names()
        sfreq = self.config.FS
        ch_types = ['eeg'] * len(ch_names)
        return mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
