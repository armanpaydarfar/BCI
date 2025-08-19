from collections import deque
import numpy as np
import mne
from pylsl import StreamInlet
from Utils.preprocessing import (
    initialize_filter_bank,
    apply_streaming_filters,
    get_valid_channel_mask_and_metadata,
    select_channels,
)

class EEGStreamState:
    def __init__(self, inlet: StreamInlet, config, mode = "motor", logger=None):
        self.inlet = inlet
        self.config = config
        self.logger = logger
        self.mode = mode

        # Override with ERRP-specific values if selected
        self.lowcut = config.LOWCUT
        self.highcut = config.HIGHCUT
        if self.mode == "errp":
            self.lowcut = config.LOWCUT_ERRP
            self.highcut = config.HIGHCUT_ERRP


        # Filtering
        self.filter_bank = initialize_filter_bank(
            fs=config.FS,
            lowcut=self.lowcut,
            highcut=self.highcut,
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
        self.subset_indices = None
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
                valid_channel_names, valid_raw, valid_indices = get_valid_channel_mask_and_metadata(
                    raw_chunk, all_ch_names, fs=self.config.FS, drop_mastoids=True
                )
                # Store indices of valid EEG channels in original stream
                            
                self.valid_channel_indices = valid_indices
                self.channel_names = valid_channel_names

                # Optional: select only motor-related EEG channels
                if self.mode == "motor":
                    motor_raw = select_channels(valid_raw, keep_channels = self.config.MOTOR_CHANNEL_NAMES)
                    self.subset_indices = [self.channel_names.index(ch) for ch in motor_raw.ch_names]
                    self.channel_names = motor_raw.ch_names
                    self.final_indices = [self.valid_channel_indices[i] for i in self.subset_indices]
                elif self.mode == "errp":
                    errp_raw = select_channels(valid_raw, keep_channels = self.config.ERRP_CHANNEL_NAMES)
                    self.subset_indices = [self.channel_names.index(ch) for ch in errp_raw.ch_names]
                    self.channel_names = errp_raw.ch_names
                    self.final_indices = [self.valid_channel_indices[i] for i in self.subset_indices]
                
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
        """
        Read raw channel labels from the LSL stream metadata without any renaming.
        Renaming/normalization should be done in the preprocessing helper only.
        """
        info = self.inlet.info()
        ch = info.desc().child("channels").child("channel")
        names = []

        while ch.name():
            label_node = ch.child("label").first_child()
            if not label_node:
                raise RuntimeError("Channel label missing in LSL stream metadata")
            names.append(label_node.value())  # <-- raw, unmodified label
            ch = ch.next_sibling()

        return names




    def _make_dummy_info(self):
        ch_names = self._get_channel_names()
        sfreq = self.config.FS
        ch_types = ['eeg'] * len(ch_names)
        return mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
