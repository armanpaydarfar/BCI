import logging
import json
import csv
import time
import re
from pathlib import Path
from datetime import datetime

class LoggerManager:
    def __init__(self, log_base: Path, run_id: str = None, mode: str = "offline"):
        timestamp_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        session_label = f"{mode.upper()}_{timestamp_str}"
        run_suffix = f"_{run_id}" if run_id else ""

        # Final path: logs/ONLINE_YYYY-MM-DD_HH-MM-SS[_run-XXX]
        self.log_dir = log_base / "logs" / f"{session_label}{run_suffix}"
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.event_log_path = self.log_dir / "event_log.txt"
        logging.basicConfig(
            level=logging.INFO,
            filename=self.event_log_path,
            filemode='w',
            format='%(asctime)s [%(levelname)s] %(message)s',
        )

        # Also output to terminal
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter('[%(levelname)s] %(message)s')
        console_handler.setFormatter(console_formatter)
        logging.getLogger().addHandler(console_handler)


        self.decoder_csv_path = self.log_dir / "decoder_output.csv"
        with open(self.decoder_csv_path, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                "Trial", "Timestamp", "P(MI)", "P(REST)",
                "True Label", "Predicted Label", "Early Cutout",
                "MI Threshold", "REST Threshold", "Phase"
            ])


        self.trial_summary_path = self.log_dir / "trial_summary.csv"
        with open(self.trial_summary_path, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                "Trial", "True Label", "Predicted Label", "Early Cutout",
                "Accuracy Threshold", "Confidence", "Num Predictions"
            ])


    def log_event(self, message: str, level="info"):
        getattr(logging, level)(message)

    def log_decoder_output(self, trial, timestamp, prob_mi, prob_rest, true_label, predicted_label,
                        early_cutout, mi_threshold, rest_threshold, phase):
        with open(self.decoder_csv_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                trial, timestamp, prob_mi, prob_rest,
                true_label, predicted_label, early_cutout,
                mi_threshold, rest_threshold, phase
            ])


    def log_trial_summary(self, trial_number, true_label, predicted_label,
                        early_cutout, accuracy_threshold, confidence, num_predictions):
        with open(self.trial_summary_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                trial_number, true_label, predicted_label,
                early_cutout, accuracy_threshold, confidence, num_predictions
            ])


    def save_config_snapshot(self, config: dict):
        with open(self.log_dir / "config_snapshot.json", 'w') as f:
            json.dump(config, f, indent=2)


    @classmethod
    def auto_detect_from_subject(cls, subject: str, base_path: Path, mode: str = "offline"):
        subject_path = base_path / f"sub-{subject}"
        eeg_dirs = list(subject_path.glob("ses-*/eeg"))

        all_xdfs = []
        for eeg_dir in eeg_dirs:
            all_xdfs.extend(eeg_dir.glob("*.xdf"))

        if not all_xdfs:
            fallback_dir = subject_path / "ses-Debug"
            logger = cls(log_base=fallback_dir, mode=mode)
            logger.log_base = fallback_dir  # ✅ Ensure attribute always exists
            logger.is_fallback = True       # ✅ Optional but useful flag
            logger.log_event("No .xdf files found — fallback to test mode.")
            return logger

        latest_xdf = max(all_xdfs, key=lambda f: f.stat().st_mtime)

        try:
            size1 = Path(latest_xdf).resolve().stat().st_size
            time.sleep(1)
            size2 = Path(latest_xdf).resolve().stat().st_size

            if size2 > size1:
                session_dir = latest_xdf.parents[1]  # eeg/ → ses-XXX
                run_id = next((p for p in latest_xdf.stem.split('_') if p.startswith("run-")), None)
                logger = cls(log_base=session_dir, run_id=run_id, mode=mode)
                logger.log_base = session_dir       # ✅ Explicitly set it
                logger.is_fallback = False
                logger.log_event(f"Auto-detected active recording: {latest_xdf.name}")
                return logger

        except Exception:
            pass

        fallback_dir = subject_path / "ses-Debug"
        logger = cls(log_base=fallback_dir, mode=mode)
        logger.log_base = fallback_dir          # ✅ Ensure log_base even on fallback
        logger.is_fallback = True
        logger.log_event(f"Most recent file '{latest_xdf.name}' was not growing — fallback to test mode.")
        return logger
