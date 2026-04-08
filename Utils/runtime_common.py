
# runtime_common.py
# Centralizes functions used by both ExperimentDriver_Online and ExperimentDriver_Bimanual
# with minimal edits to math/logic. Runtime objects are "wired" in by the drivers at startup.

import time
import numpy as np
import pygame
from sklearn.covariance import LedoitWolf
from pyriemann.estimation import Shrinkage
from pyriemann.utils.geodesic import geodesic_riemann
from pyriemann.utils.base import invsqrtm
from pyriemann.tangentspace import tangent_space
import pandas as pd
from sklearn.metrics import confusion_matrix
# Visualization utils (UI draws)
from Utils.visualization import (
    draw_class_feedback_cues,
    draw_class_fixation_idle,
)
# Experiment utilities
from Utils.experiment_utils import (
    generate_trial_sequence,
    LeakyIntegrator,
    RollingScaler,
)

# UDP helper
from Utils.networking import send_udp_message,display_multiple_messages_with_udp

# --------- Runtime "globals" (wired by each driver right after init) ---------
# These names intentionally match what the original functions referenced.
# Contract: drivers must assign these before calling runtime functions. For example,
# `udp_socket_fes` and `FES_toggle` must be valid if FES is enabled.
config = None
logger = None
model = None

# Surfaces/screen geometry used by draw helpers
screen = None
screen_width = None
screen_height = None

# UDP sockets
udp_socket_marker = None
udp_socket_robot  = None
udp_socket_fes    = None

# Flags
FES_toggle = None

# Adaptive recentering state
Prev_T = None
counter = 0
Prev_T_beta = None
counter_beta = 0
_decoder_checked = False


# ----------------- Common helpers -----------------

def log_confusion_matrix_from_trial_summary(logger):
    df = pd.read_csv(logger.trial_summary_path)

    # Separate into valid and ambiguous trials
    ambiguous_trials = df[df["Predicted Label"].isna()]
    valid_trials = df.dropna(subset=["Predicted Label"])

    valid_trials.loc[:, "Predicted Label"] = valid_trials["Predicted Label"].astype(int)
    valid_trials.loc[:, "True Label"] = valid_trials["True Label"].astype(int)

    # Count correct predictions
    correct = (valid_trials["Predicted Label"] == valid_trials["True Label"]).sum()
    incorrect = len(valid_trials) - correct
    ambiguous = len(ambiguous_trials)
    total = correct + incorrect + ambiguous

    # Generate confusion matrix
    if not valid_trials.empty:
        cm = confusion_matrix(
            valid_trials["True Label"], valid_trials["Predicted Label"],
            labels=[200, 100]
        )
        logger.log_event("Confusion Matrix (Correct/Incorrect Only):")
        logger.log_event(f"  Actual 200 (MI)    | Predicted 200 (MI): {cm[0][0]} | Predicted 100 (REST): {cm[0][1]}")
        logger.log_event(f"  Actual 100 (REST)  | Predicted 200 (MI): {cm[1][0]} | Predicted 100 (REST): {cm[1][1]}")
    else:
        logger.log_event("No non-ambiguous trials to compute confusion matrix.")

    # Log summary stats
    if total:
        percent_correct_incl_ambiguous = (correct / total) * 100
        percent_correct_excl_ambiguous = (correct / (correct + incorrect)) * 100 if (correct + incorrect) > 0 else 0
        logger.log_event(f"✅ % Total Accuracy (Including ambiguous): {percent_correct_incl_ambiguous:.2f}%")
        logger.log_event(f"✅ % Decision Accuracy (Excluding ambiguous): {percent_correct_excl_ambiguous:.2f}%")
        logger.log_event(f"⚠️ Ambiguous trials (not counted in exclusive metric): {ambiguous}")
    else:
        logger.log_event("No trials available to compute statistics.")



def append_trial_probabilities_to_csv(trial_probabilities, mode, trial_number,
                                      predicted_label, early_cutout,
                                      mi_threshold, rest_threshold, logger,
                                      phase):
    correct_class = 200 if mode == 0 else 100
    trial_probabilities = np.array(trial_probabilities, dtype=float)

    # Expect: [ts, P(REST)_inst, P(MI)_inst, P(MI)_avg, P(REST)_avg]
    if trial_probabilities.ndim != 2 or trial_probabilities.shape[1] != 5:
        logger.log_event(
            f"❌ Error: Unexpected shape {trial_probabilities.shape}. "
            f"Expected (N,5): [ts, P(REST)_inst, P(MI)_inst, P(MI)_avg, P(REST)_avg]. Skipping save."
        )
        return

    for row in trial_probabilities:
        ts, prest_inst, pmi_inst, pmi_avg, prest_avg = row.tolist()
        logger.log_decoder_output(
            trial=trial_number,
            timestamp=ts,
            prob_mi_inst=pmi_inst,
            prob_rest_inst=prest_inst,
            prob_mi_avg=pmi_avg,
            prob_rest_avg=prest_avg,
            true_label=correct_class,
            predicted_label=predicted_label,
            early_cutout=early_cutout,
            mi_threshold=mi_threshold,
            rest_threshold=rest_threshold,
            phase=phase
        )

    logger.log_event(
        f"✅ Logged {len(trial_probabilities)} rows for Trial {trial_number} | "
        f"True: {correct_class}, Predicted: {predicted_label}, Early Cut: {early_cutout}, Phase: {phase}"
    )



def display_fixation_period(duration=3, eeg_state=None):
    """
    Displays a blank screen with a fixation cross for a given duration.
    
    Parameters:
    - duration (int): Time in seconds for which the fixation period lasts.
    - eeg_state: Optional EEGState object to be updated during the fixation period.
    """
    start_time = time.time()
    clock = pygame.time.Clock()

    while time.time() - start_time < duration:
        # Fill screen with background color
        pygame.display.get_surface().fill(config.black)

        vis_style = getattr(config, "CLASS_VISUAL_STYLE", "classic")
        draw_class_fixation_idle(vis_style, screen_width, screen_height)

        pygame.display.flip()

        # Update EEG buffer if provided
        if eeg_state is not None:
            eeg_state.update()

        # Handle quit events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

        clock.tick(60)


# Interpolation function to compute fill amount between SHAPE_MIN and SHAPE_MAX
def interpolate_fill(value):
    return max(0, min(1, (value - config.SHAPE_MIN) / (config.SHAPE_MAX - config.SHAPE_MIN)))

def calculate_fill_levels(running_avg_confidence, mode):
    """
    Determines the fill levels for both MI (arrow) and Rest (ball) based on accumulated probability.

    Parameters:
        running_avg_confidence (float): The leaky-integrated probability estimate.
        mode (int): 0 for MI trial (fill square), 1 for Rest trial (fill ball).

    Returns:
        tuple: (fill_arrow, fill_ball) - Values between 0 and 1 indicating fill levels
        for the MI arrow and REST ball UI (roles swap in Rest mode).
    """
    # Ensure probability stays within configured bounds
    prob = max(0, min(1, running_avg_confidence))
    prob_inverse = 1 - prob  # Inverse probability for the other shape


    # Determine fill levels
    fill_mi = interpolate_fill(prob) if prob >= config.SHAPE_MIN else 0  # MI shape fills when prob > SHAPE_MIN
    fill_rest = interpolate_fill(prob_inverse) if prob_inverse >= config.SHAPE_MIN else 0  # Rest shape fills when 1-prob > SHAPE_MIN

    # Swap roles if in Rest mode
    if mode == 1:
        return fill_rest, fill_mi  # Flip values for Rest condition
    return fill_mi, fill_rest  # Default for MI mode


def handle_fes_activation(mode, running_avg_confidence, fes_active):
    """
    Manages the activation of sensory FES based on the running average probability.

    Parameters:
        mode (int): 0 for MI (Motor Imagery), 1 for Rest.
        running_avg_confidence (float): Current probability estimate.
        fes_active (bool): Current state of FES (True if active, False if inactive).
        logger: LoggerManager instance used for structured logging.

    Returns:
        bool: Updated FES state after processing.
    """
    # Thresholding logic (kept here for traceability):
    # - MI (mode==0): running_avg_confidence > 0.5 -> activate
    # - Rest (mode==1): running_avg_confidence < 0.5 -> deactivate
    # Determine if FES should be active:
    # - If mode is MI (0) and confidence > 0.5 → Turn on FES
    # - If mode is Rest (1) and confidence < 0.5 → Turn on FES
    fes_should_be_active = (mode == 0 and running_avg_confidence > 0.5) or \
                           (mode == 1 and running_avg_confidence < 0.5)

    # Activate FES if needed
    if fes_should_be_active and not fes_active:
        if FES_toggle == 1:
            send_udp_message(udp_socket_fes, config.UDP_FES["IP"], config.UDP_FES["PORT"], "FES_SENS_GO", logger=logger)
            logger.log_event("Sensory FES activated.")
        else:
            logger.log_event("FES toggle is off — activation skipped.")
        return True

    # Deactivate FES if needed
    elif not fes_should_be_active and fes_active:
        if FES_toggle == 1:
            send_udp_message(udp_socket_fes, config.UDP_FES["IP"], config.UDP_FES["PORT"], "FES_STOP", logger=logger)
            logger.log_event("Sensory FES stopped.")
        else:
            logger.log_event("FES toggle is off — stop command skipped.")
        return False

    # No change in state
    return fes_active


def _is_xgb_bundle(obj):
    return isinstance(obj, dict) and "model" in obj and "scaler" in obj and "label_to_bin" in obj


def _covariance_from_window(window_2d):
    cov = (window_2d @ window_2d.T)
    tr = np.trace(cov)
    if tr <= 0:
        tr = 1e-12
    return cov / tr


def _shrink_single_cov(cov, raw_window=None):
    def _runtime_shrinkage_lambda() -> float:
        backend = str(getattr(config, "DECODER_BACKEND", "mdm")).strip().lower()
        if backend.startswith("xgb"):
            return float(getattr(config, "SHRINKAGE_PARAM_XGB", getattr(config, "SHRINKAGE_PARAM", 0.1)))
        return float(getattr(config, "SHRINKAGE_PARAM_MDM", getattr(config, "SHRINKAGE_PARAM", 0.02)))

    if config.LEDOITWOLF:
        if raw_window is None:
            raise RuntimeError(
                "LEDOITWOLF requires the raw EEG window to be passed to _shrink_single_cov. "
                "LedoitWolf must be fit to raw observations (n_timepoints x n_channels), not the precomputed covariance."
            )
        # Fit LW to raw data to estimate the optimal shrinkage coefficient, then apply it
        # to the already trace-normalised covariance (raw_window is n_channels x n_timepoints).
        lam = LedoitWolf().fit(raw_window.T).shrinkage_
        n = cov.shape[0]
        return (1 - lam) * cov + lam * (np.trace(cov) / n) * np.eye(n)
    lam = _runtime_shrinkage_lambda()
    return np.squeeze(
        Shrinkage(shrinkage=lam).fit_transform(np.expand_dims(cov, axis=0)),
        axis=0,
    )


def _tangent_with_fitted_ref_single(cov, state_name="mu"):
    if not isinstance(model, dict):
        raise RuntimeError("XGBoost decoder bundle is not loaded.")

    ref_key = "tangent_ref_beta" if state_name == "beta" else "tangent_ref_mu"
    ref = model.get(ref_key, None)
    if ref is None:
        raise RuntimeError(
            f"XGBoost bundle missing required fitted {ref_key}. "
            "Retrain XGB model with fitted tangent-space references."
        )
    return tangent_space(np.expand_dims(cov, axis=0), ref, metric="riemann")[0]


def _adaptive_recenter_cov(cov, state_name="mu", *, update_recentering: bool = True):
    """
    Apply online adaptive recentering to one covariance matrix.
    Uses independent state per branch (mu/beta) to avoid cross-band leakage.
    """
    if not config.RECENTERING:
        return cov

    global Prev_T, counter, Prev_T_beta, counter_beta

    if state_name == "beta":
        if counter_beta == 0 or Prev_T_beta is None:
            Prev_T_beta = cov
        T_test = geodesic_riemann(Prev_T_beta, cov, 1 / (counter_beta + 1))
        T_invsqrtm = invsqrtm(Prev_T_beta)
        cov_rec = T_invsqrtm @ cov @ T_invsqrtm.T
        if bool(update_recentering):
            Prev_T_beta = T_test
            counter_beta += 1
        return cov_rec

    if counter == 0 or Prev_T is None:
        Prev_T = cov
    T_test = geodesic_riemann(Prev_T, cov, 1 / (counter + 1))
    T_invsqrtm = invsqrtm(Prev_T)
    cov_rec = T_invsqrtm @ cov @ T_invsqrtm.T
    if bool(update_recentering):
        Prev_T = T_test
        counter += 1
    return cov_rec


def _default_erd_bands_from_config():
    """
    Match offline `xgb_feature_pipeline` convention when a bundle has no
    `feature_spec["erd_bands"]` (legacy or hand-built pickle).
    """
    bands = getattr(config, "XGB_ERD_BANDS", None)
    if bands is not None:
        return [tuple(map(float, b)) for b in bands]
    # Default to mu-only unless configured otherwise.
    return [(float(config.LOWCUT), float(config.HIGHCUT))]


def _log_bandpower(signal, fs, band):
    lo, hi = float(band[0]), float(band[1])
    n_t = signal.shape[1]
    freqs = np.fft.rfftfreq(n_t, d=1.0 / fs)
    fft = np.fft.rfft(signal, axis=1)
    power = (np.abs(fft) ** 2) / float(max(n_t, 1))
    mask = (freqs >= lo) & (freqs < hi)
    if np.any(mask):
        bp = power[:, mask].mean(axis=1)
    else:
        bp = np.zeros(signal.shape[0], dtype=float)
    return np.log10(bp + 1e-12)


def _build_xgb_features(eeg_state, window_size_samples, *, update_recentering: bool = True):
    """
    Build online XGBoost features to match offline ordering:
    [cov_mu] + [cov_beta] + [erd]
    """
    feature_spec = model.get("feature_spec", {})
    use_cov_mu = bool(feature_spec.get("use_cov_mu", True))
    use_cov_beta = bool(feature_spec.get("use_cov_beta", False))
    n_cov_mu = int(feature_spec.get("n_cov_mu", 0))
    n_cov_beta = int(feature_spec.get("n_cov_beta", 0))
    n_erd = int(feature_spec.get("n_erd", 0))

    if n_erd > 0:
        mu_win, beta_win, _ = eeg_state.get_multiband_baseline_corrected_window(window_size_samples)
    elif use_cov_beta:
        mu_win, beta_win, _ = eeg_state.get_multiband_baseline_corrected_window(window_size_samples)
    else:
        mu_win, _ = eeg_state.get_baseline_corrected_window(window_size_samples)
        beta_win = None

    feature_blocks = []

    if use_cov_mu:
        cov_mu = _shrink_single_cov(_covariance_from_window(mu_win), mu_win)
        cov_mu = _adaptive_recenter_cov(cov_mu, "mu", update_recentering=update_recentering)
        feature_blocks.append(_tangent_with_fitted_ref_single(cov_mu, "mu"))

    if use_cov_beta:
        if beta_win is None:
            raise RuntimeError("XGBoost model expects beta covariance features, but beta stream is unavailable.")
        cov_beta = _shrink_single_cov(_covariance_from_window(beta_win), beta_win)
        cov_beta = _adaptive_recenter_cov(cov_beta, "beta", update_recentering=update_recentering)
        feature_blocks.append(_tangent_with_fitted_ref_single(cov_beta, "beta"))

    if n_erd > 0:
        bands = feature_spec.get("erd_bands") or _default_erd_bands_from_config()
        apply_csd = bool(feature_spec.get("apply_csd_erd_only", False))
        if apply_csd:
            try:
                from Utils.xgb_feature_pipeline import apply_surface_laplacian_csd
                mu_erd = apply_surface_laplacian_csd(mu_win, list(eeg_state.channel_names), fs=config.FS)
                beta_erd = apply_surface_laplacian_csd(beta_win, list(eeg_state.channel_names), fs=config.FS)
            except Exception as e:
                logger.log_event(f"⚠️ Online CSD failed, falling back to non-CSD ERD: {e}")
                mu_erd = mu_win
                beta_erd = beta_win
        else:
            mu_erd = mu_win
            beta_erd = beta_win

        baseline_mu, baseline_beta = eeg_state.get_multiband_baseline_segments()
        if baseline_mu is None or baseline_beta is None:
            raise RuntimeError("Missing baseline segments for ERD feature computation.")
        if apply_csd:
            try:
                from Utils.xgb_feature_pipeline import apply_surface_laplacian_csd
                baseline_mu = apply_surface_laplacian_csd(baseline_mu, list(eeg_state.channel_names), fs=config.FS)
                baseline_beta = apply_surface_laplacian_csd(baseline_beta, list(eeg_state.channel_names), fs=config.FS)
            except Exception:
                pass

        erd_vals = []
        for (lo, hi) in bands:
            if float(hi) <= float(config.HIGHCUT):
                w = _log_bandpower(mu_erd, config.FS, (lo, hi))
                b = _log_bandpower(baseline_mu, config.FS, (lo, hi))
            else:
                w = _log_bandpower(beta_erd, config.FS, (lo, hi))
                b = _log_bandpower(baseline_beta, config.FS, (lo, hi))
            erd_vals.append((w - b).reshape(-1, 1))
        erd_vec = np.hstack(erd_vals).reshape(-1)
        feature_blocks.append(erd_vec)

    feat = np.concatenate(feature_blocks, axis=0).reshape(1, -1)
    if n_cov_mu and use_cov_mu and feat.shape[1] < n_cov_mu:
        raise RuntimeError("Feature construction failed for covariance mu block.")
    if n_cov_beta and use_cov_beta and feat.shape[1] < (n_cov_mu + n_cov_beta):
        raise RuntimeError("Feature construction failed for covariance beta block.")
    if n_erd and feat.shape[1] < (n_cov_mu + n_cov_beta + n_erd):
        raise RuntimeError("Feature construction failed for ERD block.")
    return feat


def _predict_with_active_decoder(eeg_state, window_size_samples, *, update_recentering: bool = True):
    if not _is_xgb_bundle(model):
        raise RuntimeError("XGBoost decoder bundle is not loaded.")
    feat = _build_xgb_features(eeg_state, window_size_samples, update_recentering=update_recentering)
    expected = int(model["scaler"].n_features_in_)
    if feat.shape[1] != expected:
        raise RuntimeError(f"XGBoost feature length mismatch. expected={expected}, got={feat.shape[1]}")
    x = model["scaler"].transform(feat)
    probs_bin = model["model"].predict_proba(x)[0]
    rest_label = min(model["label_to_bin"], key=lambda k: model["label_to_bin"][k])
    mi_label = max(model["label_to_bin"], key=lambda k: model["label_to_bin"][k])
    rest_idx = int(model["label_to_bin"][rest_label])
    mi_idx = int(model["label_to_bin"][mi_label])
    prob_rest = float(probs_bin[rest_idx])
    prob_mi = float(probs_bin[mi_idx])
    pred = int(mi_label if prob_mi >= prob_rest else rest_label)
    return np.array([prob_rest, prob_mi], dtype=float), pred

def classify_real_time(eeg_state, window_size_samples, all_probabilities, predictions, mode, leaky_integrator, update_recentering=True):
    global _decoder_checked
    global counter
    global Prev_T

    pygame.display.flip()
    pygame.event.get()  # Heartbeat to OS

    if _is_xgb_bundle(model):
        try:
            probabilities, predicted_label = _predict_with_active_decoder(
                eeg_state,
                window_size_samples,
                update_recentering=bool(update_recentering),
            )
        except ValueError:
            return leaky_integrator.accumulated_probability, predictions, all_probabilities

        if not _decoder_checked:
            expected = int(model["scaler"].n_features_in_)
            logger.log_event(f"XGBoost decoder initialized. expected_features={expected}")
            _decoder_checked = True

        correct_label = 200 if mode == 0 else 100
        current_confidence = float(probabilities[1] if correct_label == 200 else probabilities[0])
        predictions.append(int(predicted_label))
        all_probabilities.append([time.time(), float(probabilities[0]), float(probabilities[1])])
        return current_confidence, predictions, all_probabilities

    # ---------------- Legacy MDM path (unchanged behavior) ----------------
    try:
        window, _ = eeg_state.get_baseline_corrected_window(window_size_samples)
    except ValueError:
        return leaky_integrator.accumulated_probability, predictions, all_probabilities

    cov_matrix = (window @ window.T) / np.trace(window @ window.T)

    if config.LEDOITWOLF:
        lam = LedoitWolf().fit(window.T).shrinkage_
        n = cov_matrix.shape[0]
        cov_matrix = np.array([(1 - lam) * cov_matrix + lam * (np.trace(cov_matrix) / n) * np.eye(n)])
    else:
        cov_matrix = np.expand_dims(cov_matrix, axis=0)
        lam = float(getattr(config, "SHRINKAGE_PARAM_MDM", getattr(config, "SHRINKAGE_PARAM", 0.02)))
        shrinkage = Shrinkage(shrinkage=lam)
        cov_matrix = shrinkage.fit_transform(cov_matrix)

    if config.RECENTERING:
        cov_matrix = np.squeeze(cov_matrix, axis=0)
        if counter == 0 or Prev_T is None:
            Prev_T = cov_matrix
        T_test = geodesic_riemann(Prev_T, cov_matrix, 1 / (counter + 1))
        T_invsqrtm = invsqrtm(Prev_T)
        cov_matrix = T_invsqrtm @ cov_matrix @ T_invsqrtm.T
        cov_matrix = np.expand_dims(cov_matrix, axis=0)

    probabilities = model.predict_proba(cov_matrix)[0]
    predicted_label = model.classes_[np.argmax(probabilities)]

    correct_label = 200 if mode == 0 else 100
    correct_class_idx = np.where(model.classes_ == correct_label)[0][0]
    current_confidence = probabilities[correct_class_idx]

    should_update_T = False
    if config.RECENTERING and update_recentering:
        # Adaptive reference update is unconditional (matches the historical
        # behavior of USE_CONFIDENCE_GATE == 0).
        should_update_T = True

    if should_update_T:
        Prev_T = T_test
        counter += 1

    predictions.append(predicted_label)
    all_probabilities.append([time.time(), probabilities[0], probabilities[1]])
    return current_confidence, predictions, all_probabilities




def hold_messages_and_classify(messages, colors, offsets, duration, mode, udp_socket, udp_ip, udp_port,
                               eeg_state, leaky_integrator):
    """
    Holds visual messages on the screen while running real-time EEG classification in the background.
    Classifies every STEP_SIZE seconds using the most recent WINDOW_SIZE seconds of EEG data.

    Returns:
    - int: Final classification result (200 or 100)
    - list: All classification probabilities
    - bool: Whether an early stop occurred
    """
    font = pygame.font.SysFont(None, 72)
    start_time = time.time()
    early_stop = False

    step_size = config.STEP_SIZE  # e.g. 1/16s
    window_size = config.CLASSIFY_WINDOW / 1000  # ms → seconds
    window_size_samples = int(window_size * config.FS)

    correct_class = 200 if mode == 0 else 100
    incorrect_class = 100 if mode == 0 else 200

    min_predictions_before_stop = config.MIN_PREDICTIONS
    num_predictions = 0
    accuracy_threshold = config.THRESHOLD_MI if mode == 0 else config.THRESHOLD_REST 

    all_probabilities = []
    running_average_list = []
    predictions = []
    running_avg_confidence = 0.5
    current_confidence = 0.5

    next_tick = time.time()  # Classify immediately
    pygame.display.update()
    clock = pygame.time.Clock()

    while time.time() - start_time < duration:
        now = time.time()

        # === Update EEG Buffer ===
        eeg_state.update()

        # === Draw Messages ===
        pygame.display.get_surface().fill((0, 0, 0))
        for i, text in enumerate(messages):
            message = font.render(text, True, colors[i])
            pygame.display.get_surface().blit(
                message,
                (pygame.display.get_surface().get_width() // 2 - message.get_width() // 2,
                 pygame.display.get_surface().get_height() // 2 + offsets[i])
            )
        pygame.display.flip()

        # === Classify every step_size seconds ===
        just_classified = False
        if now >= next_tick:
            current_confidence, predictions, all_probabilities = classify_real_time(
                eeg_state, window_size_samples,
                all_probabilities, predictions,
                mode, leaky_integrator,
                update_recentering=config.UPDATE_DURING_MOVE
            )
            next_tick += step_size 
            if all_probabilities and getattr(config, "SEND_PROBS", False):
                prob_mi, prob_rest = all_probabilities[-1][2], all_probabilities[-1][1]
                send_udp_message(
                    udp_socket_marker,
                    config.UDP_MARKER["IP"],
                    config.UDP_MARKER["PORT"],
                    f"{config.TRIGGERS['ROBOT_PROBS']},{prob_mi:.5f},{prob_rest:.5f}",
                    quiet=True
                )
            just_classified = True

            if current_confidence > 0:
                num_predictions += 1

            running_avg_confidence = leaky_integrator.update(current_confidence)
            if just_classified and all_probabilities:
                ts, prest_inst, pmi_inst = all_probabilities[-1]
                if mode == 0:  # MI context
                    pmi_avg   = running_avg_confidence
                    prest_avg = 1.0 - running_avg_confidence
                else:          # REST context
                    prest_avg = running_avg_confidence
                    pmi_avg   = 1.0 - running_avg_confidence
                all_probabilities[-1] = [ts, prest_inst, pmi_inst, pmi_avg, prest_avg]

            if num_predictions >= min_predictions_before_stop and running_avg_confidence < config.RELAXATION_RATIO * accuracy_threshold:
                early_stop = True

                logger.log_event(f"Early stop triggered! Confidence: {running_avg_confidence:.2f} after {num_predictions} predictions")

                send_udp_message(udp_socket_marker, config.UDP_MARKER["IP"], config.UDP_MARKER["PORT"], config.TRIGGERS["ROBOT_EARLYSTOP"], logger=logger)
                send_udp_message(udp_socket_marker, config.UDP_MARKER["IP"], config.UDP_MARKER["PORT"], config.TRIGGERS["ROBOT_END"], logger=logger)

                if FES_toggle == 1:
                    send_udp_message(udp_socket_fes, config.UDP_FES["IP"], config.UDP_FES["PORT"], "FES_STOP", logger=logger)
                    logger.log_event("FES_STOP signal sent due to early stop.")
                else:
                    logger.log_event("FES is disabled — no FES_STOP sent.")

                display_multiple_messages_with_udp(
                    ["Stopping Robot"], [(255, 0, 0)], [0], duration=3,
                    udp_messages=[config.ROBOT_OPCODES["STOP"]], udp_socket=udp_socket, udp_ip=udp_ip, udp_port=udp_port, logger=logger
                )
                break

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return None

        clock.tick(60)

    if not early_stop:
        send_udp_message(udp_socket_marker, config.UDP_MARKER["IP"], config.UDP_MARKER["PORT"], config.TRIGGERS["ROBOT_END"], logger=logger)

    final_class = correct_class if running_avg_confidence >= config.RELAXATION_RATIO * accuracy_threshold else incorrect_class
    logger.log_event(f"Confidence at the end of motion: {running_avg_confidence:.2f} after {num_predictions} predictions")

    return final_class, all_probabilities, early_stop




def show_feedback(duration=5, mode=0, eeg_state=None,
                  headline_text=None,
                  subtext=None,
                  object_text=None):
    """
    Displays feedback animation, collects EEG data, and performs real-time classification
    using a sliding window approach with early stopping based on posterior probabilities.
    """
    start_time = time.time()
    step_size = config.STEP_SIZE  # Sliding window step size (seconds)
    window_size = config.CLASSIFY_WINDOW / 1000  # Convert ms to seconds
    window_size_samples = int(window_size * config.FS)
    step_size_samples = int(step_size * config.FS)
    FES_active = False
    all_probabilities = []
    predictions = []
    running_avg_list = []
    leaky_integrator = LeakyIntegrator(alpha=config.INTEGRATOR_ALPHA)  # Confidence smoothing
    min_predictions = config.MIN_PREDICTIONS
    earlystop_flag = False

    classification_results = []
    # Define the correct class based on mode
    # Define the correct class based on mode
    correct_class = 200 if mode == 0 else 100  # 200 = Right Arm MI, 100 = Rest
    incorrect_class = 100 if mode == 0 else 200  # The opposite class

    # accuracy threshold based on mode
    accuracy_threshold = config.THRESHOLD_MI if mode == 0 else config.THRESHOLD_REST 
    opposed_threshold = config.THRESHOLD_REST if mode == 0 else config.THRESHOLD_MI
    # Preprocess the baseline dataset before feedback starts
    # Preprocess the baseline dataset before feedback starts
    pygame.display.flip()

    # Send UDP triggers
    if mode == 0:  # Red Arrow Mode (Motor Imagery)
        send_udp_message(udp_socket_marker, config.UDP_MARKER["IP"], config.UDP_MARKER["PORT"], config.TRIGGERS["MI_BEGIN"], logger=logger)
    else:  # Blue Ball Mode (Rest)
        send_udp_message(udp_socket_marker, config.UDP_MARKER["IP"], config.UDP_MARKER["PORT"], config.TRIGGERS["REST_BEGIN"], logger=logger)

    clock = pygame.time.Clock()
    running_avg_confidence = 0.5  # Initial placeholder
    current_confidence = 0.5 # Initial placeholder for initial window updates
    next_tick = start_time + window_size  # Skip first second

    while time.time() - start_time < duration:
        eeg_state.update()

        now = time.time()
        just_classified = False
        if now >= next_tick:
            current_confidence, predictions, all_probabilities = classify_real_time(
                eeg_state,
                window_size_samples,
                all_probabilities,
                predictions,
                mode,
                leaky_integrator
            )
            next_tick += step_size 
            if all_probabilities and getattr(config, "SEND_PROBS", False):
                prob_mi, prob_rest = all_probabilities[-1][2], all_probabilities[-1][1]
                send_udp_message(
                    udp_socket_marker,
                    config.UDP_MARKER["IP"],
                    config.UDP_MARKER["PORT"],
                    f"{config.TRIGGERS['MI_PROBS' if mode == 0 else 'REST_PROBS']},{prob_mi:.5f},{prob_rest:.5f}",
                    quiet=True
                )
            just_classified=True


        running_avg_confidence = leaky_integrator.update(current_confidence)
        if FES_toggle == 1:
            FES_active = handle_fes_activation(mode, running_avg_confidence, FES_active)

        screen.fill(config.black)
        MI_fill, Rest_fill = calculate_fill_levels(running_avg_confidence, mode)
        if just_classified and all_probabilities:
            ts, prest_inst, pmi_inst = all_probabilities[-1]
            if mode == 0:  # MI trial: confidence is P(MI)
                pmi_avg   = running_avg_confidence
                prest_avg = 1.0 - running_avg_confidence
            else:          # REST trial: confidence is P(REST)
                prest_avg = running_avg_confidence
                pmi_avg   = 1.0 - running_avg_confidence
            all_probabilities[-1] = [ts, prest_inst, pmi_inst, pmi_avg, prest_avg]

        # -------------------------------------------------
        # Core visualization (classic vs modern — config.CLASS_VISUAL_STYLE)
        # -------------------------------------------------
        vis_style = getattr(config, "CLASS_VISUAL_STYLE", "classic")
        accum_bar = running_avg_confidence if str(vis_style).lower() == "modern" else None
        if mode == 0:
            draw_class_feedback_cues(
                vis_style, 0, MI_fill, Rest_fill, screen_width, screen_height, 2,
                accumulation=accum_bar,
            )
            cue_color = config.red
        else:
            draw_class_feedback_cues(
                vis_style, 1, MI_fill, Rest_fill, screen_width, screen_height, 3,
                accumulation=accum_bar,
            )
            cue_color = config.blue

        # -------------------------------------------------
        # Text overlay (backwards compatible)
        # -------------------------------------------------
        headline_font = pygame.font.SysFont(None, 96)
        subtext_font = pygame.font.SysFont(None, 56)
        object_font = pygame.font.SysFont(None, 52)

        # Legacy fallback if no text passed
        if headline_text is None:
            if mode == 0:
                headline_text = f"Move {config.ARM_SIDE.upper()} Arm"
            else:
                headline_text = "Rest"

        headline_surface = headline_font.render(headline_text, True, cue_color)
        screen.blit(
            headline_surface,
            (screen_width // 2 - headline_surface.get_width() // 2,
            screen_height // 2 + 260)
        )

        if subtext is not None:
            sub_surface = subtext_font.render(subtext, True, cue_color)
            screen.blit(
                sub_surface,
                (screen_width // 2 - sub_surface.get_width() // 2,
                screen_height // 2 + 330)
            )

        if object_text is not None:
            obj_surface = object_font.render(object_text, True, config.white)
            screen.blit(
                obj_surface,
                (screen_width // 2 - obj_surface.get_width() // 2,
                screen_height // 2 - 360)
            )
        pygame.display.flip()
        clock.tick(60)
        # --- Early-stop logic (supports correct-only or either-threshold) ---
        hit_correct   = (len(predictions) >= min_predictions) and (running_avg_confidence >= accuracy_threshold)
        hit_incorrect = (len(predictions) >= min_predictions) and (running_avg_confidence <= (1 - opposed_threshold))

        should_earlystop = hit_correct or (config.EARLYSTOP_MODE == "either" and hit_incorrect)
        if should_earlystop:
            earlystop_flag = True

            # Figure out which class triggered the stop (for logging/triggers)
            if hit_correct:
                stop_reason = "correct"
                trigger_key = "MI_EARLYSTOP" if mode == 0 else "REST_EARLYSTOP"
            else:
                stop_reason = "incorrect"
                trigger_key = "REST_EARLYSTOP" if mode == 0 else "MI_EARLYSTOP"

            logger.log_event(
                f"Early stopping triggered ({stop_reason}). "
                f"Confidence={running_avg_confidence:.2f}, "
                f"min_preds={min_predictions}, "
                f"mode={'MI' if mode==0 else 'REST'}"
            )

            # Stop FES if active
            if FES_toggle == 1:
                send_udp_message(udp_socket_fes, config.UDP_FES["IP"], config.UDP_FES["PORT"], "FES_STOP", logger=logger)
            else:
                logger.log_event("FES is disabled.")

            # Emit the appropriate EARLYSTOP trigger
            send_udp_message(
                udp_socket_marker,
                config.UDP_MARKER["IP"],
                config.UDP_MARKER["PORT"],
                config.TRIGGERS[trigger_key],
                logger=logger
            )
            break

    
    pygame.display.flip()
    # Final Decision
    if running_avg_confidence >= accuracy_threshold:
        final_class = correct_class
    elif running_avg_confidence <= (1 - opposed_threshold):
        final_class = incorrect_class
    else:
        final_class = None  # Ambiguous zone
    
    if final_class is not None:
        logger.log_event(
            f"Final decision: {final_class}, Confidence for correct({correct_class}) class: "
            f"{running_avg_confidence:.2f}, at sample size {len(predictions)}"
        )
    else:
        logger.log_event(
            f"Ambiguous final decision — no threshold met. Confidence: {running_avg_confidence:.2f}, "
            f"MI threshold: {config.THRESHOLD_MI}, REST threshold: {config.THRESHOLD_REST}, "
            f"Samples: {len(predictions)}"
        )
    if FES_toggle == 1 and FES_active:
        send_udp_message(udp_socket_fes, config.UDP_FES["IP"], config.UDP_FES["PORT"], "FES_STOP", logger=logger)
    else:
        logger.log_event("FES disable not needed.")


    send_udp_message(udp_socket_marker, config.UDP_MARKER["IP"], config.UDP_MARKER["PORT"], config.TRIGGERS["MI_END" if mode==0 else "REST_END"], logger=logger)
    pygame.time.delay(300)  # ~300 ms delay to allow the visual feedback to complete rendering
    return final_class, running_avg_confidence, leaky_integrator, all_probabilities, earlystop_flag
