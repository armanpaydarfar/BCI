
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
import pandas as pd
from sklearn.metrics import confusion_matrix
# Visualization utils (UI draws)
from Utils.visualization import (
    draw_arrow_fill,
    draw_ball_fill,
    draw_fixation_cross,
    draw_time_balls,
)
# Experiment utilities
from Utils.experiment_utils import (
    generate_trial_sequence,
    display_multiple_messages_with_udp,
    LeakyIntegrator,
    RollingScaler,
    save_transform,
    load_transform
)

# UDP helper
from Utils.networking import send_udp_message

# --------- Runtime "globals" (wired by each driver right after init) ---------
# These names intentionally match what the original functions referenced.
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
    trial_probabilities = np.array(trial_probabilities)

    if trial_probabilities.shape[1] != 3:
        logger.log_event(f"❌ Error: Unexpected shape {trial_probabilities.shape}. Expected (N,3). Skipping save.")
        return

    for row in trial_probabilities:
        timestamp, prob_rest, prob_mi = row
        logger.log_decoder_output(
            trial=trial_number,
            timestamp=timestamp,
            prob_mi=prob_mi,
            prob_rest=prob_rest,
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

        # Draw UI elements
        draw_fixation_cross(screen_width, screen_height)
        draw_ball_fill(0, screen_width, screen_height)
        draw_arrow_fill(0, screen_width, screen_height)
        draw_time_balls(0, screen_width, screen_height)

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
        tuple: (fill_arrow, fill_ball) - Values between 0 and 1 indicating fill levels.
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

def classify_real_time(eeg_state, window_size_samples, all_probabilities, predictions, mode, leaky_integrator, update_recentering=True):
    global counter
    global Prev_T

    pygame.display.flip()
    pygame.event.get()  # Heartbeat to OS

    try:
        window, _ = eeg_state.get_baseline_corrected_window(window_size_samples)
    except ValueError:
        return leaky_integrator.accumulated_probability, predictions, all_probabilities

    # === Covariance Matrix ===
    cov_matrix = (window @ window.T) / np.trace(window @ window.T)

    if config.LEDOITWOLF:
        cov_matrix = np.array([LedoitWolf().fit(cov_matrix).covariance_])
    else:
        cov_matrix = np.expand_dims(cov_matrix, axis=0)
        shrinkage = Shrinkage(shrinkage=config.SHRINKAGE_PARAM)
        cov_matrix = shrinkage.fit_transform(cov_matrix)

    # === Adaptive Recentering ===
    if config.RECENTERING:
        cov_matrix = np.squeeze(cov_matrix, axis=0)

        if counter == 0 or Prev_T is None:
            Prev_T = cov_matrix

        T_test = geodesic_riemann(Prev_T, cov_matrix, 1 / (counter + 1))
        T_invsqrtm = invsqrtm(Prev_T)
        cov_matrix = T_invsqrtm @ cov_matrix @ T_invsqrtm.T
        cov_matrix = np.expand_dims(cov_matrix, axis=0)

    # === Classification ===
    probabilities = model.predict_proba(cov_matrix)[0]
    predicted_label = model.classes_[np.argmax(probabilities)]

    correct_label = 200 if mode == 0 else 100
    correct_class_idx = np.where(model.classes_ == correct_label)[0][0]
    current_confidence = probabilities[correct_class_idx]

    # === Determine if recentering update should occur ===
    should_update_T = False
    if config.RECENTERING and update_recentering:
        if config.USE_CONFIDENCE_GATE:
            correct_label = 200 if mode == 0 else 100
            correct_class_idx = np.where(model.classes_ == correct_label)[0][0]
            current_confidence = probabilities[correct_class_idx]
            predicted_correct = (predicted_label == correct_label)
            confident_enough = (current_confidence >= config.RECENTERING_CONFIDENCE_THRESHOLD)
            should_update_T = predicted_correct and confident_enough
        else:
            # Always update if gating is disabled
            should_update_T = True

    if should_update_T:
        Prev_T = T_test
        counter += 1


    # === Update Logs ===
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
        if now >= next_tick:
            current_confidence, predictions, all_probabilities = classify_real_time(
                eeg_state, window_size_samples,
                all_probabilities, predictions,
                mode, leaky_integrator,
                update_recentering=config.UPDATE_DURING_MOVE
            )
            next_tick += step_size 
            if all_probabilities:
                prob_mi, prob_rest = all_probabilities[-1][2], all_probabilities[-1][1]
                send_udp_message(
                    udp_socket_marker,
                    config.UDP_MARKER["IP"],
                    config.UDP_MARKER["PORT"],
                    f"{config.TRIGGERS['ROBOT_PROBS']},{prob_mi:.5f},{prob_rest:.5f}",
                    quiet=True
                )

            if current_confidence > 0:
                num_predictions += 1

            running_avg_confidence = leaky_integrator.update(current_confidence)

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
                    ["Stopping Robot"], [(255, 0, 0)], [0], duration=5,
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
        if FES_toggle == 1:
            send_udp_message(udp_socket_fes, config.UDP_FES["IP"], config.UDP_FES["PORT"], "FES_STOP", logger=logger)
            logger.log_event("FES_STOP signal sent after robot motion complete successfully.")
        else:
            logger.log_event("FES is disabled — no FES_STOP sent.")


    final_class = correct_class if running_avg_confidence >= config.RELAXATION_RATIO * accuracy_threshold else incorrect_class
    logger.log_event(f"Confidence at the end of motion: {running_avg_confidence:.2f} after {num_predictions} predictions")

    return final_class, all_probabilities, early_stop




def show_feedback(duration=5, mode=0, eeg_state = None):
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
        if FES_toggle == 1:
            send_udp_message(udp_socket_fes, config.UDP_FES["IP"], config.UDP_FES["PORT"], "FES_SENS_GO", logger=logger)
            FES_active = True
        else:
            logger.log_event("FES is disabled.")
            FES_active = False
    else:  # Blue Ball Mode (Rest)
        send_udp_message(udp_socket_marker, config.UDP_MARKER["IP"], config.UDP_MARKER["PORT"], config.TRIGGERS["REST_BEGIN"], logger=logger)
        FES_active = False

    clock = pygame.time.Clock()
    running_avg_confidence = 0.5  # Initial placeholder
    current_confidence = 0.5 # Initial placeholder for initial window updates
    next_tick = start_time + window_size  # Skip first second

    while time.time() - start_time < duration:
        eeg_state.update()

        now = time.time()
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


        if all_probabilities:
            prob_mi, prob_rest = all_probabilities[-1][2], all_probabilities[-1][1]
            send_udp_message(
                udp_socket_marker,
                config.UDP_MARKER["IP"],
                config.UDP_MARKER["PORT"],
                f"{config.TRIGGERS['MI_PROBS' if mode == 0 else 'REST_PROBS']},{prob_mi:.5f},{prob_rest:.5f}",
                quiet = True
            )

        running_avg_confidence = leaky_integrator.update(current_confidence)
        if FES_toggle == 1:
            FES_active = handle_fes_activation(mode, running_avg_confidence, FES_active)

        screen.fill(config.black)
        MI_fill, Rest_fill = calculate_fill_levels(running_avg_confidence, mode)

        if mode == 0:
            draw_arrow_fill(MI_fill, screen_width, screen_height)
            draw_fixation_cross(screen_width, screen_height)
            draw_ball_fill(Rest_fill, screen_width, screen_height)
            draw_time_balls(2, screen_width, screen_height)
            message = pygame.font.SysFont(None, 96).render(f"Move {config.ARM_SIDE.upper()} Arm", True, config.white)
        else:
            draw_ball_fill(Rest_fill, screen_width, screen_height)
            draw_fixation_cross(screen_width, screen_height)
            draw_arrow_fill(MI_fill, screen_width, screen_height)
            draw_time_balls(3, screen_width, screen_height)
            message = pygame.font.SysFont(None, 96).render("Rest", True, config.white)

        screen.blit(message, (screen_width // 2 - message.get_width() // 2, screen_height // 2 + 300))
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
    pygame.time.delay(300)  # ~300 ms delay to allow the visual feedback to complete rendering
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
    return final_class, running_avg_confidence, leaky_integrator, all_probabilities, earlystop_flag
