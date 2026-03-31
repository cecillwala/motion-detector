"""
EAR (Eye Aspect Ratio) - Full Logic & Calculation Module
Drowsiness detection core. No camera, no UI, no hardware.
"""

import numpy as np
from scipy.spatial import distance as dist
from dataclasses import dataclass

# --- Constants ---
EAR_THRESHOLD = 0.25       # Below this → eye is closing
CONSECUTIVE_FRAMES = 20    # Frames needed to confirm drowsiness (~667ms at 30fps)

# dlib 68-point landmark indices (0-based)
LEFT_EYE_INDICES  = list(range(36, 42))   # Points 37–42
RIGHT_EYE_INDICES = list(range(42, 48))   # Points 43–48


# ─────────────────────────────────────────
# STEP 1: Extract eye regions from landmarks
# ─────────────────────────────────────────

def extract_eye_regions(landmarks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Slice the left and right eye points from the full 68-point landmark array.

    Args:
        landmarks: (68, 2) array of (x, y) coordinates from dlib.

    Returns:
        (left_eye, right_eye) — each a (6, 2) numpy array.
    """
    left_eye  = landmarks[LEFT_EYE_INDICES]   # Points 37–42
    right_eye = landmarks[RIGHT_EYE_INDICES]  # Points 43–48
    return left_eye, right_eye


# ─────────────────────────────────────────
# STEP 2: Compute EAR for a single eye
# ─────────────────────────────────────────

def compute_EAR(eye_points: np.ndarray) -> float:
    """
    Compute Eye Aspect Ratio for one eye.

    Landmark layout (image coordinate space, origin top-left):

             p2(x,y)  p3(x,y)
    p1(x,y)                    p4(x,y)
             p6(x,y)  p5(x,y)

    Formula:
        A = ||p2 - p6||  (vertical, outer column)
        B = ||p3 - p5||  (vertical, inner column)
        C = ||p1 - p4||  (horizontal, eye width)

        EAR = (A + B) / (2 * C)

    EAR open  → ~0.30 (vertical distances are large)
    EAR closed → ~0.03 (vertical distances collapse)

    Args:
        eye_points: (6, 2) array ordered p1..p6.

    Returns:
        EAR as float. Returns 0.0 on degenerate input (C == 0).
    """
    if eye_points.shape != (6, 2):
        raise ValueError(f"Expected (6, 2) array, got {eye_points.shape}")

    A = dist.euclidean(eye_points[1], eye_points[5])  # ||p2 - p6||
    B = dist.euclidean(eye_points[2], eye_points[4])  # ||p3 - p5||
    C = dist.euclidean(eye_points[0], eye_points[3])  # ||p1 - p4||

    # Euclidean: sqrt((x2-x1)^2 + (y2-y1)^2)
    # C is eye width — used to normalize for distance/scale invariance

    return 0.0 if C == 0.0 else (A + B) / (2.0 * C)


# ─────────────────────────────────────────
# STEP 3: Average both eyes
# ─────────────────────────────────────────

def average_EAR(left_eye: np.ndarray, right_eye: np.ndarray) -> float:
    """
    Average EAR across both eyes for a stable, noise-reduced signal.
    Single-eye readings are vulnerable to partial occlusion and head tilt.

    Args:
        left_eye:  (6, 2) landmark array.
        right_eye: (6, 2) landmark array.

    Returns:
        Mean EAR float.
    """
    left_ear  = compute_EAR(left_eye)
    right_ear = compute_EAR(right_eye)
    return (left_ear + right_ear) / 2.0


# ─────────────────────────────────────────
# STEP 4: Drowsiness decision logic
# ─────────────────────────────────────────

def evaluate_drowsiness(ear: float, frame_counter: int) -> tuple[int, bool]:
    """
    Stateless debounce logic — caller owns the frame_counter state.

    Rules:
        EAR < threshold  → increment counter
        counter >= 20    → alert triggered
        EAR >= threshold → hard reset to 0 (eyes open = awake)

    Why 20 frames?
        A blink lasts ~150–400ms = 5–12 frames at 30fps.
        20 frames (~667ms) filters blinks, catches genuine closure.

    Args:
        ear:           Current averaged EAR.
        frame_counter: Consecutive below-threshold frame count.

    Returns:
        (updated_counter, alert_triggered)
    """
    if ear < EAR_THRESHOLD:
        frame_counter += 1
        return frame_counter, frame_counter >= CONSECUTIVE_FRAMES

    return 0, False  # Eyes open — reset immediately


# ─────────────────────────────────────────
# STEP 5: Alert placeholder
# ─────────────────────────────────────────

def trigger_alert() -> None:
    """
    Drowsiness confirmed. Plug in your alert mechanism here.
    Examples: audio beep, push notification, GPIO buzzer, API call.
    """
    pass


# ─────────────────────────────────────────
# STEP 6: Single entry point for one frame
# ─────────────────────────────────────────

def process_frame(landmarks: np.ndarray, frame_counter: int) -> tuple[float, int, bool]:
    """
    Full EAR pipeline for a single frame.
    Feed this the 68-point landmarks from your detector.

    Args:
        landmarks:     (68, 2) dlib landmark array.
        frame_counter: Current consecutive drowsy frame count (caller-owned).

    Returns:
        (ear, updated_frame_counter, alert_triggered)

    Usage:
        ear, frame_counter, alert = process_frame(landmarks, frame_counter)
        if alert:
            trigger_alert()
    """
    left_eye, right_eye = extract_eye_regions(landmarks)
    ear                 = average_EAR(left_eye, right_eye)
    frame_counter, alert = evaluate_drowsiness(ear, frame_counter)

    return ear, frame_counter, alert

@dataclass
class EARResult:
    ear_value: float = 0.0
    left_ear: float = 0.0
    right_ear: float = 0.0
    eyes_closed: bool = False
    consecutive_frames: int = 0
    drowsy: bool = False


class EARCalculator:
    def __init__(self, ear_threshold=0.25, consec_frames=20):
        self.ear_threshold = ear_threshold
        self.consec_frames = consec_frames
        self.closed_frame_counter = 0

    def calculate(self, left_eye, right_eye):
        left_ear = compute_EAR(np.array(left_eye))
        right_ear = compute_EAR(np.array(right_eye))
        avg_ear = (left_ear + right_ear) / 2.0

        eyes_closed = avg_ear < self.ear_threshold

        if eyes_closed:
            self.closed_frame_counter += 1
        else:
            self.closed_frame_counter = 0

        drowsy = self.closed_frame_counter >= self.consec_frames

        return EARResult(
            ear_value=avg_ear,
            left_ear=left_ear,
            right_ear=right_ear,
            eyes_closed=eyes_closed,
            consecutive_frames=self.closed_frame_counter,
            drowsy=drowsy
        )

    def reset(self):
        self.closed_frame_counter = 0