"""
Module: ear_logic.py
Owner: Timo (Timothy Muoki)
Branch: timo

═══════════════════════════════════════════════════════════
PURPOSE
═══════════════════════════════════════════════════════════
This module calculates the EYE ASPECT RATIO (EAR) to determine
if someone's eyes are open or closed. If eyes stay closed for
too many consecutive frames, it triggers a DROWSINESS alert.

This is the core "intelligence" of the drowsiness detection
system. The algorithm is based on the 2016 paper by Soukupová
& Čech: "Real-Time Eye Blink Detection using Facial Landmarks."

═══════════════════════════════════════════════════════════
HOW IT FITS IN THE PIPELINE
═══════════════════════════════════════════════════════════
In main.py, your module runs after Deno's face/eye detection.
IMPORTANT: Per the EyeGuard spec, face/eye detection only runs
when motion is detected. So your module only runs when BOTH
motion is present AND a face is found:

    if motion_result.detected:
        face_result = face_eye.detect(frame)           # Deno
        if face_result.face_detected:
            ear_result = ear_calc.calculate(            # YOUR MODULE
                face_result.left_eye,
                face_result.right_eye
            )
            alerts.check_drowsiness(ear_result)         # Ritchie

═══════════════════════════════════════════════════════════
THE EAR FORMULA (this is the key concept)
═══════════════════════════════════════════════════════════
Each eye has 6 landmark points:

         p1                p2
          ●────────────────●
         /                  \\
  p0 ●──                    ──● p3
         \\                  /
          ●────────────────●
         p5                p4

The Eye Aspect Ratio formula:

         ||p1 - p5|| + ||p2 - p4||
  EAR = ───────────────────────────
              2 × ||p0 - p3||

Where || || means Euclidean distance.

Numerator: Sum of two VERTICAL distances (eye height)
Denominator: Two times the HORIZONTAL distance (eye width)

When the eye is OPEN:
  - Vertical distances are large
  - EAR ≈ 0.30 or above

When the eye is CLOSED:
  - Vertical distances shrink toward zero
  - EAR < 0.25 (our threshold, per EyeGuard spec section 4.5)

When we BLINK:
  - EAR drops briefly then returns to normal
  - Only 1-3 frames at low EAR

When we're DROWSY:
  - EAR stays low for many consecutive frames (20+)
  - This is how we distinguish blinks from drowsiness

═══════════════════════════════════════════════════════════
EYEGUARD SPEC VALUES (from section 4.5)
═══════════════════════════════════════════════════════════
  EAR_THRESHOLD = 0.25
  CONSECUTIVE_FRAMES = 20

  if ear < EAR_THRESHOLD:
      frame_counter += 1
      if frame_counter >= CONSECUTIVE_FRAMES:
          trigger_alert()
  else:
      frame_counter = 0

These are the values specified in the project document. Use
them as defaults but they can be adjusted during testing.

═══════════════════════════════════════════════════════════
ALERT TIERS (maps to Ritchie's LED/buzzer logic)
═══════════════════════════════════════════════════════════
Your EARResult feeds into Ritchie's tiered alert system:

  EAR >= 0.25          → NORMAL  (green LED, no buzzer)
  EAR < 0.25           → WARNING (amber LED, short beep every 2s)
  EAR < 0.25 for 20+   → CRITICAL (red LED, continuous alarm)
    consecutive frames

To support this, your EARResult includes both eyes_closed (for
the warning tier) and drowsy (for the critical tier).

═══════════════════════════════════════════════════════════
EUCLIDEAN DISTANCE REFRESHER
═══════════════════════════════════════════════════════════
The distance between two points (x1,y1) and (x2,y2) is:

  d = sqrt((x2-x1)² + (y2-y1)²)

In Python with scipy:
  from scipy.spatial import distance
  d = distance.euclidean((x1,y1), (x2,y2))

Or with numpy:
  d = np.linalg.norm(np.array([x1,y1]) - np.array([x2,y2]))

Both work. scipy is slightly cleaner for this use case.

═══════════════════════════════════════════════════════════
STEP-BY-STEP IMPLEMENTATION GUIDE
═══════════════════════════════════════════════════════════
Step 1: Implement _compute_ear() for a single eye
  - This is pure math — you can test it without a camera
  - Use the sample data in the __main__ block

Step 2: Implement calculate() for both eyes
  - Average the left and right EAR values
  - Add the consecutive frame counter logic

Step 3: Test with the hardcoded sample data
  - Verify open eyes give EAR >= 0.25
  - Verify closed eyes give EAR < 0.25
  - Verify drowsy triggers after 20 consecutive frames

Step 4: Integrate with Deno's module to test with real eyes

═══════════════════════════════════════════════════════════
COMMON PITFALLS
═══════════════════════════════════════════════════════════
- Division by zero: If the horizontal distance is 0 (eyes at
  same point), return 0.0 instead of crashing
- Threshold tuning: 0.25 is the spec default. May need
  adjustment per individual — consider making it configurable.
- Don't confuse blinks with drowsiness. A blink lasts 1-3 frames
  (~100ms). Drowsiness means eyes stay closed for 20+ frames
  (~670ms at 30fps). The consecutive frame counter handles this.
- The eye point indices (p0-p5) map to the list indices [0]-[5]
  from Deno's module. Make sure you're using the right pairs:
  p1↔p5 means index [1]↔[5], p2↔p4 means index [2]↔[4]
"""

import numpy as np
from scipy.spatial import distance as dist
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class EARResult:
    """
    Result of EAR calculation and drowsiness detection.

    Attributes:
        ear_value: Average EAR across both eyes (0.0 to ~0.4).
                   Below 0.25 usually means eyes are closed (per spec).
        left_ear: EAR for the left eye only (for debugging)
        right_ear: EAR for the right eye only (for debugging)
        eyes_closed: True if ear_value is below the threshold.
                     Maps to WARNING tier (amber LED) in Ritchie's alerts.
        consecutive_frames: How many frames in a row the eyes
                            have been closed. Resets to 0 when
                            eyes open.
        drowsy: True if eyes have been closed for long enough
                to trigger a drowsiness alert (20+ frames per spec).
                Maps to CRITICAL tier (red LED + buzzer) in Ritchie's alerts.
    """
    ear_value: float = 0.0
    left_ear: float = 0.0
    right_ear: float = 0.0
    eyes_closed: bool = False
    consecutive_frames: int = 0
    drowsy: bool = False


class EARCalculator:
    def __init__(self, ear_threshold=0.25, consec_frames=20):
        """
        Initialize the EAR calculator.

        Args:
            ear_threshold: EAR values below this are considered "eyes closed."
                           Default 0.25 as specified in the EyeGuard system spec
                           (section 4.5). You may need to adjust based on testing:
                             - If alerts trigger too easily → lower the threshold
                             - If alerts don't trigger when eyes close → raise it
            consec_frames: Number of consecutive frames with eyes closed before
                           triggering a drowsiness alert. Default 20 as specified
                           in the EyeGuard spec. At 30fps:
                             - 20 frames = ~0.67 seconds (spec default)
                             - 15 frames = ~0.5 seconds (more sensitive)
                             - 30 frames = ~1.0 second (less sensitive)

        What to do:
            1. Store both parameters:
               self.ear_threshold = ear_threshold
               self.consec_frames = consec_frames

            2. Initialize the frame counter:
               self.closed_frame_counter = 0

               This counter goes UP by 1 each frame that eyes are closed,
               and RESETS to 0 when eyes open again.
        """
        # YOUR CODE HERE
        pass

    def _compute_ear(self, eye_points):
        """
        Calculate the Eye Aspect Ratio for a SINGLE eye.

        This is the core math of the entire drowsiness system.

        Args:
            eye_points: List of exactly 6 (x, y) tuples representing
                        the eye landmarks in this order:
                          [0] = outer corner  (p0)
                          [1] = upper outer   (p1)
                          [2] = upper inner   (p2)
                          [3] = inner corner  (p3)
                          [4] = lower inner   (p4)
                          [5] = lower outer   (p5)

        Returns:
            float: The Eye Aspect Ratio. Typically:
              - ~0.30 or above when eye is open
              - ~0.05-0.20 when eye is closed

        What to do:
            1. Calculate the two VERTICAL distances:

               vertical_1 = dist.euclidean(eye_points[1], eye_points[5])
               vertical_2 = dist.euclidean(eye_points[2], eye_points[4])

               These measure the "height" of the eye opening.
               When the eye closes, these distances shrink.

               Visually:
                   p1 ●          ● p2
                      |          |       ← vertical distances
                   p5 ●          ● p4

            2. Calculate the HORIZONTAL distance:

               horizontal = dist.euclidean(eye_points[0], eye_points[3])

               This measures the "width" of the eye.
               This stays roughly constant whether open or closed.

               Visually:
                   p0 ●────────────● p3
                      ← horizontal →

            3. Guard against division by zero:

               if horizontal == 0:
                   return 0.0

               This shouldn't normally happen, but malformed landmarks
               could cause it.

            4. Calculate and return EAR:

               ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
               return ear

               We multiply horizontal by 2 because we have TWO vertical
               measurements but only ONE horizontal. This normalizes
               the ratio.
        """
        # YOUR CODE HERE
        pass

    def calculate(self, left_eye, right_eye):
        """
        Calculate EAR for both eyes and determine drowsiness state.

        This is the method called by main.py every frame (when motion
        is detected and a face is found).

        Args:
            left_eye: List of 6 (x, y) tuples from Deno's module
            right_eye: List of 6 (x, y) tuples from Deno's module

        Returns:
            EARResult with all fields populated

        What to do:
            1. Compute EAR for each eye individually:
               left_ear = self._compute_ear(left_eye)
               right_ear = self._compute_ear(right_eye)

            2. Average both eyes:
               avg_ear = (left_ear + right_ear) / 2.0

               Why average? One eye might be partially occluded or
               at a slightly different angle. Averaging gives a more
               stable signal.

            3. Determine if eyes are currently closed:
               eyes_closed = avg_ear < self.ear_threshold

               This maps to the WARNING alert tier in Ritchie's module
               (amber LED, short beep every 2 seconds).

            4. Update the consecutive frame counter:
               if eyes_closed:
                   self.closed_frame_counter += 1
               else:
                   self.closed_frame_counter = 0

               This is the KEY logic: each frame with closed eyes
               increments the counter. One frame with open eyes
               resets it completely. This is how we distinguish
               a brief blink from sustained drowsiness.

            5. Check if drowsy (closed long enough):
               drowsy = self.closed_frame_counter >= self.consec_frames

               This maps to the CRITICAL alert tier in Ritchie's module
               (red LED fast flash, continuous buzzer alarm).

            6. Build and return the result:
               return EARResult(
                   ear_value=avg_ear,
                   left_ear=left_ear,
                   right_ear=right_ear,
                   eyes_closed=eyes_closed,
                   consecutive_frames=self.closed_frame_counter,
                   drowsy=drowsy
               )
        """
        # YOUR CODE HERE
        pass

    def reset(self):
        """
        Reset the consecutive frame counter.

        Called by main.py when no face is detected or when no motion
        is detected — we can't track eye state without a face, so we
        reset to avoid stale state.

        What to do:
            self.closed_frame_counter = 0
        """
        # YOUR CODE HERE
        pass


# ── FOR STANDALONE TESTING ──────────────────────────────
if __name__ == "__main__":
    """
    Test EAR calculation with known data:
        python src/ear_logic.py

    You don't need a camera for this! The test uses hardcoded
    eye landmarks to verify the math is correct.

    What you should see:
        - Open eye EAR >= 0.25 (probably around 0.3)
        - Closed eye EAR < 0.25 (probably around 0.05-0.10)
        - Drowsy alert triggers after 20 consecutive closed frames
        - Counter resets when eyes open

    What to implement:
        1. Create calculator:
           calc = EARCalculator(ear_threshold=0.25, consec_frames=20)

        2. Define sample OPEN eye landmarks:
           open_eye = [(10, 20), (12, 14), (18, 14), (22, 20), (18, 26), (12, 26)]

           These form a wide, tall eye shape. Vertical distances are large
           relative to horizontal → high EAR (above 0.25).

        3. Define sample CLOSED eye landmarks:
           closed_eye = [(10, 20), (12, 19), (18, 19), (22, 20), (18, 21), (12, 21)]

           Same horizontal distance but vertical distances are tiny
           (19→21 = 2 pixels) → low EAR (below 0.25).

        4. Test open eyes:
           result = calc.calculate(open_eye, open_eye)
           print(f"Open eyes  → EAR: {result.ear_value:.3f}, "
                 f"Closed: {result.eyes_closed}, Drowsy: {result.drowsy}")
           assert result.ear_value >= 0.25, "Open eye EAR should be >= 0.25!"
           assert not result.drowsy, "Should NOT be drowsy with open eyes!"

        5. Test closed eyes (simulate 25 consecutive frames to exceed
           the 20-frame threshold):
           for i in range(25):
               result = calc.calculate(closed_eye, closed_eye)
               print(f"Closed frame {i+1:2d} → EAR: {result.ear_value:.3f}, "
                     f"Consecutive: {result.consecutive_frames:2d}, "
                     f"Drowsy: {result.drowsy}")

           After 20 frames, drowsy should become True.

        6. Test reset when eyes open:
           result = calc.calculate(open_eye, open_eye)
           print(f"Eyes opened → Consecutive: {result.consecutive_frames}, "
                 f"Drowsy: {result.drowsy}")
           assert result.consecutive_frames == 0, "Counter should reset!"
           assert not result.drowsy, "Should not be drowsy after opening eyes!"

        7. Print "All tests passed!" if everything works
    """
    # YOUR CODE HERE
    pass