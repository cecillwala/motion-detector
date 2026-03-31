"""
Module: main.py
Owners: All team members
Branch: main (merged after all features are complete)

═══════════════════════════════════════════════════════════
PURPOSE
═══════════════════════════════════════════════════════════
This is the INTEGRATION file that wires all modules together
into the final EyeGuard system. It doesn't contain much logic
itself — it just calls each module in the correct order.

DO NOT modify this file on your feature branch. This file
should only be updated on the main branch after all modules
are merged and working.

═══════════════════════════════════════════════════════════
THE PIPELINE (per EyeGuard spec section 6.2)
═══════════════════════════════════════════════════════════

  1. Initialize camera (VideoCapture)
  2. Load dlib facial landmark model
  3. Capture video frame
  4. Run motion detection (frame differencing)
  5. If motion → run face detection           ← GATED
  6. If face → extract eye landmarks           ← GATED
  7. Calculate EAR, compute average
  8. If EAR < 0.25 for 20+ frames → trigger alert
  9. Display annotated frame with EAR overlay
  10. Repeat from step 3

IMPORTANT: Face/eye detection is GATED behind motion detection.
This is specified in EyeGuard spec section 4.4:
  "Motion detection acts as a wake trigger: the more
   computationally expensive eye-detection pipeline only
   activates when motion is present, conserving Raspberry
   Pi 3's processing resources."

This means if nobody is moving, we skip the expensive dlib
processing entirely, which keeps the Pi 3 responsive.

═══════════════════════════════════════════════════════════
ALERT TIERS (from spec section 4.3)
═══════════════════════════════════════════════════════════

  NORMAL:   Green LED steady, no buzzer
  WARNING:  Amber LED slow blink, short beep every 2s
  CRITICAL: Red LED fast flash, continuous buzzer alarm

These are handled by Ritchie's alerts.py module based on
the EAR result from Timo's module.

═══════════════════════════════════════════════════════════
INTEGRATION CHECKLIST
═══════════════════════════════════════════════════════════
Before running main.py, verify each module works standalone:

  □ python src/camera.py       → Shows live webcam feed
  □ python src/motion.py       → Green boxes around movement
  □ python src/face_eye.py     → Yellow dots on eye landmarks
  □ python src/ear_logic.py    → Math tests pass (EAR 0.25 threshold)
  □ python src/alerts.py       → Tier cycling (green/amber/red)
"""

# ═══════════════════════════════════════════════════════════
# IMPORTS
# Each import brings in one team member's module.
# If an import fails, that module has an issue.
# ═══════════════════════════════════════════════════════════
from camera import CameraStream
from motion import MotionDetector
from face_eye import FaceEyeDetector
from ear_logic import EARCalculator
from alerts import AlertManager
import cv2
import sys


# ═══════════════════════════════════════════════════════════
# CONFIGURATION
# Update PI_IP to match your Raspberry Pi's actual IP address.
# Use "localhost" if running directly on the Pi.
# ═══════════════════════════════════════════════════════════
PI_IP = "172.21.125.170"  # ← CHANGE THIS to your Pi's IP

CAMERA_SOURCE = "http://172.21.125.24:5000"


def main():
    """
    Main application loop.

    Initializes all modules, then runs the frame-by-frame pipeline
    until the user presses 'q' to quit.
    """

    # ── STEP 1: INITIALIZE ALL MODULES ──────────────────
    # Each module is created once and reused every frame.

    print("═" * 50)
    print("  EyeGuard — Motion & Drowsiness Detection")
    print("  COMP 494: Special Topics in Computer Science")
    print("═" * 50)
    print("\nInitializing modules...")
    source = ""

    if len(sys.argv) > 1:
        source = sys.argv[1]
        # If it's a plain number, convert to int (local camera index)
        if source.isdigit():
            source = int(source)
    else:
        source = CAMERA_SOURCE

    # Silvana's module: opens the webcam
    camera = CameraStream(source=source)
    print("  [✓] Camera initialized (640x480)")

    # Melanie & Stan's module: detects motion between frames
    # threshold=5000 means contours smaller than 5000 pixels are ignored
    motion = MotionDetector(threshold=5000)
    print("  [✓] Motion detector initialized")

    # Deno's module: finds faces and extracts eye landmarks
    # Make sure the .dat file is in the project root!
    face_eye = FaceEyeDetector()
    print("  [✓] Face/eye detector initialized (dlib 68-point)")

    # Timo's module: calculates EAR and tracks drowsiness
    # EAR threshold and consecutive frames per EyeGuard spec:
    #   EAR_THRESHOLD = 0.25
    #   CONSECUTIVE_FRAMES = 20
    ear_calc = EARCalculator(ear_threshold=0.25, consec_frames=20)
    print("  [✓] EAR calculator initialized (threshold=0.25, frames=20)")

    # Ritchie's module: handles tiered alerts (LED + buzzer + visual + MQTT)
    alerts = AlertManager(pi_ip=PI_IP)
    print("  [✓] Alert manager initialized (3-tier: green/amber/red)")

    print("\nAll modules ready. Press 'q' to quit.\n")

    # ── STEP 2: MAIN LOOP ──────────────────────────────
    # This runs ~30 times per second (depending on camera FPS
    # and processing speed).
    #
    # KEY DESIGN: Face/eye detection is GATED behind motion
    # detection. If no motion, we skip the expensive dlib
    # processing to conserve CPU on the Pi 3.

    ear_result = None  # Track across frames for alert state
    frame_count = 0

    try:
        while True:
            # ── CAPTURE FRAME (Silvana) ─────────────────
            # Step 3 in spec workflow: Capture video frame
            ret, frame = camera.read_frame()
            if not ret:
                print("Error: Failed to capture frame. Camera disconnected?")
                break
            frame_count += 1

            if frame_count % 3 == 0:
                # ── DETECT MOTION (Melanie & Stan) ──────────
                # Step 4 in spec workflow: Run motion detection
                # This runs EVERY frame — it's lightweight.
                motion_result = motion.detect(frame)
                motion.draw_on_frame(frame, motion_result)

                # ── GATED: FACE/EYE/EAR PIPELINE ───────────
                # Steps 5-7 in spec workflow:
                # Only run the expensive face/eye detection when
                # motion is detected. This conserves CPU resources
                # on the Raspberry Pi 3 (per spec section 4.4).
                if motion_result.detected:

                    # ── DETECT FACE & EYES (Deno) ───────────
                    # Step 5-6: If motion → run face detection
                    # If face → extract eye landmarks (points 37-48)
                    face_result = face_eye.detect(frame)
                    face_eye.draw_on_frame(frame, face_result)

                    # ── CALCULATE EAR (Timo) ────────────────
                    # Step 7: Calculate EAR for both eyes
                    if face_result.face_detected:
                        ear_result = ear_calc.calculate(
                            face_result.left_eye,
                            face_result.right_eye
                        )
                    else:
                        # Face not found even though motion exists
                        # (could be non-human motion). Reset EAR state.
                        ear_calc.reset()
                        ear_result = None

                else:
                    # ── NO MOTION — SKIP EXPENSIVE PROCESSING ──
                    # No motion detected. Skip face/eye detection
                    # entirely to save CPU. Reset drowsiness state
                    # so we don't carry stale alerts.
                    ear_calc.reset()
                    ear_result = None

                # ── TRIGGER ALERTS (Ritchie) ────────────────
                # Steps 8-9: Check results and trigger tiered alerts
                # This runs EVERY frame regardless of motion.

                # Check motion (updates alert state, may publish MQTT)
                alerts.check_motion(motion_result)

                # Check drowsiness (uses ear_result, may be None)
                alerts.check_drowsiness(ear_result)

                # Update hardware LEDs and buzzer (no-op on laptop)
                alerts.update_leds()

                # Draw visual overlays (borders, text, simulated LEDs)
                alerts.draw_alerts(frame)

                # ── DISPLAY THE FRAME ───────────────────────
                # Step 9: Display annotated frame with EAR overlay
                # At this point, the frame has annotations from:
                #   - Melanie & Stan (green motion boxes)
                #   - Deno (yellow eye landmarks, if motion present)
                #   - Ritchie (tier borders, LED indicators, status bar)
                cv2.imshow("EyeGuard - Motion & Drowsiness Detection", frame)

                # ── CHECK FOR QUIT ──────────────────────────
                if cv2.waitKey(30) & 0xFF == ord('q'):
                    break

    finally:
        # ── CLEANUP ─────────────────────────────────────
        # Always runs, even if an error occurred above.
        print("\nShutting down EyeGuard...")
        camera.release()
        alerts.cleanup()
        cv2.destroyAllWindows()
        print("System shut down cleanly.")


# ═══════════════════════════════════════════════════════════
# ENTRY POINT
# Run with: python src/main.py
# ═══════════════════════════════════════════════════════════
if __name__ == "__main__":
    main()
