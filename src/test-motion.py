"""
test_motion.py
--------------
Tests motion.py without needing a display window.
Works with opencv-python-headless.

Run with:
    python test_motion.py

What it does:
    - Opens your webcam
    - Captures 60 frames
    - Runs motion detection on each frame
    - Prints results to the terminal
    - Saves one annotated frame as 'test_output.jpg' so you can see it worked
"""

import cv2
import sys
import os

# ── Make sure Python can find motion.py ──────────────────
# If motion.py is in src/, use this line instead:
# sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from motion import MotionDetector, MotionResult

print("=" * 55)
print("  EyeGuard – Motion Detection Test (headless)")
print("=" * 55)

# ── Step 1: Open the webcam ───────────────────────────────
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("\nERROR: Could not open camera.")
    print("  - Check it is plugged in")
    print("  - Check no other app is using it")
    sys.exit(1)

w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"\n[OK] Camera opened at {w}x{h}")

# ── Step 2: Create the detector ───────────────────────────
detector = MotionDetector(threshold=5000)
print("[OK] MotionDetector created\n")

# ── Step 3: Capture 60 frames and print results ───────────
print("  Capturing 60 frames — MOVE in front of the camera!\n")
print(f"  {'Frame':<8} {'Motion':<10} {'Boxes':<8} {'Largest Area (px)'}")
print(f"  {'-'*45}")

motion_count  = 0
saved_frame   = None   # we'll save the first frame where motion is detected

for i in range(60):
    ret, frame = cap.read()

    if not ret:
        print(f"\nERROR: Failed to read frame {i+1}. Stopping early.")
        break

    result = detector.detect(frame)

    if result.detected:
        motion_count += 1
        status = "YES  <<<"          # arrow makes it easy to spot in terminal
        if saved_frame is None:      # save the first motion frame as a jpg
            detector.draw_on_frame(frame, result)
            saved_frame = frame.copy()
    else:
        status = "no"

    print(f"  {i+1:<8} {status:<10} {len(result.bounding_boxes):<8} {result.contour_area:.0f}")

# ── Step 4: Save an annotated frame as proof ──────────────
if saved_frame is not None:
    output_path = "test_output.jpg"
    cv2.imwrite(output_path, saved_frame)
    print(f"\n[OK] Saved annotated frame → {output_path}")
    print("     Open this image to see the green bounding boxes.")
else:
    # Save the last frame anyway so you have something to look at
    ret, frame = cap.read()
    if ret:
        cv2.imwrite("test_output.jpg", frame)
    print("\n[NOTE] No motion was detected during the test.")
    print("       Try running again and waving in front of the camera.")

# ── Step 5: Summary ───────────────────────────────────────
print(f"\n{'=' * 55}")
print(f"  RESULT: Motion detected in {motion_count} / 60 frames")

if motion_count > 0:
    print("  STATUS: PASS — motion detection is working correctly")
else:
    print("  STATUS: No motion found — try waving during the test")
    print("  TIP: Lower threshold if you moved but nothing triggered")
    print("       e.g. MotionDetector(threshold=2000)")

print(f"{'=' * 55}\n")

cap.release()