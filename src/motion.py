"""
Module: motion.py
Owners: Melanie & Stan
Branch: melanie (preprocessing) / stan (contours & drawing)

═══════════════════════════════════════════════════════════
PURPOSE
═══════════════════════════════════════════════════════════
This module detects MOVEMENT in the camera feed. It works by
comparing consecutive frames — if something changed between
frame N and frame N+1, something moved.

This is called "frame differencing" and it's one of the simplest
and most effective motion detection techniques.

═══════════════════════════════════════════════════════════
HOW IT FITS IN THE PIPELINE
═══════════════════════════════════════════════════════════
In main.py, your module runs right after the camera:

    ret, frame = camera.read_frame()         # Silvana's module
    motion_result = motion.detect(frame)      # YOUR MODULE
    motion.draw_on_frame(frame, motion_result)
    alerts.check_motion(motion_result)        # Ritchie's module

You receive a raw frame and produce a MotionResult that tells
Ritchie's alert system whether motion was detected and where.

═══════════════════════════════════════════════════════════
TASK SPLIT
═══════════════════════════════════════════════════════════
MELANIE handles the "math" side:
  - _preprocess(): Convert frame to grayscale, apply blur
  - _compute_delta(): Compare frames, threshold the difference

STAN handles the "visual" side:
  - _find_motion_regions(): Find contours, filter by size
  - draw_on_frame(): Draw boxes around moving things
  - detect(): Wire the full pipeline together

═══════════════════════════════════════════════════════════
THE ALGORITHM (step by step)
═══════════════════════════════════════════════════════════
1. GRAYSCALE: Convert the color frame to grayscale.
   Why? Color info isn't needed for motion and slows things down.

2. BLUR: Apply Gaussian blur to reduce noise.
   Why? Tiny pixel-level changes (sensor noise) would trigger
   false motion detections without this.

3. DELTA: Compute the absolute difference between this frame
   and the previous frame. Pixels that changed will be bright.

4. THRESHOLD: Convert the delta to pure black & white.
   Pixels with a difference > 25 become white (255), rest become
   black (0). This gives us a clean mask of "what moved."

5. DILATE: Expand the white regions slightly to fill gaps.

6. CONTOURS: Find the outlines of the white regions.

7. FILTER: Only keep contours larger than the threshold (5000 pixels).
   Small contours are noise; big ones are real movement.

8. BOUNDING BOXES: Draw rectangles around the remaining contours.
   cv2.boundingRect(contour) → (x, y, width, height)
"""

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class MotionResult:
    """
    Result of motion detection on a single frame.

    Attributes:
        detected      : True if any significant motion was found
        bounding_boxes: List of (x, y, width, height) tuples marking
                        where motion was detected in the frame
        contour_area  : Area of the largest detected contour (useful
                        for debugging / tuning the threshold)
    """
    detected: bool = False
    bounding_boxes: List[Tuple[int, int, int, int]] = field(default_factory=list)
    contour_area: float = 0.0


class MotionDetector:

    def __init__(self, threshold=5000, blur_size=21, delta_thresh=25):
        """
        Initialize the motion detector.

        Args:
            threshold   : Minimum contour area (pixels) to count as real motion.
                          Start with 5000. Increase if too many false positives,
                          decrease if real motion is being missed.
            blur_size   : Gaussian blur kernel size. MUST be an odd number.
                          Larger = smoother = fewer false positives.
            delta_thresh: How different a pixel must be (0-255) between frames
                          to count as "moved". Default 25 works well indoors.
        """
        self.threshold   = threshold
        self.blur_size   = blur_size
        self.delta_thresh = delta_thresh
        self.prev_frame  = None   # No previous frame exists on first call


    #preprocessing + math
    def _preprocess(self, frame):
        """
        Convert a raw color frame into a blurred grayscale image
        ready for frame-differencing.

        Steps:
            1. Convert BGR → grayscale   (cv2.cvtColor)
            2. Apply Gaussian blur        (cv2.GaussianBlur)
            3. Return the blurred image

        Why grayscale?
            A color frame has 3 channels (B, G, R). Motion detection only
            needs brightness changes, not color. Grayscale = 1 channel =
            3× less data to process.

        Why blur?
            Camera sensors produce tiny random brightness differences between
            frames even when nothing moves (noise). Blurring averages nearby
            pixels together so those tiny differences disappear and don't
            trigger false motion alerts.

        Args:
            frame: Raw BGR frame from Silvana's camera module.
                   Shape: (height, width, 3)

        Returns:
            blurred: Grayscale blurred frame. Shape: (height, width)
        """
        # Step 1 — Convert color frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Step 2 — Blur to kill sensor noise
        # (blur_size, blur_size) is the kernel — must be odd numbers
        # The final 0 tells OpenCV to auto-calculate standard deviation
        blurred = cv2.GaussianBlur(gray, (self.blur_size, self.blur_size), 0)

        return blurred


    def _compute_delta(self, current_gray):
        """
        Compare the current frame with the previous frame to find
        what changed between them — i.e., what moved.

        Steps:
            1. First-frame guard: if no previous frame, store this one
               and return None (nothing to compare against yet)
            2. cv2.absdiff — subtract frames pixel by pixel
            3. cv2.threshold — convert diff to pure black/white mask
            4. cv2.dilate — expand white blobs to fill small gaps
            5. Update self.prev_frame for the next call
            6. Return the binary threshold image

        Why absdiff?
            Subtracting frame A from frame B gives a bright pixel wherever
            the brightness changed. Unchanged pixels → near 0 (black).
            Moved pixels → bright value.

        Why threshold?
            The diff image has many shades of grey. Thresholding snaps
            everything to pure black (0) or pure white (255) so we get
            a clean mask we can find contours in.

        Why dilate?
            After thresholding, a moving object might appear as lots of
            small disconnected white dots. Dilation expands each white
            pixel outward, merging nearby dots into solid blobs so contour
            detection works reliably.

        Args:
            current_gray: Blurred grayscale frame from _preprocess().
                          Shape: (height, width)

        Returns:
            thresh: Binary (black/white) image — white where motion exists.
            None  : On the very first call (no previous frame to compare).
        """
        # Step 1 — First-frame guard
        if self.prev_frame is None:
            self.prev_frame = current_gray
            return None

        # Step 2 — Absolute difference: bright where pixels changed
        delta = cv2.absdiff(self.prev_frame, current_gray)

        # Step 3 — Threshold: anything brighter than delta_thresh → white
        _, thresh = cv2.threshold(
            delta,
            self.delta_thresh,  # pixels changed more than this → white
            255,                # white value
            cv2.THRESH_BINARY
        )

        # Step 4 — Dilate: expand white blobs to fill gaps between them
        thresh = cv2.dilate(thresh, None, iterations=2)

        # Step 5 — Save this frame so the NEXT call can compare against it
        self.prev_frame = current_gray

        # Step 6 — Return the clean binary motion mask
        return thresh

    #contours + drawing + wiring
    def _find_motion_regions(self, thresh_frame):
        """
        Find the outlines of the white blobs in the threshold image,
        filter out small ones (noise), and return bounding rectangles
        around the real motion regions.

        Steps:
            1. cv2.findContours — get outlines of all white blobs
            2. Loop through contours
            3. Skip any contour whose area is below self.threshold
            4. For the rest, get the bounding rectangle and store it
            5. Track the largest area seen
            6. Return (boxes list, max_area)

        Why filter by area?
            Even after blurring and thresholding, tiny white specks remain
            from lighting flicker, compression artifacts, etc. Requiring a
            minimum area of 5000 pixels (~70×70 square) eliminates these
            while keeping real human-sized movement.

        Args:
            thresh_frame: Binary image from _compute_delta().
                          White pixels = motion detected there.

        Returns:
            boxes   : List of (x, y, w, h) tuples — one per motion region
            max_area: Float — area of the largest contour found (0.0 if none)
        """
        # Step 1 — Find contours (outlines of white regions)
        # .copy() protects against older OpenCV versions that modify input
        # RETR_EXTERNAL = outermost contours only (ignore holes inside blobs)
        # CHAIN_APPROX_SIMPLE = store only corner points, saves memory
        contours, _ = cv2.findContours(
            thresh_frame.copy(),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        boxes    = []
        max_area = 0.0

        # Step 2-5 — Filter, collect bounding boxes, track largest area
        for contour in contours:
            area = cv2.contourArea(contour)

            # Step 3 — Skip tiny contours (they're just noise)
            if area <= self.threshold:
                continue

            # Step 4 — Get the rectangle that encloses this contour
            x, y, w, h = cv2.boundingRect(contour)
            boxes.append((x, y, w, h))

            # Step 5 — Keep track of the biggest region seen
            if area > max_area:
                max_area = area

        return boxes, max_area


    def detect(self, frame):
        """
        Run the complete motion detection pipeline on one frame.
        This is the method that main.py (and everyone else) calls.

        Pipeline:
            frame → _preprocess → _compute_delta → _find_motion_regions
            → MotionResult

        Args:
            frame: Raw BGR frame from Silvana's camera.read_frame()

        Returns:
            MotionResult with:
                detected       : True if motion was found
                bounding_boxes : list of (x, y, w, h) rectangles
                contour_area   : area of the largest moving region
        """
        # Step 1 — Grayscale + blur  
        gray = self._preprocess(frame)

        # Step 2 — Frame differencing → binary mask  (Melanie's delta)
        thresh = self._compute_delta(gray)

        # Step 3 — First frame: nothing to compare yet, return empty result
        if thresh is None:
            return MotionResult()

        # Step 4 — Find contours and bounding boxes  (Stan's regions)
        boxes, max_area = self._find_motion_regions(thresh)

        # Step 5 — Package everything into a MotionResult for other modules
        return MotionResult(
            detected=len(boxes) > 0,
            bounding_boxes=boxes,
            contour_area=max_area
        )


    def draw_on_frame(self, frame, result):
        """
        Draw visual indicators onto the frame so the user can see
        exactly where motion was detected.

        Called in main.py after detect(), before cv2.imshow().

        What gets drawn:
            - A green rectangle around each moving region
            - "MOTION DETECTED" text at the top if motion is present

        Args:
            frame : The BGR frame to draw on (modified in place)
            result: MotionResult returned by detect()

        Returns:
            frame: The same frame, now annotated
        """
        # Step 1 — Green rectangle around each detected motion region
        for (x, y, w, h) in result.bounding_boxes:
            cv2.rectangle(
                frame,
                (x, y),          # top-left corner of box
                (x + w, y + h),  # bottom-right corner of box
                (0, 255, 0),     # green in BGR
                2                # 2-pixel border thickness
            )

        # Step 2 — "MOTION DETECTED" label when motion is active
        if result.detected:
            cv2.putText(
                frame,
                "MOTION DETECTED",   # text to display
                (10, 30),            # position: near top-left
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,                 # font scale
                (0, 255, 0),         # green
                2                    # thickness
            )

        return frame


if __name__ == "__main__":
    print("EyeGuard – Motion Detection standalone test")
    print("Move in front of the camera.  Press 'q' to quit.\n")

    # Open the default webcam (index 0)
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("ERROR: Could not open camera. Is it connected and not in use?")
        exit(1)

    # Create the detector — adjust threshold here if needed:
    #   higher → less sensitive (fewer false positives)
    #   lower  → more sensitive (catches smaller movements)
    detector = MotionDetector(threshold=5000)

    while True:
        ret, frame = cap.read()

        if not ret:
            print("ERROR: Failed to read frame. Camera may have disconnected.")
            break

        # Run the full pipeline
        result = detector.detect(frame)

        # Draw boxes and label onto the frame
        detector.draw_on_frame(frame, result)

        # Debug line in terminal — helpful for tuning the threshold
        print(
            f"  motion={result.detected}"
            f"  |  boxes={len(result.bounding_boxes)}"
            f"  |  largest_area={result.contour_area:.0f} px",
            end="\r"
        )

        # Show the annotated live feed
        cv2.imshow("EyeGuard – Motion Detection", frame)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("\nCamera released. Bye!")