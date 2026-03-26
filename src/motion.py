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

Both of you should understand the full flow, but this split
lets you work in parallel. Coordinate on the interface between
_compute_delta() and _find_motion_regions().

═══════════════════════════════════════════════════════════
THE ALGORITHM (step by step)
═══════════════════════════════════════════════════════════
1. GRAYSCALE: Convert the color frame to grayscale.
   Why? Color info isn't needed for motion and slows things down.
   cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

2. BLUR: Apply Gaussian blur to reduce noise.
   Why? Tiny pixel-level changes (sensor noise) would trigger
   false motion detections without this.
   cv2.GaussianBlur(gray, (21, 21), 0)

3. DELTA: Compute the absolute difference between this frame
   and the previous frame. Pixels that changed will be bright.
   cv2.absdiff(prev_frame, current_frame)

4. THRESHOLD: Convert the delta to pure black & white.
   Pixels with a difference > 25 become white (255), rest become
   black (0). This gives us a clean mask of "what moved."
   cv2.threshold(delta, 25, 255, cv2.THRESH_BINARY)

5. DILATE: Expand the white regions slightly to fill gaps.
   cv2.dilate(thresh, None, iterations=2)

6. CONTOURS: Find the outlines of the white regions.
   cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

7. FILTER: Only keep contours larger than the threshold (5000 pixels).
   Small contours are noise; big ones are real movement.
   cv2.contourArea(contour) > threshold

8. BOUNDING BOXES: Draw rectangles around the remaining contours.
   cv2.boundingRect(contour) → (x, y, width, height)

═══════════════════════════════════════════════════════════
COMMON PITFALLS
═══════════════════════════════════════════════════════════
- First frame: There's no "previous frame" to compare to on the
  very first call. Handle this by returning no motion detected.
- Threshold tuning: 5000 pixels works for a 640x480 feed. If you
  get too many false positives, increase it. Too few detections,
  decrease it.
- blur_size must be ODD (21, not 20). OpenCV will error otherwise.
- findContours modifies the input image in some OpenCV versions —
  pass a copy if needed.
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
        detected: True if any significant motion was found
        bounding_boxes: List of (x, y, width, height) tuples marking
                        where motion was detected in the frame
        contour_area: Area of the largest detected contour (useful
                      for debugging sensitivity)
    """
    detected: bool = False
    bounding_boxes: List[Tuple[int, int, int, int]] = field(default_factory=list)
    contour_area: float = 0.0


class MotionDetector:
    def __init__(self, threshold=5000, blur_size=21, delta_thresh=25):
        """
        Initialize the motion detector.

        Args:
            threshold: Minimum contour area (in pixels) to count as real
                       motion. Smaller = more sensitive, larger = less
                       sensitive. Start with 5000 and adjust.
            blur_size: Size of the Gaussian blur kernel. Must be an ODD
                       number. Larger = more smoothing = fewer false
                       positives, but might miss small movements.
            delta_thresh: Brightness threshold for the frame difference.
                          Pixels that changed by more than this value
                          between frames are considered "moved."

        What to do:
            1. Store all three parameters as instance variables:
               self.threshold = threshold
               self.blur_size = blur_size
               self.delta_thresh = delta_thresh
            2. Initialize self.prev_frame = None
               (We have no previous frame on the first call)
        """
        # YOUR CODE HERE
        pass

    def _preprocess(self, frame):
        """
        ┌─────────────────────────────────────────────────┐
        │  MELANIE: This is your primary method.          │
        └─────────────────────────────────────────────────┘

        Convert a color frame to a blurred grayscale image.
        This prepares the frame for comparison with the previous one.

        Args:
            frame: Raw BGR (color) frame from the camera.
                   Shape: (height, width, 3)

        Returns:
            Blurred grayscale frame. Shape: (height, width)

        What to do:
            1. Convert to grayscale:
               gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

               Why? A color frame has 3 channels (Blue, Green, Red).
               Grayscale has 1 channel. Motion detection only cares
               about brightness changes, not color changes.

            2. Apply Gaussian blur:
               blurred = cv2.GaussianBlur(gray, (self.blur_size, self.blur_size), 0)

               Why? Camera sensors produce noise — tiny random brightness
               changes between frames. Blurring smooths these out so they
               don't trigger false motion alerts. The kernel size (21x21)
               determines how much smoothing. The 0 means OpenCV auto-
               calculates the standard deviation.

            3. Return blurred
        """
        # YOUR CODE HERE (MELANIE)
        pass

    def _compute_delta(self, current_gray):
        """
        ┌─────────────────────────────────────────────────┐
        │  MELANIE: This is your second method.           │
        └─────────────────────────────────────────────────┘

        Compare the current preprocessed frame with the previous one
        to find what changed (i.e., what moved).

        Args:
            current_gray: Preprocessed grayscale frame from _preprocess()

        Returns:
            Binary (black/white) image where white = motion, black = still.
            Returns None if this is the first frame (nothing to compare to).

        What to do:
            1. Handle the first frame:
               if self.prev_frame is None:
                   self.prev_frame = current_gray
                   return None
               (On the first call there's no previous frame to diff against)

            2. Compute absolute difference:
               delta = cv2.absdiff(self.prev_frame, current_gray)

               This subtracts the two frames pixel by pixel. If a pixel
               was bright in frame 1 and dark in frame 2 (or vice versa),
               the delta will be bright. Unchanged pixels → delta is dark.

            3. Threshold to binary:
               _, thresh = cv2.threshold(delta, self.delta_thresh, 255, cv2.THRESH_BINARY)

               Any pixel in delta that's brighter than self.delta_thresh
               (default 25) becomes pure white (255). Everything else
               becomes black (0). This creates a clean mask.

            4. Dilate to fill gaps:
               thresh = cv2.dilate(thresh, None, iterations=2)

               Dilation expands white regions. This fills small gaps
               between nearby motion pixels, making contour detection
               more reliable.

            5. Update the stored previous frame:
               self.prev_frame = current_gray

            6. Return thresh
        """
        # YOUR CODE HERE (MELANIE)
        pass

    def _find_motion_regions(self, thresh_frame):
        """
        ┌─────────────────────────────────────────────────┐
        │  STAN: This is your primary method.             │
        └─────────────────────────────────────────────────┘

        Find contours in the thresholded frame and filter out
        small ones (noise). Return bounding boxes for real motion.

        Args:
            thresh_frame: Binary image from _compute_delta().
                          White pixels = something moved there.

        Returns:
            List of (x, y, w, h) tuples. Each tuple is a bounding
            box around a region where motion was detected.
            Empty list if no significant motion found.

        What to do:
            1. Find contours:
               contours, _ = cv2.findContours(
                   thresh_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
               )

               cv2.RETR_EXTERNAL = only outermost contours (ignore nested)
               cv2.CHAIN_APPROX_SIMPLE = compress contour to fewer points

            2. Filter and collect bounding boxes:
               boxes = []
               max_area = 0
               for contour in contours:
                   area = cv2.contourArea(contour)
                   if area > self.threshold:
                       (x, y, w, h) = cv2.boundingRect(contour)
                       boxes.append((x, y, w, h))
                       max_area = max(max_area, area)

               cv2.contourArea() gives the area in pixels.
               cv2.boundingRect() gives the smallest rectangle
               that encloses the contour.

            3. Return boxes (and you may want to store max_area
               somewhere accessible for the MotionResult)
        """
        # YOUR CODE HERE (STAN)
        pass

    def detect(self, frame):
        """
        ┌─────────────────────────────────────────────────┐
        │  STAN: This method wires the pipeline together. │
        └─────────────────────────────────────────────────┘

        Run the full motion detection pipeline on a single frame.

        Args:
            frame: Raw BGR frame from camera (Silvana's module)

        Returns:
            MotionResult with:
              - detected: True if motion found
              - bounding_boxes: list of (x,y,w,h) rectangles
              - contour_area: largest contour area

        What to do:
            1. Preprocess: gray = self._preprocess(frame)
            2. Compute delta: thresh = self._compute_delta(gray)
            3. Handle first frame:
               if thresh is None:
                   return MotionResult()   # No motion on first frame
            4. Find regions: boxes = self._find_motion_regions(thresh)
            5. Build and return result:
               return MotionResult(
                   detected=len(boxes) > 0,
                   bounding_boxes=boxes,
                   contour_area=<largest area found>
               )
        """
        # YOUR CODE HERE (STAN)
        pass

    def draw_on_frame(self, frame, result):
        """
        ┌─────────────────────────────────────────────────┐
        │  STAN: Draw visual indicators on the frame.     │
        └─────────────────────────────────────────────────┘

        Annotate the frame with motion detection results so
        the user can SEE what's happening.

        Args:
            frame: Frame to draw on (will be modified in place)
            result: MotionResult from detect()

        Returns:
            The annotated frame (same object, modified)

        What to do:
            1. Draw a green rectangle for each bounding box:
               for (x, y, w, h) in result.bounding_boxes:
                   cv2.rectangle(frame, (x, y), (x + w, y + h),
                                 (0, 255, 0), 2)

               Color (0, 255, 0) = green in BGR format.
               Thickness 2 = 2 pixel wide border.

            2. If motion detected, add text label:
               if result.detected:
                   cv2.putText(frame, "MOTION DETECTED",
                               (10, 30),                    # position
                               cv2.FONT_HERSHEY_SIMPLEX,    # font
                               0.8,                          # font scale
                               (0, 255, 0),                  # color (green)
                               2)                            # thickness

            3. Return frame
        """
        # YOUR CODE HERE (STAN)
        pass


# ── FOR STANDALONE TESTING ──────────────────────────────
if __name__ == "__main__":
    """
    Test motion detection independently:
        python src/motion.py

    What you should see:
        - Your webcam feed opens in a window
        - When you move, green boxes appear around the motion
        - "MOTION DETECTED" text appears at the top
        - When still, the boxes and text disappear
        - Press 'q' to quit

    What to implement:
        1. cap = cv2.VideoCapture(0)
        2. detector = MotionDetector(threshold=5000)
        3. while True:
            a. ret, frame = cap.read()
            b. if not ret: break
            c. result = detector.detect(frame)
            d. detector.draw_on_frame(frame, result)
            e. Print result.detected and result.contour_area for debugging
            f. cv2.imshow("Motion Detection Test", frame)
            g. if cv2.waitKey(1) & 0xFF == ord('q'): break
        4. cap.release()
        5. cv2.destroyAllWindows()

    Debugging tips:
        - If EVERYTHING triggers motion: increase threshold
        - If NOTHING triggers motion: decrease threshold or check
          that _preprocess and _compute_delta are correct
        - Print the number of contours found to see if filtering
          is working correctly
    """
    # YOUR CODE HERE (MELANIE & STAN)
    pass