"""
Module: face_eye.py
Owner: Deno
Branch: deno

═══════════════════════════════════════════════════════════
PURPOSE
═══════════════════════════════════════════════════════════
This module finds a FACE in the camera frame and extracts
the exact pixel coordinates of the EYE landmarks. These
landmarks are then passed to Timo's module for drowsiness
calculation.

You'll use two tools from the dlib library:
  1. A face detector (finds where the face is in the image)
  2. A shape predictor (finds 68 specific facial landmarks
     within the detected face region)

═══════════════════════════════════════════════════════════
HOW IT FITS IN THE PIPELINE
═══════════════════════════════════════════════════════════
In main.py, your module runs after motion detection:

    motion_result = motion.detect(frame)         # Melanie & Stan
    face_result = face_eye.detect(frame)          # YOUR MODULE
    face_eye.draw_on_frame(frame, face_result)
    if face_result.face_detected:
        ear_result = ear_calc.calculate(          # Timo's module
            face_result.left_eye,                 # <-- you provide this
            face_result.right_eye                 # <-- and this
        )

Your FaceEyeResult feeds directly into Timo's EAR calculator.
If you don't detect a face, Timo's module is skipped that frame.

═══════════════════════════════════════════════════════════
BACKGROUND: THE 68-POINT FACIAL LANDMARK MODEL
═══════════════════════════════════════════════════════════
dlib's shape predictor identifies 68 specific points on a face:

  Points  0-16:  Jawline
  Points 17-21:  Right eyebrow
  Points 22-26:  Left eyebrow
  Points 27-35:  Nose
  Points 36-41:  RIGHT EYE  ← you need these
  Points 42-47:  LEFT EYE   ← and these
  Points 48-67:  Mouth

Each eye has 6 landmark points arranged like this:

         p1 (upper-outer)    p2 (upper-inner)
  p0 ●────────●──────────────●──────────● p3
  (outer)     │              │          (inner)
              ●──────────────●
         p5 (lower-outer)    p4 (lower-inner)

These 6 points per eye are what Timo needs to calculate the
Eye Aspect Ratio (EAR).

═══════════════════════════════════════════════════════════
SETUP REQUIRED (before you can code)
═══════════════════════════════════════════════════════════
1. Install dlib (can be tricky — needs cmake):
   pip install cmake
   pip install dlib

   If dlib fails to install, try:
   sudo apt install build-essential cmake   (Linux)
   brew install cmake                       (Mac)
   Then retry pip install dlib

2. Download the shape predictor model (~100MB):
   wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
   bunzip2 shape_predictor_68_face_landmarks.dat.bz2

   Place the resulting .dat file in the project root folder.

3. Install imutils (helper library for dlib):
   pip install imutils

═══════════════════════════════════════════════════════════
STEP-BY-STEP IMPLEMENTATION GUIDE
═══════════════════════════════════════════════════════════
Step 1: Get __init__ working
  - Load the face detector and shape predictor
  - Test that the model file loads without errors

Step 2: Implement _detect_face()
  - Run face detection on a grayscale frame
  - Print how many faces were found (for debugging)

Step 3: Implement _extract_landmarks()
  - Get the 68 landmarks for a detected face
  - Print the first few landmarks to verify

Step 4: Implement _extract_eyes()
  - Slice out the eye points from the full 68 landmarks
  - Print left_eye and right_eye to verify 6 points each

Step 5: Wire it all together in detect()
  - Handle the case where no face is found

Step 6: Implement draw_on_frame()
  - Draw the eye points so you can visually verify

═══════════════════════════════════════════════════════════
COMMON PITFALLS
═══════════════════════════════════════════════════════════
- The shape predictor file is ~100MB. Don't commit it to git!
  (It's already in .gitignore)
- dlib's face detector works on GRAYSCALE images, not color.
  You must convert the frame before passing it to the detector.
- If no face is detected, return FaceEyeResult(face_detected=False).
  Don't crash or return None.
- Face detection is the SLOWEST part of the pipeline on a Pi 3.
  Consider running it every 2-3 frames instead of every frame
  (optimization for later).
- dlib uses RGB, but OpenCV uses BGR. For the face detector
  grayscale is fine, so this usually doesn't matter, but be
  aware of it if you see color-related issues.
"""

import cv2
import dlib
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from imutils import face_utils


# ═══════════════════════════════════════════════════════════
# EYE LANDMARK INDICES
# These map to specific points in dlib's 68-point model.
# Don't change these numbers — they're defined by the model.
# ═══════════════════════════════════════════════════════════
RIGHT_EYE_START, RIGHT_EYE_END = 36, 42   # Points 36,37,38,39,40,41
LEFT_EYE_START, LEFT_EYE_END = 42, 48     # Points 42,43,44,45,46,47


@dataclass
class FaceEyeResult:
    """
    Result of face and eye detection on a single frame.

    Attributes:
        face_detected: True if at least one face was found
        left_eye: List of 6 (x, y) tuples for the left eye landmarks.
                  These are pixel coordinates in the original frame.
                  Order: [outer_corner, upper_outer, upper_inner,
                          inner_corner, lower_inner, lower_outer]
        right_eye: Same as left_eye but for the right eye
        all_landmarks: Full 68-point numpy array (for debugging/drawing)
    """
    face_detected: bool = False
    left_eye: List[Tuple[int, int]] = field(default_factory=list)
    right_eye: List[Tuple[int, int]] = field(default_factory=list)
    all_landmarks: Optional[np.ndarray] = None


class FaceEyeDetector:
    def __init__(self, predictor_path="shape_predictor_68_face_landmarks.dat"):
        """
        Initialize the face and eye detector.

        Args:
            predictor_path: Path to the dlib shape predictor .dat file.
                            This model file must be downloaded separately
                            (see setup instructions at the top of this file).

        What to do:
            1. Create the face detector:
               self.detector = dlib.get_frontal_face_detector()

               This is a HOG (Histogram of Oriented Gradients) based
               detector. It's fast and works well for frontal faces.

            2. Load the shape predictor:
               try:
                   self.predictor = dlib.shape_predictor(predictor_path)
               except RuntimeError:
                   raise FileNotFoundError(
                       f"Could not find shape predictor model at: {predictor_path}\\n"
                       f"Download it with:\\n"
                       f"  wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2\\n"
                       f"  bunzip2 shape_predictor_68_face_landmarks.dat.bz2"
                   )

            3. Print a confirmation message

        Raises:
            FileNotFoundError: If the .dat model file is not found
        """
        # YOUR CODE HERE
        pass

    def _detect_face(self, gray_frame):
        """
        Detect faces in a grayscale frame.

        Args:
            gray_frame: Grayscale image (single channel, dtype uint8)

        Returns:
            List of dlib.rectangle objects. Each rectangle defines the
            bounding box of a detected face: (left, top, right, bottom).
            Returns an empty list if no faces found.

        What to do:
            1. Run the detector:
               faces = self.detector(gray_frame, 0)

               The second argument (0) is the number of times to
               upsample the image before detection. 0 = no upsampling
               (faster but might miss small/distant faces). Use 1 if
               you're having trouble detecting faces.

            2. Return faces

        Note: This returns ALL detected faces. In detect(), we'll
        use only the first one (closest/largest).
        """
        # YOUR CODE HERE
        pass

    def _extract_landmarks(self, gray_frame, face_rect):
        """
        Extract 68 facial landmarks from a detected face.

        Args:
            gray_frame: Grayscale image
            face_rect: A single dlib.rectangle from _detect_face()
                       This tells the predictor WHERE the face is,
                       so it can find the landmarks within that region.

        Returns:
            numpy array of shape (68, 2) where each row is an (x, y)
            coordinate of a landmark point.

        What to do:
            1. Run the shape predictor:
               shape = self.predictor(gray_frame, face_rect)

               This returns a dlib.full_object_detection with 68 parts.

            2. Convert to numpy array:
               landmarks = face_utils.shape_to_np(shape)

               face_utils.shape_to_np() is a helper from imutils that
               converts dlib's format to a simple numpy array.

            3. Return landmarks
        """
        # YOUR CODE HERE
        pass

    def _extract_eyes(self, landmarks):
        """
        Extract left and right eye coordinates from the full landmarks.

        Args:
            landmarks: numpy array of shape (68, 2) from _extract_landmarks()

        Returns:
            tuple: (left_eye_points, right_eye_points)
            Each is a list of 6 (x, y) tuples.

        What to do:
            1. Slice out the left eye points:
               left_eye = landmarks[LEFT_EYE_START:LEFT_EYE_END]
               This gets points 42,43,44,45,46,47

            2. Slice out the right eye points:
               right_eye = landmarks[RIGHT_EYE_START:RIGHT_EYE_END]
               This gets points 36,37,38,39,40,41

            3. Convert each from numpy arrays to lists of tuples:
               left_eye = [(int(x), int(y)) for x, y in left_eye]
               right_eye = [(int(x), int(y)) for x, y in right_eye]

               Why convert? Timo's module expects plain Python tuples,
               not numpy arrays. The int() ensures pixel coordinates
               are integers (numpy might store them as floats).

            4. Return (left_eye, right_eye)
        """
        # YOUR CODE HERE
        pass

    def detect(self, frame):
        """
        Run the full face and eye detection pipeline.

        Args:
            frame: Raw BGR frame from camera (Silvana's module)

        Returns:
            FaceEyeResult with eye coordinates (or face_detected=False)

        What to do:
            1. Convert to grayscale:
               gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            2. Detect faces:
               faces = self._detect_face(gray)

            3. If no faces found:
               if len(faces) == 0:
                   return FaceEyeResult(face_detected=False)

            4. Use the first (largest/closest) face:
               face_rect = faces[0]

            5. Extract landmarks:
               landmarks = self._extract_landmarks(gray, face_rect)

            6. Extract eyes:
               left_eye, right_eye = self._extract_eyes(landmarks)

            7. Return the result:
               return FaceEyeResult(
                   face_detected=True,
                   left_eye=left_eye,
                   right_eye=right_eye,
                   all_landmarks=landmarks
               )
        """
        # YOUR CODE HERE
        pass

    def draw_on_frame(self, frame, result):
        """
        Draw eye landmarks and visual indicators on the frame.

        Args:
            frame: Frame to annotate (modified in place)
            result: FaceEyeResult from detect()

        Returns:
            Annotated frame

        What to do:
            1. If no face detected, show a message:
               if not result.face_detected:
                   cv2.putText(frame, "No face detected", (10, 70),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                               (128, 128, 128), 2)
                   return frame

            2. Draw circles on each eye landmark:
               for (x, y) in result.left_eye:
                   cv2.circle(frame, (x, y), 2, (0, 255, 255), -1)
               for (x, y) in result.right_eye:
                   cv2.circle(frame, (x, y), 2, (0, 255, 255), -1)

               Color (0, 255, 255) = yellow/cyan in BGR
               Radius 2, thickness -1 = filled circle

            3. (Optional but recommended) Draw convex hull around eyes:
               left_hull = cv2.convexHull(np.array(result.left_eye))
               cv2.drawContours(frame, [left_hull], -1, (0, 255, 255), 1)
               # Same for right eye

               This draws a smooth outline around each eye, which
               looks much better than just dots.

            4. Return frame
        """
        # YOUR CODE HERE
        pass


# ── FOR STANDALONE TESTING ──────────────────────────────
if __name__ == "__main__":
    """
    Test face/eye detection independently:
        python src/face_eye.py

    What you should see:
        - Your webcam opens in a window
        - Yellow dots appear on your eye landmarks when your face
          is detected
        - "No face detected" appears when you cover/turn your face
        - Press 'q' to quit

    Prerequisites:
        - shape_predictor_68_face_landmarks.dat must be in the
          project root (or adjust the path below)

    What to implement:
        1. cap = cv2.VideoCapture(0)
        2. detector = FaceEyeDetector("shape_predictor_68_face_landmarks.dat")
        3. while True:
            a. ret, frame = cap.read()
            b. if not ret: break
            c. result = detector.detect(frame)
            d. detector.draw_on_frame(frame, result)
            e. If face detected, print the eye coordinates for debugging:
               print(f"Left eye: {result.left_eye}")
               print(f"Right eye: {result.right_eye}")
            f. cv2.imshow("Face & Eye Detection Test - Deno", frame)
            g. if cv2.waitKey(1) & 0xFF == ord('q'): break
        4. cap.release()
        5. cv2.destroyAllWindows()

    Debugging tips:
        - If face detection is slow, that's normal on lower-end
          hardware. Consider detecting every 2nd or 3rd frame.
        - If landmarks look wrong, make sure you're using the correct
          .dat file (68 landmarks, not 5)
        - Make sure you're converting to grayscale BEFORE detection
    """
    # YOUR CODE HERE
    pass

"""
Module: face_eye.py
Owner: Deno
Branch: deno

═══════════════════════════════════════════════════════════
PURPOSE
═══════════════════════════════════════════════════════════
This module finds a FACE in the camera frame and extracts
the exact pixel coordinates of the EYE landmarks. These
landmarks are then passed to Timo's module for drowsiness
calculation.

You'll use two tools from the dlib library:
  1. A face detector (finds where the face is in the image)
  2. A shape predictor (finds 68 specific facial landmarks
     within the detected face region)

═══════════════════════════════════════════════════════════
HOW IT FITS IN THE PIPELINE
═══════════════════════════════════════════════════════════
In main.py, your module runs after motion detection:

    motion_result = motion.detect(frame)         # Melanie & Stan
    face_result = face_eye.detect(frame)          # YOUR MODULE
    face_eye.draw_on_frame(frame, face_result)
    if face_result.face_detected:
        ear_result = ear_calc.calculate(          # Timo's module
            face_result.left_eye,                 # <-- you provide this
            face_result.right_eye                 # <-- and this
        )

Your FaceEyeResult feeds directly into Timo's EAR calculator.
If you don't detect a face, Timo's module is skipped that frame.

═══════════════════════════════════════════════════════════
BACKGROUND: THE 68-POINT FACIAL LANDMARK MODEL
═══════════════════════════════════════════════════════════
dlib's shape predictor identifies 68 specific points on a face:

  Points  0-16:  Jawline
  Points 17-21:  Right eyebrow
  Points 22-26:  Left eyebrow
  Points 27-35:  Nose
  Points 36-41:  RIGHT EYE  ← you need these
  Points 42-47:  LEFT EYE   ← and these
  Points 48-67:  Mouth

Each eye has 6 landmark points arranged like this:

         p1 (upper-outer)    p2 (upper-inner)
  p0 ●────────●──────────────●──────────● p3
  (outer)     │              │          (inner)
              ●──────────────●
         p5 (lower-outer)    p4 (lower-inner)

These 6 points per eye are what Timo needs to calculate the
Eye Aspect Ratio (EAR).

═══════════════════════════════════════════════════════════
SETUP REQUIRED (before you can code)
═══════════════════════════════════════════════════════════
1. Install dlib (can be tricky — needs cmake):
   pip install cmake
   pip install dlib

   If dlib fails to install, try:
   sudo apt install build-essential cmake   (Linux)
   brew install cmake                       (Mac)
   Then retry pip install dlib

2. Download the shape predictor model (~100MB):
   wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
   bunzip2 shape_predictor_68_face_landmarks.dat.bz2

   Place the resulting .dat file in the project root folder.

3. Install imutils (helper library for dlib):
   pip install imutils

═══════════════════════════════════════════════════════════
STEP-BY-STEP IMPLEMENTATION GUIDE
═══════════════════════════════════════════════════════════
Step 1: Get __init__ working
  - Load the face detector and shape predictor
  - Test that the model file loads without errors

Step 2: Implement _detect_face()
  - Run face detection on a grayscale frame
  - Print how many faces were found (for debugging)

Step 3: Implement _extract_landmarks()
  - Get the 68 landmarks for a detected face
  - Print the first few landmarks to verify

Step 4: Implement _extract_eyes()
  - Slice out the eye points from the full 68 landmarks
  - Print left_eye and right_eye to verify 6 points each

Step 5: Wire it all together in detect()
  - Handle the case where no face is found

Step 6: Implement draw_on_frame()
  - Draw the eye points so you can visually verify

═══════════════════════════════════════════════════════════
COMMON PITFALLS
═══════════════════════════════════════════════════════════
- The shape predictor file is ~100MB. Don't commit it to git!
  (It's already in .gitignore)
- dlib's face detector works on GRAYSCALE images, not color.
  You must convert the frame before passing it to the detector.
- If no face is detected, return FaceEyeResult(face_detected=False).
  Don't crash or return None.
- Face detection is the SLOWEST part of the pipeline on a Pi 3.
  Consider running it every 2-3 frames instead of every frame
  (optimization for later).
- dlib uses RGB, but OpenCV uses BGR. For the face detector
  grayscale is fine, so this usually doesn't matter, but be
  aware of it if you see color-related issues.
"""

import cv2
import dlib
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from imutils import face_utils


# ═══════════════════════════════════════════════════════════
# EYE LANDMARK INDICES
# These map to specific points in dlib's 68-point model.
# Don't change these numbers — they're defined by the model.
# ═══════════════════════════════════════════════════════════
RIGHT_EYE_START, RIGHT_EYE_END = 36, 42   # Points 36,37,38,39,40,41
LEFT_EYE_START, LEFT_EYE_END = 42, 48     # Points 42,43,44,45,46,47


@dataclass
class FaceEyeResult:
    """
    Result of face and eye detection on a single frame.

    Attributes:
        face_detected: True if at least one face was found
        left_eye: List of 6 (x, y) tuples for the left eye landmarks.
                  These are pixel coordinates in the original frame.
                  Order: [outer_corner, upper_outer, upper_inner,
                          inner_corner, lower_inner, lower_outer]
        right_eye: Same as left_eye but for the right eye
        all_landmarks: Full 68-point numpy array (for debugging/drawing)
    """
    face_detected: bool = False
    left_eye: List[Tuple[int, int]] = field(default_factory=list)
    right_eye: List[Tuple[int, int]] = field(default_factory=list)
    all_landmarks: Optional[np.ndarray] = None


class FaceEyeDetector:
    def __init__(self, predictor_path="shape_predictor_68_face_landmarks.dat"):
        """
        Initialize the face and eye detector.

        Args:
            predictor_path: Path to the dlib shape predictor .dat file.
                            This model file must be downloaded separately
                            (see setup instructions at the top of this file).

        What to do:
            1. Create the face detector:
               self.detector = dlib.get_frontal_face_detector()

               This is a HOG (Histogram of Oriented Gradients) based
               detector. It's fast and works well for frontal faces.

            2. Load the shape predictor:
               try:
                   self.predictor = dlib.shape_predictor(predictor_path)
               except RuntimeError:
                   raise FileNotFoundError(
                       f"Could not find shape predictor model at: {predictor_path}\\n"
                       f"Download it with:\\n"
                       f"  wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2\\n"
                       f"  bunzip2 shape_predictor_68_face_landmarks.dat.bz2"
                   )

            3. Print a confirmation message

        Raises:
            FileNotFoundError: If the .dat model file is not found
        """
        self.detector = dlib.get_frontal_face_detector()

        try:
            self.predictor = dlib.shape_predictor(predictor_path)
        except RuntimeError:
            raise FileNotFoundError(
                f"Could not find shape predictor model at: {predictor_path}\n"
                f"Download it with:\n"
                f"  wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2\n"
                f"  bunzip2 shape_predictor_68_face_landmarks.dat.bz2"
            )

        print("[FaceEye] Detector and shape predictor loaded successfully.")

    def _detect_face(self, gray_frame):
        """
        Detect faces in a grayscale frame.

        Args:
            gray_frame: Grayscale image (single channel, dtype uint8)

        Returns:
            List of dlib.rectangle objects. Each rectangle defines the
            bounding box of a detected face: (left, top, right, bottom).
            Returns an empty list if no faces found.

        What to do:
            1. Run the detector:
               faces = self.detector(gray_frame, 0)

               The second argument (0) is the number of times to
               upsample the image before detection. 0 = no upsampling
               (faster but might miss small/distant faces). Use 1 if
               you're having trouble detecting faces.

            2. Return faces

        Note: This returns ALL detected faces. In detect(), we'll
        use only the first one (closest/largest).
        """
        faces = self.detector(gray_frame, 0)

        return faces

    def _extract_landmarks(self, gray_frame, face_rect):
        """
        Extract 68 facial landmarks from a detected face.

        Args:
            gray_frame: Grayscale image
            face_rect: A single dlib.rectangle from _detect_face()
                       This tells the predictor WHERE the face is,
                       so it can find the landmarks within that region.

        Returns:
            numpy array of shape (68, 2) where each row is an (x, y)
            coordinate of a landmark point.

        What to do:
            1. Run the shape predictor:
               shape = self.predictor(gray_frame, face_rect)

               This returns a dlib.full_object_detection with 68 parts.

            2. Convert to numpy array:
               landmarks = face_utils.shape_to_np(shape)

               face_utils.shape_to_np() is a helper from imutils that
               converts dlib's format to a simple numpy array.

            3. Return landmarks
        """
        shape = self.predictor(gray_frame, face_rect)

        landmarks = face_utils.shape_to_np(shape)

        return landmarks

    def _extract_eyes(self, landmarks):
        """
        Extract left and right eye coordinates from the full landmarks.

        Args:
            landmarks: numpy array of shape (68, 2) from _extract_landmarks()

        Returns:
            tuple: (left_eye_points, right_eye_points)
            Each is a list of 6 (x, y) tuples.

        What to do:
            1. Slice out the left eye points:
               left_eye = landmarks[LEFT_EYE_START:LEFT_EYE_END]
               This gets points 42,43,44,45,46,47

            2. Slice out the right eye points:
               right_eye = landmarks[RIGHT_EYE_START:RIGHT_EYE_END]
               This gets points 36,37,38,39,40,41

            3. Convert each from numpy arrays to lists of tuples:
               left_eye = [(int(x), int(y)) for x, y in left_eye]
               right_eye = [(int(x), int(y)) for x, y in right_eye]

               Why convert? Timo's module expects plain Python tuples,
               not numpy arrays. The int() ensures pixel coordinates
               are integers (numpy might store them as floats).

            4. Return (left_eye, right_eye)
        """
        left_eye = landmarks[LEFT_EYE_START:LEFT_EYE_END]
        right_eye = landmarks[RIGHT_EYE_START:RIGHT_EYE_END]

        left_eye = [(int(x), int(y)) for x, y in left_eye]
        right_eye = [(int(x), int(y)) for x, y in right_eye]

        return (left_eye, right_eye)

    def detect(self, frame):
        """
        Run the full face and eye detection pipeline.

        Args:
            frame: Raw BGR frame from camera (Silvana's module)

        Returns:
            FaceEyeResult with eye coordinates (or face_detected=False)

        What to do:
            1. Convert to grayscale:
               gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            2. Detect faces:
               faces = self._detect_face(gray)

            3. If no faces found:
               if len(faces) == 0:
                   return FaceEyeResult(face_detected=False)

            4. Use the first (largest/closest) face:
               face_rect = faces[0]

            5. Extract landmarks:
               landmarks = self._extract_landmarks(gray, face_rect)

            6. Extract eyes:
               left_eye, right_eye = self._extract_eyes(landmarks)

            7. Return the result:
               return FaceEyeResult(
                   face_detected=True,
                   left_eye=left_eye,
                   right_eye=right_eye,
                   all_landmarks=landmarks
               )
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = self._detect_face(gray)

        if len(faces) == 0:
            return FaceEyeResult(face_detected=False)

        face_rect = faces[0]

        landmarks = self._extract_landmarks(gray, face_rect)

        left_eye, right_eye = self._extract_eyes(landmarks)

        return FaceEyeResult(
            face_detected=True,
            left_eye=left_eye,
            right_eye=right_eye,
            all_landmarks=landmarks
        )

    def draw_on_frame(self, frame, result):
        """
        Draw eye landmarks and visual indicators on the frame.

        Args:
            frame: Frame to annotate (modified in place)
            result: FaceEyeResult from detect()

        Returns:
            Annotated frame

        What to do:
            1. If no face detected, show a message:
               if not result.face_detected:
                   cv2.putText(frame, "No face detected", (10, 70),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                               (128, 128, 128), 2)
                   return frame

            2. Draw circles on each eye landmark:
               for (x, y) in result.left_eye:
                   cv2.circle(frame, (x, y), 2, (0, 255, 255), -1)
               for (x, y) in result.right_eye:
                   cv2.circle(frame, (x, y), 2, (0, 255, 255), -1)

               Color (0, 255, 255) = yellow/cyan in BGR
               Radius 2, thickness -1 = filled circle

            3. (Optional but recommended) Draw convex hull around eyes:
               left_hull = cv2.convexHull(np.array(result.left_eye))
               cv2.drawContours(frame, [left_hull], -1, (0, 255, 255), 1)
               # Same for right eye

               This draws a smooth outline around each eye, which
               looks much better than just dots.

            4. Return frame
        """
        if not result.face_detected:
            cv2.putText(frame, "No face detected", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (128, 128, 128), 2)
            return frame

        for (x, y) in result.left_eye:
            cv2.circle(frame, (x, y), 2, (0, 255, 255), -1)
        for (x, y) in result.right_eye:
            cv2.circle(frame, (x, y), 2, (0, 255, 255), -1)

        left_hull = cv2.convexHull(np.array(result.left_eye))
        cv2.drawContours(frame, [left_hull], -1, (0, 255, 255), 1)

        right_hull = cv2.convexHull(np.array(result.right_eye))
        cv2.drawContours(frame, [right_hull], -1, (0, 255, 255), 1)

        return frame


# ── FOR STANDALONE TESTING ──────────────────────────────
if __name__ == "__main__":
    """
    Test face/eye detection independently:
        python src/face_eye.py

    What you should see:
        - Your webcam opens in a window
        - Yellow dots appear on your eye landmarks when your face
          is detected
        - "No face detected" appears when you cover/turn your face
        - Press 'q' to quit

    Prerequisites:
        - shape_predictor_68_face_landmarks.dat must be in the
          project root (or adjust the path below)

    What to implement:
        1. cap = cv2.VideoCapture(0)
        2. detector = FaceEyeDetector("shape_predictor_68_face_landmarks.dat")
        3. while True:
            a. ret, frame = cap.read()
            b. if not ret: break
            c. result = detector.detect(frame)
            d. detector.draw_on_frame(frame, result)
            e. If face detected, print the eye coordinates for debugging:
               print(f"Left eye: {result.left_eye}")
               print(f"Right eye: {result.right_eye}")
            f. cv2.imshow("Face & Eye Detection Test - Deno", frame)
            g. if cv2.waitKey(1) & 0xFF == ord('q'): break
        4. cap.release()
        5. cv2.destroyAllWindows()

    Debugging tips:
        - If face detection is slow, that's normal on lower-end
          hardware. Consider detecting every 2nd or 3rd frame.
        - If landmarks look wrong, make sure you're using the correct
          .dat file (68 landmarks, not 5)
        - Make sure you're converting to grayscale BEFORE detection
    """
    cap = cv2.VideoCapture(0)
    detector = FaceEyeDetector("shape_predictor_68_face_landmarks.dat")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result = detector.detect(frame)
        detector.draw_on_frame(frame, result)

        if result.face_detected:
            print(f"Left eye: {result.left_eye}")
            print(f"Right eye: {result.right_eye}")

        cv2.imshow("Face & Eye Detection Test - Deno", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()