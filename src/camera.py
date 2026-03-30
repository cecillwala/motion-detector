"""
Module: camera.py
Owner: Silvana
Branch: silvana

═══════════════════════════════════════════════════════════
PURPOSE
═══════════════════════════════════════════════════════════
This module is the ENTRY POINT of the entire pipeline.
Every other module depends on you — they all need frames
from the camera to do their work. Your job is to:

  1. Open the laptop's webcam using OpenCV
  2. Continuously capture frames in a loop
  3. Provide a clean interface so other modules can grab frames
  4. Display the final annotated frame (after all modules have
     drawn their overlays on it)
  5. Handle errors gracefully (camera not found, disconnected, etc.)

═══════════════════════════════════════════════════════════
HOW IT FITS IN THE PIPELINE
═══════════════════════════════════════════════════════════
Your module is called first in main.py:

    camera = CameraStream(source=0)
    while True:
        ret, frame = camera.read_frame()   <-- your code provides this
        # frame goes to motion.py, face_eye.py, etc.
        cv2.imshow("Detection", frame)     <-- your code displays this

═══════════════════════════════════════════════════════════
KEY CONCEPTS YOU'LL USE
═══════════════════════════════════════════════════════════
- cv2.VideoCapture(source): Opens a camera. source=0 means the
  default webcam. You can also pass a video file path for testing.

- cap.read(): Returns (bool, frame). The bool is True if a frame
  was successfully captured. The frame is a numpy array of shape
  (height, width, 3) in BGR color format.

- cap.set(cv2.CAP_PROP_FRAME_WIDTH, w): Sets camera resolution.
  Not all cameras support all resolutions — 640x480 is safe.

- cv2.imshow(window_name, frame): Displays a frame in a window.

- cv2.waitKey(1): Waits 1ms for a keypress. Returns the key code.
  Use (cv2.waitKey(1) & 0xFF == ord('q')) to detect 'q' press.

- Context managers (__enter__/__exit__): Let you use the class with
  Python's 'with' statement for automatic cleanup:
      with CameraStream() as cam:
          ret, frame = cam.read_frame()
      # camera is automatically released when 'with' block ends

═══════════════════════════════════════════════════════════
STEP-BY-STEP IMPLEMENTATION GUIDE
═══════════════════════════════════════════════════════════
Step 1: Get __init__ working
  - Create the VideoCapture object
  - Set resolution
  - Check if it opened successfully, raise error if not
  - Test: python src/camera.py should not crash

Step 2: Implement read_frame()
  - Call self.cap.read()
  - Return the tuple (success, frame)
  - Test: you should see print output of frame shapes

Step 3: Implement release()
  - Call self.cap.release()
  - Call cv2.destroyAllWindows()

Step 4: Build the standalone test (__main__ block)
  - Create CameraStream, loop read_frame, display with imshow
  - You should see your live webcam feed in a window
  - Press 'q' to quit cleanly

Step 5: Add get_frame_dimensions()
  - Useful for other modules that need to know frame size

═══════════════════════════════════════════════════════════
COMMON PITFALLS
═══════════════════════════════════════════════════════════
- If the camera doesn't open, check if another app is using it
- Always call release() when done, or the camera stays locked
- cv2.imshow() requires cv2.waitKey() to actually render — without
  waitKey the window will appear frozen
- On Linux, you may need to install: sudo apt install python3-opencv
"""

import cv2


class CameraStream:
    def __init__(self, source="tcp://192.168.122.1:5000", width=640, height=480):
        """
        Initialize the camera.

        Args:
            source: Camera index (0 = default webcam) or a video file path
                    as a string (e.g., "test_video.mp4") for testing without
                    a live camera
            width: Desired frame width in pixels
            height: Desired frame height in pixels

        What to do:
            1. Create self.cap = cv2.VideoCapture(source)
            2. Set the width:  self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            3. Set the height: self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            4. Check if it opened:
                if not self.cap.isOpened():
                    raise RuntimeError(f"Could not open camera source: {source}")
            5. Store width and height as instance variables for later use
            6. Print a confirmation message so you know it worked

        Raises:
            RuntimeError: If the camera cannot be opened
        """
        # YOUR CODE HERE
        pass

    def read_frame(self):
        """
        Capture a single frame from the camera.

        Returns:
            tuple: (success, frame)
                - success (bool): True if frame was captured
                - frame (numpy.ndarray): The image as a HxWx3 BGR array,
                  or None if capture failed

        What to do:
            1. Call: ret, frame = self.cap.read()
            2. Return (ret, frame)

        That's it! This is simple but critical — every other module
        calls this method every frame (typically 30 times per second).
        """
        # YOUR CODE HERE
        pass

    def get_frame_dimensions(self):
        """
        Get the dimensions of frames from this camera.

        Returns:
            tuple: (width, height) as integers

        What to do:
            1. Read the properties from the capture object:
               w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
               h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            2. Return (w, h)

        This is used by other modules (especially Melanie & Stan's
        motion detection) to know the frame size for drawing.
        """
        # YOUR CODE HERE
        pass

    def release(self):
        """
        Release camera resources. MUST be called when done.

        What to do:
            1. Call self.cap.release() to free the camera hardware
            2. Call cv2.destroyAllWindows() to close any display windows

        If you skip this, the camera may stay locked and other apps
        (or even re-running this script) won't be able to access it.
        """
        # YOUR CODE HERE
        pass

    def __enter__(self):
        """Allows using this class with Python's 'with' statement."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Automatically calls release() when the 'with' block ends."""
        self.release()


# ── FOR STANDALONE TESTING ──────────────────────────────
if __name__ == "__main__":
    """
    Run this file directly to test the camera:
        python src/camera.py

    What you should see:
        - A window opens showing your live webcam feed
        - The frame dimensions are printed to the terminal
        - Pressing 'q' closes the window and exits cleanly
        - No error messages

    What to implement below:
        1. Create a CameraStream instance (use default source=0)
        2. Print the frame dimensions using get_frame_dimensions()
        3. Start a while True loop:
            a. Call read_frame() to get a frame
            b. If read failed (ret is False), print error and break
            c. Display the frame: cv2.imshow("Camera Test - Silvana", frame)
            d. Check for 'q' keypress:
               if cv2.waitKey(1) & 0xFF == ord('q'):
                   break
        4. After the loop, call release() to clean up

    Bonus (optional):
        - Add an FPS counter: measure time between frames and display
          it on the frame using cv2.putText()
        - This helps the team know if the camera is running smoothly
    """
    # YOUR CODE HERE
    pass

import cv2
import time


class CameraStream:

    def __init__(self, source="tcp://192.168.122.1:5000", width=640, height=480):
        # Open the webcam — source=0 means the laptop's built-in camera
        self.cap = cv2.VideoCapture(source)

        # Set the resolution to 640x480 pixels
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        # Stop the program if the camera could not be opened
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open camera source: {source}")

        # Save width and height so other methods can use them
        self.width = width
        self.height = height

        print(f"[Camera] Webcam opened. Resolution: {width} x {height}")

    def read_frame(self):
        # Capture one frame from the webcam
        # ret  = True if it worked, False if something went wrong
        # frame = the actual image as a grid of pixel values
        ret, frame = self.cap.read()

        return (ret, frame)

    def get_frame_dimensions(self):
        # Read the actual width and height from the camera
        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        return (w, h)

    def release(self):
        # Free the camera so other apps can use it
        self.cap.release()

        # Close any open OpenCV windows
        cv2.destroyAllWindows()

        print("[Camera] Camera released.")

    def __enter__(self):
        # Allows using this class with Python's 'with' statement
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Automatically release the camera when 'with' block ends
        self.release()


if __name__ == "__main__":

    # Open the webcam
    camera = CameraStream(source="tcp://192.168.122.1:5000")

    # Print the frame size to the terminal
    width, height = camera.get_frame_dimensions()
    print(f"[Camera] Frame size: {width} x {height}")
    print("[Camera] Press 'q' to quit")

    # Track time so we can calculate FPS
    prev_time = time.time()

    while True:
        # Grab one frame from the webcam
        ret, frame = camera.read_frame()

        # If the frame failed, stop the loop
        if not ret:
            print("[Camera] Failed to read frame. Stopping.")
            break

        # Calculate how many frames are showing per second (FPS)
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time

        # Draw the FPS number on the video in green (top-left)
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Draw the system label at the bottom of the video
        cv2.putText(frame, "EyeGuard System", (10, height - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # Show the frame in a pop-up window
        cv2.imshow("EyeGuard - Live Feed", frame)

        # Wait 1ms and check if the user pressed 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Clean up the camera when done
    camera.release()
