"""
Module: alerts.py
Owner: Ritchie (Rich Victor Kariuki)
Branch: ritchie

═══════════════════════════════════════════════════════════
PURPOSE
═══════════════════════════════════════════════════════════
This module is the OUTPUT stage of the pipeline. You take
detection results from the other modules and turn them into
things the user can see and hear:

  1. VISUAL ALERTS on the camera feed (colored borders, warning text)
  2. LED ALERTS via GPIO (green/amber/red tiered system)
  3. BUZZER ALERTS via GPIO (beeps and continuous alarm)
  4. MQTT MESSAGES to the Raspberry Pi (when running on laptop)

═══════════════════════════════════════════════════════════
HOW IT FITS IN THE PIPELINE
═══════════════════════════════════════════════════════════
In main.py, your module is called last every frame:

    motion_result = motion.detect(frame)           # Melanie & Stan
    if motion_result.detected:                     # Gated! (per spec)
        face_result = face_eye.detect(frame)       # Deno
        ear_result = ear_calc.calculate(...)       # Timo

    alerts.check_motion(motion_result)             # YOUR MODULE
    alerts.check_drowsiness(ear_result)            # YOUR MODULE
    alerts.update_leds()                           # YOUR MODULE (GPIO)
    alerts.draw_alerts(frame)                      # YOUR MODULE (visual)

═══════════════════════════════════════════════════════════
TIERED ALERT SYSTEM (from EyeGuard spec section 4.3)
═══════════════════════════════════════════════════════════
The spec defines THREE alert tiers:

  ┌──────────┬──────────────────┬─────────────────────────┐
  │  Tier    │  LED State       │  Buzzer Behavior        │
  ├──────────┼──────────────────┼─────────────────────────┤
  │ NORMAL   │ Green ON (steady)│ Silent                  │
  │ WARNING  │ Amber slow blink │ Short beep every 2 sec  │
  │ CRITICAL │ Red fast flash   │ Continuous alarm        │
  └──────────┴──────────────────┴─────────────────────────┘

  NORMAL:   No motion or eyes are open (EAR >= 0.25)
  WARNING:  Eyes are closing (EAR < 0.25 but not yet 20 frames)
  CRITICAL: Drowsiness confirmed (EAR < 0.25 for 20+ frames)

═══════════════════════════════════════════════════════════
GPIO PIN ASSIGNMENTS (from EyeGuard spec section 5.2)
═══════════════════════════════════════════════════════════

  Component       │ GPIO Pin │ Physical Pin │ Function
  ────────────────┼──────────┼──────────────┼─────────────────
  Buzzer (+)      │ GPIO 27  │ Pin 13       │ Audible alert
  Red LED anode   │ GPIO 17  │ Pin 11       │ Critical/warning
  Green LED anode │ GPIO 22  │ Pin 15       │ Normal/active
  LED cathodes    │ GND      │ Pin 9, 14    │ Ground
  Buzzer (-)      │ GND      │ Pin 9        │ Ground

  Amber is achieved by turning ON BOTH Red (GPIO 17) and
  Green (GPIO 22) simultaneously on an RGB LED. If using
  separate LEDs, both light up together for amber.

═══════════════════════════════════════════════════════════
LAPTOP SIMULATION MODE (from spec section 6.5)
═══════════════════════════════════════════════════════════
When running on a laptop (no GPIO available), the system
must gracefully fall back to software-only alerts:

  Hardware         │ Laptop Equivalent
  ─────────────────┼──────────────────────────────
  GPIO buzzer      │ winsound.Beep() on Windows,
                   │ or terminal bell ('\a') on Linux/Mac
  LED indicators   │ Colored rectangles on OpenCV frame
  USB / webcam     │ cv2.VideoCapture(0) — identical

Your code should detect whether GPIO is available and
switch between hardware and software alerts automatically.

═══════════════════════════════════════════════════════════
STEP-BY-STEP IMPLEMENTATION GUIDE
═══════════════════════════════════════════════════════════
Step 1: Get __init__ and _connect_mqtt() working
  - Try connecting to the Pi's MQTT broker
  - Handle the case where the Pi isn't reachable

Step 2: Implement _setup_gpio()
  - Try importing RPi.GPIO
  - If import fails (laptop), set gpio_available = False
  - If available, set up pins 17, 22, 27 as outputs

Step 3: Implement check_motion() and check_drowsiness()
  - Update the AlertState based on incoming results
  - Determine the current alert tier (NORMAL/WARNING/CRITICAL)

Step 4: Implement update_leds()
  - Control GPIO pins based on current alert tier
  - On laptop, this becomes a no-op (visual alerts handle it)

Step 5: Implement _publish()
  - Send JSON messages to MQTT topics with cooldown

Step 6: Implement draw_alerts()
  - Draw colored borders, text, and simulated LED indicators
  - On laptop, draw colored circles to simulate the LEDs

Step 7: Test standalone with simulated events

═══════════════════════════════════════════════════════════
COMMON PITFALLS
═══════════════════════════════════════════════════════════
- MQTT connection failure should NOT crash the system.
- GPIO import failure should NOT crash the system (laptop mode).
- Cooldown timing: Use time.time() to track last publish time.
- OpenCV colors are BGR (Blue, Green, Red), NOT RGB:
    Red = (0, 0, 255)    Orange/Amber = (0, 200, 255)
    Green = (0, 255, 0)  White = (255, 255, 255)
- For amber LED: turn on BOTH GPIO 17 (red) and GPIO 22 (green)
- Make sure to call client.loop_start() after connecting MQTT.
- When running on the Pi itself, MQTT connects to localhost.
  When running on a laptop, MQTT connects to the Pi's IP.
"""

import cv2
import json
import time
import platform
import os

# ═══════════════════════════════════════════════════════════
# CONDITIONAL IMPORTS
# GPIO is only available on the Raspberry Pi.
# MQTT may not be needed if running everything on the Pi.
# These are imported with try/except so the module works
# on any platform.
# ═══════════════════════════════════════════════════════════
try:
    import paho.mqtt.client as mqtt
    MQTT_AVAILABLE = True
except ImportError:
    MQTT_AVAILABLE = False

# ═══════════════════════════════════════════════════════════
# Result type imports — uncomment when integrating:
# from motion import MotionResult
# from ear_logic import EARResult
# ═══════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════
# ALERT TIER CONSTANTS
# These match the EyeGuard spec section 4.3
# ═══════════════════════════════════════════════════════════
TIER_NORMAL = "NORMAL"
TIER_WARNING = "WARNING"
TIER_CRITICAL = "CRITICAL"

# ═══════════════════════════════════════════════════════════
# GPIO PIN ASSIGNMENTS (from EyeGuard spec section 5.2)
# ═══════════════════════════════════════════════════════════
GPIO_RED_LED = 17       # Pin 11 — Critical / warning alert
GPIO_GREEN_LED = 22     # Pin 15 — Normal / active state
GPIO_BUZZER = 27        # Pin 13 — Audible alert output


from dataclasses import dataclass


@dataclass
class AlertState:
    """
    Tracks the current alert status for rendering each frame.

    Attributes:
        motion_active: True if motion was detected this frame
        eyes_closed: True if EAR is below threshold (WARNING tier)
        drowsy_active: True if drowsy for 20+ frames (CRITICAL tier)
        current_tier: One of TIER_NORMAL, TIER_WARNING, TIER_CRITICAL
        last_motion_time: Timestamp of last motion MQTT publish
        last_drowsy_time: Timestamp of last drowsy MQTT publish
        last_warning_time: Timestamp of last warning MQTT publish
    """
    motion_active: bool = False
    eyes_closed: bool = False
    drowsy_active: bool = False
    current_tier: str = TIER_NORMAL
    last_motion_time: float = 0.0
    last_drowsy_time: float = 0.0
    last_warning_time: float = 0.0


class AlertManager:
    def __init__(self, pi_ip="192.168.x.x", mqtt_port=1883, cooldown=2.0):
        """
        Initialize the alert system.

        Args:
            pi_ip: IP address of the Raspberry Pi running Mosquitto.
                   Replace "192.168.x.x" with the actual IP.
                   Use "localhost" if running on the Pi itself.
            mqtt_port: Port for MQTT (default 1883)
            cooldown: Minimum seconds between MQTT publishes
        """
        # Store configuration
        self.pi_ip = pi_ip
        self.mqtt_port = mqtt_port
        self.cooldown = cooldown

        # Define MQTT topics
        self.topic_motion = "alerts/motion"
        self.topic_warning = "alerts/warning"
        self.topic_drowsy = "alerts/drowsiness"

        # Create alert state
        self.state = AlertState()

        # Set up GPIO (handles laptop fallback)
        self._setup_gpio()

        # Connect to MQTT
        self.mqtt_connected = False
        self._connect_mqtt(pi_ip, mqtt_port)

        # Track LED blink timing
        self.last_blink_time = time.time()
        self.blink_state = False  # toggles for blinking effect

    def _setup_gpio(self):
        """
        Set up GPIO pins for LEDs and buzzer.

        If running on a Raspberry Pi, this configures the actual
        hardware pins. If running on a laptop, it sets a flag to
        use software-only alerts.
        """
        try:
            import RPi.GPIO as GPIO
            self.GPIO = GPIO
            self.gpio_available = True

            GPIO.setmode(GPIO.BCM)    # Use BCM pin numbering
            GPIO.setwarnings(False)   # Suppress warnings

            # Set up output pins
            GPIO.setup(GPIO_RED_LED, GPIO.OUT)
            GPIO.setup(GPIO_GREEN_LED, GPIO.OUT)
            GPIO.setup(GPIO_BUZZER, GPIO.OUT)

            # Start in NORMAL state: green LED on, others off
            GPIO.output(GPIO_GREEN_LED, GPIO.HIGH)
            GPIO.output(GPIO_RED_LED, GPIO.LOW)
            GPIO.output(GPIO_BUZZER, GPIO.LOW)

            print("[GPIO] Hardware alerts initialized")
            print(f"  Red LED:   GPIO {GPIO_RED_LED} (Pin 11)")
            print(f"  Green LED: GPIO {GPIO_GREEN_LED} (Pin 15)")
            print(f"  Buzzer:    GPIO {GPIO_BUZZER} (Pin 13)")

        except (ImportError, RuntimeError):
            self.GPIO = None
            self.gpio_available = False
            print("[GPIO] Not available — running in laptop simulation mode")
            print("[GPIO] LEDs will be simulated as colored circles on screen")

    def _connect_mqtt(self, pi_ip, port):
        """
        Attempt to connect to the MQTT broker on the Pi.

        This should NEVER crash the program.

        Args:
            pi_ip: Raspberry Pi IP address
            port: MQTT port number
        """
        if not MQTT_AVAILABLE:
            self.mqtt_connected = False
            self.client = None
            print("[MQTT] paho-mqtt not installed — skipping")
            return

        try:
            self.client = mqtt.Client()
            self.client.connect(pi_ip, port, 60)
            self.client.loop_start()
            self.mqtt_connected = True
            print(f"[MQTT] Connected to broker at {pi_ip}:{port}")
        except Exception as e:
            self.mqtt_connected = False
            self.client = None
            print(f"[MQTT] Could not connect: {e}")
            print("[MQTT] Running without MQTT (visual + GPIO alerts only)")

    def _publish(self, topic, payload):
        """
        Publish a JSON message to an MQTT topic with cooldown.

        Args:
            topic: MQTT topic string
            payload: Dictionary to be sent as JSON
        """
        # Check if MQTT is connected
        if not self.mqtt_connected or self.client is None:
            return

        # Check cooldown based on topic
        now = time.time()
        if topic == self.topic_motion:
            if now - self.state.last_motion_time < self.cooldown:
                return
            self.state.last_motion_time = now
        elif topic == self.topic_warning:
            if now - self.state.last_warning_time < self.cooldown:
                return
            self.state.last_warning_time = now
        elif topic == self.topic_drowsy:
            if now - self.state.last_drowsy_time < self.cooldown:
                return
            self.state.last_drowsy_time = now

        # Serialize and publish
        try:
            json_str = json.dumps(payload)
            self.client.publish(topic, json_str)
        except Exception as e:
            print(f"[MQTT] Publish error: {e}")

    def _determine_tier(self):
        """
        Determine the current alert tier based on state.

        The tier system (from EyeGuard spec):
          CRITICAL: Drowsiness confirmed (20+ frames eyes closed)
          WARNING:  Eyes are closing (EAR < 0.25 but < 20 frames)
          NORMAL:   Eyes open or no face detected
        """
        if self.state.drowsy_active:
            self.state.current_tier = TIER_CRITICAL
        elif self.state.eyes_closed:
            self.state.current_tier = TIER_WARNING
        else:
            self.state.current_tier = TIER_NORMAL

    def check_motion(self, motion_result):
        """
        Process a motion detection result.

        Args:
            motion_result: MotionResult from Melanie & Stan's module.
                           Has a .detected attribute (bool).
        """
        # Update visual state
        self.state.motion_active = motion_result.detected

        # If motion detected, publish to MQTT
        if motion_result.detected:
            self._publish(self.topic_motion, {
                "type": "motion",
                "timestamp": time.time()
            })

    def check_drowsiness(self, ear_result):
        """
        Process an EAR/drowsiness result and update alert tier.

        Args:
            ear_result: EARResult from Timo's module.
                        Can be None if no face was detected or
                        no motion was detected (face detection was skipped).
        """
        # Handle None (no face or no motion)
        if ear_result is None:
            self.state.eyes_closed = False
            self.state.drowsy_active = False
            self._determine_tier()
            return

        # Update states from EAR result
        self.state.eyes_closed = ear_result.eyes_closed
        self.state.drowsy_active = ear_result.drowsy

        # Determine the alert tier
        self._determine_tier()

        # Publish based on tier
        if self.state.current_tier == TIER_CRITICAL:
            self._publish(self.topic_drowsy, {
                "type": "drowsiness",
                "tier": "critical",
                "ear_value": ear_result.ear_value,
                "consecutive_frames": ear_result.consecutive_frames,
                "timestamp": time.time()
            })
        elif self.state.current_tier == TIER_WARNING:
            self._publish(self.topic_warning, {
                "type": "warning",
                "tier": "warning",
                "ear_value": ear_result.ear_value,
                "consecutive_frames": ear_result.consecutive_frames,
                "timestamp": time.time()
            })

    def update_leds(self):
        """
        Update GPIO LED and buzzer states based on current tier.

        This is a no-op on laptops (gpio_available == False).
        On the Pi, it directly controls the hardware.
        """
        if not self.gpio_available:
            return

        now = time.time()

        if self.state.current_tier == TIER_NORMAL:
            # Green LED steady on, red off, buzzer off
            self.GPIO.output(GPIO_GREEN_LED, self.GPIO.HIGH)
            self.GPIO.output(GPIO_RED_LED, self.GPIO.LOW)
            self.GPIO.output(GPIO_BUZZER, self.GPIO.LOW)

        elif self.state.current_tier == TIER_WARNING:
            # Amber = Red + Green both on, slow blink (toggle every 0.5s)
            if now - self.last_blink_time > 0.5:
                self.blink_state = not self.blink_state
                self.last_blink_time = now

            if self.blink_state:
                self.GPIO.output(GPIO_RED_LED, self.GPIO.HIGH)
                self.GPIO.output(GPIO_GREEN_LED, self.GPIO.HIGH)
            else:
                self.GPIO.output(GPIO_RED_LED, self.GPIO.LOW)
                self.GPIO.output(GPIO_GREEN_LED, self.GPIO.LOW)

            # Short beep every 2 seconds
            if now - self.state.last_warning_time >= 2.0:
                self.GPIO.output(GPIO_BUZZER, self.GPIO.HIGH)
                time.sleep(0.1)   # 100ms beep
                self.GPIO.output(GPIO_BUZZER, self.GPIO.LOW)
                self.state.last_warning_time = now

        elif self.state.current_tier == TIER_CRITICAL:
            # Red LED fast flash (toggle every 0.2s)
            if now - self.last_blink_time > 0.2:
                self.blink_state = not self.blink_state
                self.last_blink_time = now

            self.GPIO.output(GPIO_RED_LED,
                self.GPIO.HIGH if self.blink_state else self.GPIO.LOW)
            self.GPIO.output(GPIO_GREEN_LED, self.GPIO.LOW)

            # Continuous buzzer
            self.GPIO.output(GPIO_BUZZER, self.GPIO.HIGH)

    def draw_alerts(self, frame):
        """
        Draw visual alert overlays on the camera frame.

        This is called EVERY FRAME. On laptops, this also draws
        simulated LED indicators since real GPIO isn't available.

        Args:
            frame: Current camera frame to annotate (modified in place)

        Returns:
            The annotated frame
        """
        # Get frame dimensions
        h, w = frame.shape[:2]

        # Draw tier-specific alerts
        if self.state.current_tier == TIER_CRITICAL:
            # Red border (fast flash simulated by alternating thickness)
            thickness = 8 if int(time.time() * 5) % 2 == 0 else 4
            cv2.rectangle(frame, (0, 0), (w, h), (0, 0, 255), thickness)
            cv2.putText(frame, "DROWSINESS ALERT - CRITICAL!",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        1.0, (0, 0, 255), 2)

        elif self.state.current_tier == TIER_WARNING:
            # Amber/orange border
            cv2.rectangle(frame, (0, 0), (w, h), (0, 200, 255), 6)
            cv2.putText(frame, "WARNING - Eyes Closing",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 200, 255), 2)

        # If motion is active, show motion indicator
        if self.state.motion_active:
            cv2.putText(frame, "MOTION DETECTED",
                        (10, 65), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 0), 2)

        # Draw simulated LED indicators (especially useful on laptop)
        # Simulated LEDs in the top-right corner of the frame
        led_x = w - 80
        led_y_green = 30
        led_y_red = 60

        if self.state.current_tier == TIER_NORMAL:
            # Green LED on (filled circle), red LED off (outline)
            cv2.circle(frame, (led_x, led_y_green), 12, (0, 255, 0), -1)
            cv2.circle(frame, (led_x, led_y_red), 12, (0, 0, 100), 1)
        elif self.state.current_tier == TIER_WARNING:
            # Both on = amber (if blinking, simulate that too)
            blink = int(time.time() * 2) % 2 == 0
            color_g = (0, 255, 0) if blink else (0, 100, 0)
            color_r = (0, 200, 255) if blink else (0, 100, 100) # Amber-ish
            cv2.circle(frame, (led_x, led_y_green), 12, color_g, -1)
            cv2.circle(frame, (led_x, led_y_red), 12, color_r, -1)
        elif self.state.current_tier == TIER_CRITICAL:
            # Red on, green off
            blink = int(time.time() * 5) % 2 == 0
            color_r = (0, 0, 255) if blink else (0, 0, 100)
            cv2.circle(frame, (led_x, led_y_green), 12, (0, 100, 0), 1)
            cv2.circle(frame, (led_x, led_y_red), 12, color_r, -1)

        # Labels
        cv2.putText(frame, "G", (led_x - 6, led_y_green + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(frame, "R", (led_x - 5, led_y_red + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # Draw status bar at bottom
        tier_text = f"Tier: {self.state.current_tier}"
        motion_text = "MOTION" if self.state.motion_active else "STILL"
        status = f"{tier_text} | {motion_text}"
        cv2.putText(frame, status, (10, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 2)

        return frame

    def cleanup(self):
        """
        Disconnect MQTT and clean up GPIO resources.

        Called when the program exits.
        """
        # Clean up MQTT
        if self.mqtt_connected and self.client is not None:
            self.client.loop_stop()
            self.client.disconnect()
            print("[MQTT] Disconnected")

        # Clean up GPIO
        if self.gpio_available and self.GPIO is not None:
            self.GPIO.output(GPIO_BUZZER, self.GPIO.LOW)    # Silence buzzer
            self.GPIO.output(GPIO_RED_LED, self.GPIO.LOW)   # LEDs off
            self.GPIO.output(GPIO_GREEN_LED, self.GPIO.LOW)
            self.GPIO.cleanup()
            print("[GPIO] Cleaned up")


# ── FOR STANDALONE TESTING ──────────────────────────────
if __name__ == "__main__":
    """
    Test alerts independently:
        python src/alerts.py

    This test simulates cycling through all three alert tiers
    to verify visual overlays, LED simulation, and GPIO control.
    """
    # 1. Create AlertManager
    alerts = AlertManager(pi_ip="localhost", cooldown=2.0)

    # 2. Open camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera. Testing with dummy frame.")
        import numpy as np
        frame_dummy = np.zeros((480, 640, 3), dtype=np.uint8)
    else:
        frame_dummy = None

    # 3. Create mock result classes
    class MockMotion:
        def __init__(self, detected):
            self.detected = detected

    class MockEAR:
        def __init__(self, eyes_closed, drowsy, ear_value=0.20, consec=25):
            self.eyes_closed = eyes_closed
            self.drowsy = drowsy
            self.ear_value = ear_value
            self.consecutive_frames = consec

    # 4. Loop cycling through tiers
    frame_count = 0
    print("Starting alert test loop. Press 'q' to quit.")

    try:
        while True:
            if cap is not None and cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
            else:
                frame = frame_dummy.copy()

            # Cycle: NORMAL → WARNING → CRITICAL every 90 frames (approx 3s at 30fps)
            cycle = (frame_count // 90) % 3

            if cycle == 0:      # NORMAL
                alerts.check_motion(MockMotion(detected=False))
                alerts.check_drowsiness(MockEAR(False, False, 0.30, 0))
            elif cycle == 1:    # WARNING
                alerts.check_motion(MockMotion(detected=True))
                alerts.check_drowsiness(MockEAR(True, False, 0.22, 10))
            elif cycle == 2:    # CRITICAL
                alerts.check_motion(MockMotion(detected=True))
                alerts.check_drowsiness(MockEAR(True, True, 0.18, 25))

            alerts.update_leds()
            alerts.draw_alerts(frame)

            # Headless safe display simulation
            try:
                if "DISPLAY" in os.environ:
                    cv2.imshow("Alert Tier Test - Ritchie", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'): break
                else:
                    # In headless, just print the current tier to verify logic
                    print(f"Frame {frame_count:03d}: {alerts.state.current_tier:8s} | Motion: {alerts.state.motion_active}", end="\r")
                    if frame_count >= 300: break # Run for a bit and stop
            except cv2.error:
                # Fallback for headless if DISPLAY is set but not functional
                print(f"Frame {frame_count:03d}: {alerts.state.current_tier:8s} | Motion: {alerts.state.motion_active}", end="\r")
                if frame_count >= 300: break # Run for a bit and stop

            frame_count += 1
            time.sleep(0.033) # Simulate 30fps
    except KeyboardInterrupt:
        pass
    finally:
        # 5. Cleanup
        if cap is not None and cap.isOpened():
            cap.release()
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass
        alerts.cleanup()
        print("\nTest complete.")