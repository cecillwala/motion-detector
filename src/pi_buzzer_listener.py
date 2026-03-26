"""
EyeGuard — Raspberry Pi Alert Listener

This script runs ON THE RASPBERRY PI and listens for MQTT
messages from the laptop. When a message arrives, it controls
the GPIO hardware: buzzer and LEDs.

GPIO Pin Assignments (from EyeGuard spec section 5.2):
  Buzzer (+)      → GPIO 27 (Pin 13)
  Red LED anode   → GPIO 17 (Pin 11)
  Green LED anode → GPIO 22 (Pin 15)
  All grounds     → GND (Pin 9, 14)

Alert Tiers (from spec section 4.3):
  NORMAL:   Green LED steady ON, buzzer silent
  WARNING:  Amber (Red+Green) slow blink, short beep every 2s
  CRITICAL: Red LED fast flash, continuous buzzer alarm

Run with: python3 pi_buzzer_listener.py
"""

import json
import time
import threading

# ── GPIO SETUP ──────────────────────────────────────────
try:
    import RPi.GPIO as GPIO

    GPIO_RED_LED = 17
    GPIO_GREEN_LED = 22
    GPIO_BUZZER = 27

    GPIO.setmode(GPIO.BCM)
    GPIO.setwarnings(False)
    GPIO.setup(GPIO_RED_LED, GPIO.OUT)
    GPIO.setup(GPIO_GREEN_LED, GPIO.OUT)
    GPIO.setup(GPIO_BUZZER, GPIO.OUT)

    # Start in NORMAL state
    GPIO.output(GPIO_GREEN_LED, GPIO.HIGH)
    GPIO.output(GPIO_RED_LED, GPIO.LOW)
    GPIO.output(GPIO_BUZZER, GPIO.LOW)

    GPIO_AVAILABLE = True
    print("[GPIO] Initialized:")
    print(f"  Red LED:   GPIO {GPIO_RED_LED} (Pin 11)")
    print(f"  Green LED: GPIO {GPIO_GREEN_LED} (Pin 15)")
    print(f"  Buzzer:    GPIO {GPIO_BUZZER} (Pin 13)")

except (ImportError, RuntimeError) as e:
    GPIO_AVAILABLE = False
    print(f"[GPIO] Not available: {e}")
    print("[GPIO] Running in terminal-only mode (print alerts)")

# ── MQTT SETUP ──────────────────────────────────────────
import paho.mqtt.client as mqtt

# Track current alert tier for LED control
current_tier = "NORMAL"
tier_lock = threading.Lock()


def set_tier(tier):
    """Update the current alert tier (thread-safe)."""
    global current_tier
    with tier_lock:
        current_tier = tier


def buzz(duration=0.5):
    """Activate buzzer for a given duration."""
    if GPIO_AVAILABLE:
        GPIO.output(GPIO_BUZZER, GPIO.HIGH)
        time.sleep(duration)
        GPIO.output(GPIO_BUZZER, GPIO.LOW)
    else:
        print(f"\a[BUZZ] Simulated buzzer for {duration}s")


def led_normal():
    """Green LED steady on, red off."""
    if GPIO_AVAILABLE:
        GPIO.output(GPIO_GREEN_LED, GPIO.HIGH)
        GPIO.output(GPIO_RED_LED, GPIO.LOW)
        GPIO.output(GPIO_BUZZER, GPIO.LOW)


def led_warning():
    """Amber (Red+Green both on)."""
    if GPIO_AVAILABLE:
        GPIO.output(GPIO_GREEN_LED, GPIO.HIGH)
        GPIO.output(GPIO_RED_LED, GPIO.HIGH)


def led_critical():
    """Red LED on, green off."""
    if GPIO_AVAILABLE:
        GPIO.output(GPIO_GREEN_LED, GPIO.LOW)
        GPIO.output(GPIO_RED_LED, GPIO.HIGH)


def led_control_loop():
    """
    Background thread that continuously updates LEDs based
    on the current alert tier. Handles blinking patterns.
    """
    blink_state = False
    last_blink = time.time()
    last_beep = time.time()

    while True:
        now = time.time()

        with tier_lock:
            tier = current_tier

        if tier == "NORMAL":
            led_normal()
            time.sleep(0.1)

        elif tier == "WARNING":
            # Amber slow blink (toggle every 0.5s)
            if now - last_blink > 0.5:
                blink_state = not blink_state
                last_blink = now

            if blink_state:
                led_warning()
            else:
                if GPIO_AVAILABLE:
                    GPIO.output(GPIO_GREEN_LED, GPIO.LOW)
                    GPIO.output(GPIO_RED_LED, GPIO.LOW)

            # Short beep every 2 seconds
            if now - last_beep >= 2.0:
                buzz(0.1)
                last_beep = now

            time.sleep(0.05)

        elif tier == "CRITICAL":
            # Red fast flash (toggle every 0.2s)
            if now - last_blink > 0.2:
                blink_state = not blink_state
                last_blink = now

            if blink_state:
                led_critical()
            else:
                if GPIO_AVAILABLE:
                    GPIO.output(GPIO_RED_LED, GPIO.LOW)

            # Continuous buzzer
            if GPIO_AVAILABLE:
                GPIO.output(GPIO_BUZZER, GPIO.HIGH)

            time.sleep(0.05)


# ── MQTT CALLBACKS ──────────────────────────────────────
def on_connect(client, userdata, flags, rc):
    print(f"[MQTT] Connected to broker (code {rc})")
    client.subscribe("alerts/#")
    print("[MQTT] Subscribed to alerts/#")
    print("[MQTT] Waiting for messages...\n")


def on_message(client, userdata, msg):
    try:
        data = json.loads(msg.payload.decode())
        alert_type = data.get("type", "unknown")
        tier = data.get("tier", "")
        ts = time.strftime("%H:%M:%S")

        if alert_type == "motion":
            print(f"[{ts}] MOTION detected")
            set_tier("NORMAL")  # Motion alone = normal tier

        elif alert_type == "warning":
            ear = data.get("ear_value", 0)
            frames = data.get("consecutive_frames", 0)
            print(f"[{ts}] WARNING — Eyes closing (EAR: {ear:.3f}, "
                  f"frames: {frames})")
            set_tier("WARNING")

        elif alert_type == "drowsiness":
            ear = data.get("ear_value", 0)
            frames = data.get("consecutive_frames", 0)
            print(f"[{ts}] *** CRITICAL — DROWSINESS DETECTED *** "
                  f"(EAR: {ear:.3f}, frames: {frames})")
            set_tier("CRITICAL")

    except json.JSONDecodeError:
        print(f"[WARN] Bad payload: {msg.payload}")


# ── MAIN ────────────────────────────────────────────────
if __name__ == "__main__":
    print("═" * 50)
    print("  EyeGuard — Pi Alert Listener")
    print("═" * 50)
    print()

    # Start LED control in background thread
    led_thread = threading.Thread(target=led_control_loop, daemon=True)
    led_thread.start()
    print("[LED] Background control thread started")

    # Start MQTT listener
    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_message = on_message

    print("[MQTT] Connecting to local broker...")
    try:
        client.connect("localhost", 1883, 60)
        client.loop_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
    except Exception as e:
        print(f"[MQTT] Connection failed: {e}")
        print("Make sure Mosquitto is running: sudo systemctl status mosquitto")
    finally:
        if GPIO_AVAILABLE:
            GPIO.output(GPIO_BUZZER, GPIO.LOW)
            GPIO.output(GPIO_RED_LED, GPIO.LOW)
            GPIO.output(GPIO_GREEN_LED, GPIO.LOW)
            GPIO.cleanup()
            print("[GPIO] Cleaned up")
        print("Listener stopped.")