"""
EyeGuard — GPIO Hardware Test
Run this on the Raspberry Pi to verify LEDs and buzzer are wired correctly.

Usage:
    python test_gpio.py

No camera, no display, no X11 needed — just terminal output and hardware.
"""

import time

try:
    import RPi.GPIO as GPIO
except ImportError:
    print("ERROR: RPi.GPIO not available. This script must run on the Raspberry Pi.")
    exit(1)

# Pin assignments (from EyeGuard spec section 5.2)
GPIO_RED_LED = 17      # Pin 11
GPIO_GREEN_LED = 22    # Pin 15
GPIO_BUZZER = 27       # Pin 13

def setup():
    GPIO.setmode(GPIO.BCM)
    GPIO.setwarnings(False)
    GPIO.setup(GPIO_RED_LED, GPIO.OUT)
    GPIO.setup(GPIO_GREEN_LED, GPIO.OUT)
    GPIO.setup(GPIO_BUZZER, GPIO.OUT)

    # Start with everything off
    GPIO.output(GPIO_RED_LED, GPIO.LOW)
    GPIO.output(GPIO_GREEN_LED, GPIO.LOW)
    GPIO.output(GPIO_BUZZER, GPIO.LOW)

def test_red_led():
    print("\n[TEST 1] Red LED (GPIO 17, Pin 11)")
    print("  Turning ON... ", end="", flush=True)
    GPIO.output(GPIO_RED_LED, GPIO.HIGH)
    time.sleep(2)
    print("ON for 2 seconds")
    GPIO.output(GPIO_RED_LED, GPIO.LOW)
    print("  Turned OFF")

    answer = input("  Did the RED LED light up? (y/n): ").strip().lower()
    return answer == "y"

def test_green_led():
    print("\n[TEST 2] Green LED (GPIO 22, Pin 15)")
    print("  Turning ON... ", end="", flush=True)
    GPIO.output(GPIO_GREEN_LED, GPIO.HIGH)
    time.sleep(2)
    print("ON for 2 seconds")
    GPIO.output(GPIO_GREEN_LED, GPIO.LOW)
    print("  Turned OFF")

    answer = input("  Did the GREEN LED light up? (y/n): ").strip().lower()
    return answer == "y"

def test_buzzer():
    print("\n[TEST 3] Buzzer (GPIO 27, Pin 13)")
    print("  Turning ON... ", end="", flush=True)
    GPIO.output(GPIO_BUZZER, GPIO.HIGH)
    time.sleep(1)
    print("ON for 1 second")
    GPIO.output(GPIO_BUZZER, GPIO.LOW)
    print("  Turned OFF")

    answer = input("  Did you hear the buzzer? (y/n): ").strip().lower()
    return answer == "y"

def test_amber():
    print("\n[TEST 4] Amber (Red + Green together)")
    print("  Turning both ON... ", end="", flush=True)
    GPIO.output(GPIO_RED_LED, GPIO.HIGH)
    GPIO.output(GPIO_GREEN_LED, GPIO.HIGH)
    time.sleep(2)
    print("ON for 2 seconds")
    GPIO.output(GPIO_RED_LED, GPIO.LOW)
    GPIO.output(GPIO_GREEN_LED, GPIO.LOW)
    print("  Turned OFF")

    answer = input("  Did both LEDs light up together? (y/n): ").strip().lower()
    return answer == "y"

def test_blink():
    print("\n[TEST 5] Blink pattern (simulates alert tiers)")
    print("  Green steady (NORMAL tier)... ", flush=True)
    GPIO.output(GPIO_GREEN_LED, GPIO.HIGH)
    time.sleep(2)
    GPIO.output(GPIO_GREEN_LED, GPIO.LOW)

    print("  Amber slow blink (WARNING tier)... ", flush=True)
    for _ in range(4):
        GPIO.output(GPIO_RED_LED, GPIO.HIGH)
        GPIO.output(GPIO_GREEN_LED, GPIO.HIGH)
        time.sleep(0.5)
        GPIO.output(GPIO_RED_LED, GPIO.LOW)
        GPIO.output(GPIO_GREEN_LED, GPIO.LOW)
        time.sleep(0.5)

    print("  Red fast flash + buzzer (CRITICAL tier)... ", flush=True)
    for _ in range(6):
        GPIO.output(GPIO_RED_LED, GPIO.HIGH)
        GPIO.output(GPIO_BUZZER, GPIO.HIGH)
        time.sleep(0.2)
        GPIO.output(GPIO_RED_LED, GPIO.LOW)
        GPIO.output(GPIO_BUZZER, GPIO.LOW)
        time.sleep(0.2)

    print("  Done")
    answer = input("  Did the blink patterns look correct? (y/n): ").strip().lower()
    return answer == "y"

def test_all_off():
    print("\n[TEST 6] All OFF")
    GPIO.output(GPIO_RED_LED, GPIO.LOW)
    GPIO.output(GPIO_GREEN_LED, GPIO.LOW)
    GPIO.output(GPIO_BUZZER, GPIO.LOW)
    print("  All components should be OFF now")

    answer = input("  Is everything off? (y/n): ").strip().lower()
    return answer == "y"

def main():
    print("=" * 50)
    print("  EyeGuard — GPIO Hardware Test")
    print("=" * 50)
    print("\nThis will test each component one by one.")
    print("Watch the breadboard and listen for the buzzer.\n")

    setup()

    results = {}
    results["Red LED"] = test_red_led()
    results["Green LED"] = test_green_led()
    results["Buzzer"] = test_buzzer()
    results["Amber (both)"] = test_amber()
    results["Blink pattern"] = test_blink()
    results["All OFF"] = test_all_off()

    # Summary
    print("\n" + "=" * 50)
    print("  TEST RESULTS")
    print("=" * 50)

    all_passed = True
    for component, passed in results.items():
        status = "PASS" if passed else "FAIL"
        icon = "[✓]" if passed else "[✗]"
        print(f"  {icon} {component}: {status}")
        if not passed:
            all_passed = False

    if all_passed:
        print("\nAll tests passed! Hardware is ready for EyeGuard.")
    else:
        print("\nSome tests failed. Check wiring for failed components:")
        print("  - LED not lighting? Check polarity (long leg = +)")
        print("  - LED still not working? Check resistor connection")
        print("  - Buzzer silent? Check + and - orientation")
        print("  - Nothing works? Check GND rail connection to Pi Pin 9")

    GPIO.cleanup()
    print("\n[GPIO] Cleaned up. Done.")

if __name__ == "__main__":
    main()