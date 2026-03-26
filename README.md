# EyeGuard — Camera-Based Motion and Eye Detection System

**COMP 494: Special Topics in Computer Science**
**Egerton University — Department of Computer Science**

An IoT system that uses a laptop webcam to detect motion and monitor eye
state (open/closed) in real time. When motion or drowsiness is detected,
the system displays visual alerts on screen and controls GPIO hardware
on a Raspberry Pi 3 — a tiered LED system (green/amber/red) and an
audible buzzer alarm.

---

## Table of Contents

1. [How the System Works](#how-the-system-works)
2. [How the Files Relate to Each Other](#how-the-files-relate-to-each-other)
3. [Team Assignments](#team-assignments)
4. [Prerequisites](#prerequisites)
5. [Installation & Setup](#installation--setup)
6. [Branching Strategy](#branching-strategy)
7. [Git Workflow](#git-workflow)
8. [How to Run the Program](#how-to-run-the-program)
9. [File Structure](#file-structure)
10. [Troubleshooting](#troubleshooting)

---

## How the System Works

### The Big Picture

The system uses your laptop's webcam to watch for two things:

1. **Motion** — Is something moving in front of the camera?
2. **Drowsiness** — Are the person's eyes closed for too long?

When either is detected, two things happen:
- A **visual alert** appears on your laptop screen (colored borders and warning text)
- An **MQTT message** is sent over Wi-Fi to a Raspberry Pi, which triggers a **physical buzzer**

### The Frame-by-Frame Pipeline

The system processes video in real time at approximately 30 frames per second.
Every single frame (captured roughly every 33 milliseconds) flows through
five stages in sequence:

```
 STAGE 1          STAGE 2           STAGE 3          STAGE 4          STAGE 5
┌────────┐     ┌────────────┐    ┌────────────┐   ┌────────────┐   ┌────────────┐
│ Camera │     │  Motion    │    │  Face &    │   │   EAR      │   │  Alert     │
│ Feed   │────▶│  Detection │    │  Eye       │──▶│  Calc &    │──▶│  Triggers  │
│        │     │            │    │  Detection │   │  Drowsy    │   │            │
│Silvana │     │ Mel + Stan │    │   Deno     │   │   Timo     │   │  Ritchie   │
└────────┘     └─────┬──────┘    └────────────┘   └────────────┘   └─────┬──────┘
                     │                                                    │
                     └───────────────── both feed into ──────────────────┘
```

Here is what happens at each stage:

**Stage 1 — Camera Feed (Silvana):**
The webcam captures a single frame — a snapshot of what the camera sees right now.
This raw image (a grid of pixels, 640 wide × 480 tall) is passed to every other
module. Think of it as taking a photograph 30 times per second.

**Stage 2 — Motion Detection (Melanie & Stan):**
The current frame is compared to the previous frame. If significant pixels changed
between the two frames, something moved. The algorithm converts frames to grayscale,
blurs them to reduce noise, subtracts one from the other, and looks for large
regions of change. The result is a simple yes/no answer ("motion detected") plus
the locations of moving objects (as bounding boxes).

**Stage 3 — Face & Eye Detection (Deno):**
A pre-trained machine learning model (dlib) scans the frame for a human face.
IMPORTANT: Per the EyeGuard spec (section 4.4), this stage only runs when
motion is detected in Stage 2. This gating conserves the Raspberry Pi 3's
limited processing resources by skipping the expensive dlib computation when
nobody is moving. If a face is found, a second model identifies 68 specific
facial landmarks — points along the jawline, eyebrows, nose, eyes, and mouth.
From these 68 points, the 6 left-eye landmarks (points 42-47) and 6 right-eye
landmarks (points 36-41) are extracted and passed to the next stage.

**Stage 4 — EAR Calculation & Drowsiness Logic (Timo):**
The Eye Aspect Ratio (EAR) is calculated from the eye landmarks. EAR measures
how "open" the eye is by comparing vertical distances (eye height) to horizontal
distance (eye width). When eyes are open, EAR is approximately 0.30 or above.
When closed, EAR drops below 0.25 (the threshold defined in EyeGuard spec
section 4.5). A single drop could just be a blink (1-3 frames), so the system
only flags drowsiness when eyes stay closed for 20 or more consecutive frames
(about 0.67 seconds at 30fps). This maps to the CRITICAL alert tier.

**Stage 5 — Alert Triggers (Ritchie):**
The motion result and drowsiness result feed into a three-tier alert system
defined in the EyeGuard spec:

| Tier | Condition | LED State | Buzzer |
|------|-----------|-----------|--------|
| NORMAL | Eyes open (EAR >= 0.25) | Green steady ON | Silent |
| WARNING | Eyes closing (EAR < 0.25, < 20 frames) | Amber slow blink | Short beep every 2s |
| CRITICAL | Drowsiness confirmed (EAR < 0.25, 20+ frames) | Red fast flash | Continuous alarm |

On a laptop, the LEDs are simulated as colored circles on the camera feed.
On the Pi, real GPIO hardware is controlled: Red LED on GPIO 17, Green LED
on GPIO 22, Buzzer on GPIO 27. MQTT messages are also sent to the Pi for
remote monitoring. A cooldown timer prevents the buzzer from being triggered
more than once every 2 seconds.

### How the Laptop Talks to the Raspberry Pi

The laptop and Pi communicate using **MQTT** (Message Queuing Telemetry Transport),
a lightweight messaging protocol designed for IoT devices:

```
  LAPTOP                          RASPBERRY PI 3
┌──────────────────┐           ┌──────────────────────┐
│                  │           │                      │
│  alerts.py       │  Wi-Fi   │  Mosquitto Broker    │
│  (Ritchie)       │─────────▶│  (MQTT Server)       │
│                  │           │       │              │
│  Publishes to:   │           │       ▼              │
│  alerts/motion   │           │  buzzer_listener.py  │
│  alerts/drowsiness│          │       │              │
│                  │           │       ▼              │
└──────────────────┘           │  GPIO Pin 17        │
                               │  (Physical Buzzer)   │
                               └──────────────────────┘
```

The Pi runs a program called Mosquitto, which acts as a message broker (like a post
office). The laptop publishes messages to specific "topics" (like mailing addresses).
The Pi subscribes to those topics and reacts when a message arrives.

---

## How the Files Relate to Each Other

### Dependency Map

This shows which modules depend on which. An arrow means "this module uses
output from that module":

```
camera.py (Silvana)
    │
    │ provides raw frames to:
    │
    ├──▶ motion.py (Melanie & Stan)
    │        │
    │        │ produces MotionResult
    │        │ (detected: bool, bounding_boxes: list)
    │        │
    │        └──────────────────────┐
    │                               │
    ├──▶ face_eye.py (Deno)         │
    │        │                      │
    │        │ produces FaceEyeResult│
    │        │ (left_eye: 6 points, │
    │        │  right_eye: 6 points)│
    │        │                      │
    │        ▼                      │
    │   ear_logic.py (Timo)         │
    │        │                      │
    │        │ produces EARResult   │
    │        │ (ear_value: float,   │
    │        │  drowsy: bool)       │
    │        │                      │
    │        └──────────┐           │
    │                   │           │
    │                   ▼           ▼
    │              alerts.py (Ritchie)
    │                   │
    │                   │ draws on frame + sends MQTT
    │                   │
    │                   ▼
    │              Display window
    │              (cv2.imshow)
    │
    └──▶ main.py (Everyone)
         Orchestrates the entire pipeline.
         Calls each module in order every frame.
```

### What Each File Produces and Consumes

| File | Consumes (input) | Produces (output) | Used by |
|------|-------------------|-------------------|---------|
| `camera.py` | Webcam hardware | Raw BGR frame (numpy array) | All other modules |
| `motion.py` | Raw frame | `MotionResult` (detected, bounding_boxes) | `alerts.py` |
| `face_eye.py` | Raw frame | `FaceEyeResult` (left_eye, right_eye coords) | `ear_logic.py` |
| `ear_logic.py` | Eye coordinates | `EARResult` (ear_value, drowsy flag) | `alerts.py` |
| `alerts.py` | MotionResult + EARResult | Visual overlays on frame + MQTT messages | Display + Pi |
| `main.py` | All of the above | Wires everything together | Entry point |

### Data Flow Example (Single Frame)

Here is a concrete example of what happens to one frame:

```
1. camera.py captures a 640×480 pixel BGR image (a numpy array of shape 640×480×3)

2. motion.py receives the frame:
   - Converts to grayscale (640×480×1)
   - Blurs it (still 640×480×1)
   - Compares to previous frame → binary mask of changes
   - Finds 2 large contours → boxes at (120,80,200,150) and (400,200,100,80)
   - Returns: MotionResult(detected=True, bounding_boxes=[(120,80,200,150), (400,200,100,80)])

3. face_eye.py receives the same original frame:
   - Converts to grayscale
   - dlib finds a face at rectangle (180, 100, 420, 380)
   - Shape predictor finds 68 landmarks
   - Extracts left eye: [(290,180), (295,175), (310,175), (315,180), (310,185), (295,185)]
   - Extracts right eye: [(230,180), (235,175), (250,175), (255,180), (250,185), (235,185)]
   - Returns: FaceEyeResult(face_detected=True, left_eye=[...], right_eye=[...])

4. ear_logic.py receives the eye coordinates:
   - Computes left EAR = 0.31 (eye is open)
   - Computes right EAR = 0.29 (eye is open)
   - Average EAR = 0.30
   - 0.30 > 0.20 threshold → eyes_closed = False
   - Resets consecutive counter to 0
   - Returns: EARResult(ear_value=0.30, eyes_closed=False, drowsy=False)

5. alerts.py receives MotionResult and EARResult:
   - Motion detected → draws red border, publishes to "alerts/motion"
   - Not drowsy → no orange border
   - Draws status bar: "MOTION"

6. main.py displays the annotated frame with:
   - Green boxes around the two moving regions (from motion.py)
   - Yellow dots on eye landmarks (from face_eye.py)
   - Red border and "MOTION DETECTED" text (from alerts.py)
   - Status bar showing "MOTION" (from alerts.py)
```

---

## Team Assignments

### Silvana — Live Camera Feed (`src/camera.py`)

**Branch:** `silvana`

**What you're building:**
The CameraStream class that opens the laptop's webcam and provides frames
to the rest of the system. You're the foundation — everyone else depends
on your module working first.

**Your deliverables:**
- `CameraStream.__init__()` — Open the camera, set resolution, handle errors
- `CameraStream.read_frame()` — Capture and return a single frame
- `CameraStream.get_frame_dimensions()` — Return frame width and height
- `CameraStream.release()` — Clean up camera resources
- Standalone test that shows a live webcam feed

**Key technologies:**
- `cv2.VideoCapture` for camera access
- `cv2.imshow` / `cv2.waitKey` for display

**Definition of done:**
Running `python src/camera.py` opens a window showing your live webcam.
Pressing 'q' closes it cleanly with no errors.

**Estimated effort:** Lightest workload. Aim to finish first so others
can build on top of your module.

---

### Melanie & Stan — Motion Detection (`src/motion.py`)

**Branches:** `melanie` (preprocessing & math) and `stan` (contours & drawing)

**What you're building:**
A motion detector that compares consecutive camera frames to identify
movement. It uses a technique called "frame differencing" — if pixels
changed between frame N and frame N+1, something moved there.

**Task split:**

**Melanie** handles the preprocessing and math:
- `_preprocess()` — Convert frame to grayscale and apply Gaussian blur.
  The blur smooths out camera sensor noise that would cause false motion
  detections.
- `_compute_delta()` — Compare the current frame with the previous one
  using absolute difference, then threshold to create a binary mask where
  white pixels = movement.

**Stan** handles the detection logic and visuals:
- `_find_motion_regions()` — Find contours (outlines) in the binary mask
  and filter out small ones (noise). Return bounding boxes around real
  movement.
- `detect()` — Wire the full pipeline: preprocess → delta → find regions.
  Return a MotionResult.
- `draw_on_frame()` — Draw green rectangles around detected motion and
  add "MOTION DETECTED" text.

**How Melanie and Stan coordinate:**
Since you both work on the same file (`motion.py`), follow this workflow:
1. Melanie completes `_preprocess()` and `_compute_delta()` on the `melanie` branch
2. Melanie opens a PR to `dev` and merges
3. Stan pulls the latest `dev` into his `stan` branch: `git checkout stan && git merge dev`
4. Stan builds on top of Melanie's work, implementing `_find_motion_regions()`, `detect()`, and `draw_on_frame()`
5. Stan opens a PR to `dev`

**The algorithm in plain English:**
1. Convert the color frame to grayscale (faster to process)
2. Blur it slightly to remove sensor noise
3. Subtract the previous frame from the current frame
4. Any pixel that changed significantly (> 25 brightness) becomes white
5. Find the white blobs and draw boxes around the big ones
6. Small blobs = noise, big blobs = real motion

**Key technologies:**
- `cv2.cvtColor` for grayscale conversion
- `cv2.GaussianBlur` for noise reduction
- `cv2.absdiff` for frame differencing
- `cv2.threshold` for binary mask creation
- `cv2.findContours` + `cv2.contourArea` for motion regions

**Definition of done:**
Running `python src/motion.py` opens a webcam feed. When you wave your
hand, green boxes appear around the movement and "MOTION DETECTED" shows
at the top. When you're still, the boxes and text disappear.

**Estimated effort:** Medium. The algorithm has several steps but each
is well-documented in the code.

---

### Deno — Face & Eye Detection (`src/face_eye.py`)

**Branch:** `deno`

**What you're building:**
A face detector that locates a person's face in the frame and extracts
the exact pixel coordinates of their eye landmarks. You'll use dlib's
pre-trained models — a HOG-based face detector and a 68-point facial
landmark predictor.

**Your deliverables:**
- `FaceEyeDetector.__init__()` — Load dlib's detector and shape predictor
- `_detect_face()` — Find faces in a grayscale frame
- `_extract_landmarks()` — Get 68 facial landmark points for a face
- `_extract_eyes()` — Slice out the 6 left eye and 6 right eye points
- `detect()` — Full pipeline returning FaceEyeResult
- `draw_on_frame()` — Draw eye landmarks and convex hulls on the frame

**Critical setup step:**
Before you can code anything, download the shape predictor model:
```
wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
bunzip2 shape_predictor_68_face_landmarks.dat.bz2
```
This is a ~100MB machine learning model. Without this file, nothing works.
Don't commit it to git (it's in .gitignore).

**How dlib's 68-point model maps to eyes:**
```
Points 36-41 = Right eye (6 landmarks)
Points 42-47 = Left eye (6 landmarks)

Each eye's 6 points:
       p1          p2
        ●──────────●
       / \        / \
  p0 ●    \      /    ● p3
       \   \    /   /
        ●──────────●
       p5          p4
```

**Key technologies:**
- `dlib.get_frontal_face_detector()` for face detection
- `dlib.shape_predictor()` for landmark extraction
- `imutils.face_utils.shape_to_np()` to convert landmarks to numpy

**Definition of done:**
Running `python src/face_eye.py` opens a webcam feed. When your face is
visible, yellow dots appear on your eye landmarks. When you turn away or
cover your face, "No face detected" appears instead.

**Estimated effort:** Medium-heavy. dlib installation can be tricky, and
you need to understand the landmark coordinate system. The code itself
is straightforward once setup is done.

---

### Timo — EAR Calculation & Drowsiness Logic (`src/ear_logic.py`)

**Branch:** `timo`

**What you're building:**
The Eye Aspect Ratio (EAR) calculator — the mathematical brain of the
drowsiness detection. You take the 6 eye landmark points from Deno's
module and compute a single number that tells you if the eye is open
or closed. If both eyes stay closed for too many frames, you flag it
as drowsiness.

**Your deliverables:**
- `EARCalculator.__init__()` — Store threshold, initialize counter
- `_compute_ear()` — Calculate EAR for a single eye (the core formula)
- `calculate()` — Average both eyes, track consecutive closed frames
- `reset()` — Reset the frame counter
- Standalone test with hardcoded data (no camera needed!)

**The EAR formula:**
```
         ||p1 - p5|| + ||p2 - p4||
  EAR = ───────────────────────────
              2 × ||p0 - p3||

  Open eye:   EAR ≈ 0.25 - 0.35 (tall vertical distances)
  Closed eye: EAR ≈ 0.05 - 0.15 (tiny vertical distances)
  Threshold:  0.20 (below this = closed)
```

**Drowsiness vs blinks:**
- A blink: EAR drops for 1-3 frames (~100ms), then recovers
- Drowsiness: EAR stays low for 15+ frames (~500ms)
- Your consecutive frame counter is what distinguishes them

**Key technologies:**
- `scipy.spatial.distance.euclidean` for distance calculation
- Basic Python: counters, comparisons, dataclasses

**Definition of done:**
Running `python src/ear_logic.py` passes all self-tests: open eyes
produce EAR > 0.2, closed eyes produce EAR < 0.2, and the drowsiness
flag triggers after the correct number of consecutive closed frames.

**Estimated effort:** Lightest technical effort, but requires understanding
the math. Your module is the most self-contained — you can complete and
test it without any other module being ready.

---

### Ritchie — Alert Triggers (`src/alerts.py`)

**Branch:** `ritchie`

**What you're building:**
The alert system that turns detection results into things humans can
perceive — visual overlays on the camera feed and buzzer triggers on
the Raspberry Pi via MQTT.

**Your deliverables:**
- `AlertManager.__init__()` — Set up state and connect to MQTT
- `_connect_mqtt()` — Connect to the Pi's Mosquitto broker (gracefully handle failure)
- `_publish()` — Send JSON messages with cooldown enforcement
- `check_motion()` — Process motion results, update state, publish if needed
- `check_drowsiness()` — Process EAR results, update state, publish if needed
- `draw_alerts()` — Render colored borders, warning text, and status bar
- `cleanup()` — Disconnect MQTT

**Visual alert design:**
```
Motion detected:     RED border (8px) + "MOTION DETECTED" text
Drowsiness detected: ORANGE border (8px) + "DROWSINESS ALERT!" text
Both detected:       Both borders visible (orange inset 4px from red)
Neither:             "ALL CLEAR" in status bar
```

**MQTT topics (must match the Pi's buzzer_listener.py):**
- `alerts/motion` → short buzzer burst (0.3s)
- `alerts/drowsiness` → long buzzer burst (1.0s)

**Key technologies:**
- `paho-mqtt` for MQTT publishing
- `cv2.rectangle` and `cv2.putText` for visual overlays
- `time.time()` for cooldown management
- `json.dumps` for message serialization

**Definition of done:**
Running `python src/alerts.py` opens a webcam feed with simulated
alternating alert states. Red and orange borders flash on and off
correctly. If the Pi is running Mosquitto, MQTT messages appear in
the Pi's terminal. If the Pi is offline, the script still runs with
visual alerts only.

**Estimated effort:** Medium. MQTT setup requires coordination with the
Pi, but the visual alert code is straightforward.

---

## Prerequisites

### Hardware Requirements

| Item | Purpose | Notes |
|------|---------|-------|
| Laptop with webcam | Runs the detection software | Any modern laptop will work |
| Raspberry Pi 3 | Receives alerts, triggers buzzer | Must have Wi-Fi (Pi 3 has built-in) |
| MicroSD card (16GB+) | Stores Pi's operating system | 32GB recommended |
| Micro-USB power supply | Powers the Pi 3 | 5V / 2.5A |
| Buzzer (optional) | Physical alert output | Active buzzer on GPIO pin 17 |
| Jumper wires (optional) | Connect buzzer to Pi GPIO | Female-to-female for direct connection |

### Software Requirements

**On your laptop:**

| Software | Version | Purpose | How to check |
|----------|---------|---------|-------------|
| Python | 3.8 or higher | Runs all detection code | `python3 --version` |
| pip | Any recent version | Installs Python packages | `pip --version` |
| git | Any recent version | Version control | `git --version` |
| cmake | Latest | Required to build dlib | `cmake --version` |

**On the Raspberry Pi:**

| Software | Version | Purpose |
|----------|---------|---------|
| Raspberry Pi OS Lite | Latest (64-bit) | Operating system |
| Python 3 | Pre-installed with OS | Runs buzzer listener |
| Mosquitto | Latest | MQTT message broker |

### Python Package Dependencies

These are listed in `requirements.txt` and installed automatically:

| Package | Version | Purpose | Used by |
|---------|---------|---------|---------|
| `opencv-python` | 4.9.0.80 | Camera access, image processing, drawing | All modules |
| `dlib` | 19.24.2 | Face detection, facial landmarks | Deno |
| `numpy` | 1.26.4 | Array operations for image data | All modules |
| `imutils` | 0.5.4 | Helper utilities for dlib landmarks | Deno |
| `scipy` | 1.12.0 | Euclidean distance calculation | Timo |
| `paho-mqtt` | 2.0.0 | MQTT client for Pi communication | Ritchie |
| `cmake` | 3.28.1 | Build tool required by dlib | Build dependency |

### Network Requirements

Your laptop and the Raspberry Pi must be on the **same Wi-Fi network**.
The laptop communicates with the Pi over MQTT on port 1883. Make sure
your network doesn't block device-to-device communication (most home
networks allow this; some corporate/university networks don't).

---

## Installation & Setup

### Step 1: Clone the Repository

```bash
git clone <repo-url>
cd motion-eye-detector
```

### Step 2: Create a Python Virtual Environment

A virtual environment keeps this project's packages separate from your
system Python. Always activate it before working on the project.

```bash
# Create the virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate        # Linux / Mac
# venv\Scripts\activate         # Windows

# You should see (venv) at the start of your terminal prompt
```

### Step 3: Install Python Dependencies

```bash
pip install -r requirements.txt
```

If `dlib` fails to install, install cmake separately first:

```bash
# Linux
sudo apt install build-essential cmake

# Mac
brew install cmake

# Then retry
pip install dlib
```

### Step 4: Download the Face Landmark Model

This is a ~100MB pre-trained model required by Deno's module:

```bash
wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
bunzip2 shape_predictor_68_face_landmarks.dat.bz2
```

If you don't have `wget`, use `curl` instead:

```bash
curl -O http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
bunzip2 shape_predictor_68_face_landmarks.dat.bz2
```

The resulting `.dat` file should be in the project root directory.

### Step 5: Set Up the Raspberry Pi

SSH into your Pi from your laptop:

```bash
ssh <your-username>@<pi-ip-address>
```

Install and configure the MQTT broker:

```bash
# Install Mosquitto
sudo apt update
sudo apt install mosquitto mosquitto-clients -y

# Configure it to accept external connections
sudo nano /etc/mosquitto/mosquitto.conf

# Add these two lines at the BOTTOM of the file:
#   listener 1883
#   allow_anonymous true

# Save: Ctrl+X, then Y, then Enter

# Restart the service
sudo systemctl restart mosquitto
sudo systemctl enable mosquitto   # Start on boot
```

Install Python MQTT library and copy the buzzer listener:

```bash
pip3 install paho-mqtt

# Copy pi_buzzer_listener.py to the Pi (from your laptop):
# scp pi_buzzer_listener.py <username>@<pi-ip>:~/
```

### Step 6: Update the Pi's IP Address in the Code

Open `src/main.py` and `src/alerts.py` and replace `"192.168.x.x"` with
your Pi's actual IP address. Find it by running on the Pi:

```bash
hostname -I
```

Or from your laptop:

```bash
ping raspberrypi.local
```

---

## Branching Strategy

This project uses a strict branching model with three tiers to keep
code organized and prevent anyone from accidentally breaking the working
system.

### Branch Structure

```
main ← Production-ready code. ONLY Cecil can merge into this.
 │
 └── dev ← Integration branch. All PRs go here.
      │
      ├── silvana    ← Silvana's personal branch (camera feed)
      ├── melanie    ← Melanie's personal branch (motion preprocessing)
      ├── stan       ← Stan's personal branch (motion contours)
      ├── deno       ← Deno's personal branch (face & eye detection)
      ├── timo       ← Timo's personal branch (EAR calculation)
      └── ritchie    ← Ritchie's personal branch (alert triggers)
```

### The Three Rules

**Rule 1: You ONLY commit to YOUR own branch.**

Every team member has a branch named after them. You do all your work
there. You never commit to someone else's branch, to `dev`, or to `main`.

```
  ✓  git checkout silvana    →  git commit -m "Add camera init"
  ✗  git checkout dev        →  git commit    (NEVER)
  ✗  git checkout main       →  git commit    (NEVER)
  ✗  git checkout deno       →  git commit    (NEVER, unless you are Deno)
```

**Rule 2: Pull Requests go to `dev` ONLY. Never to `main`.**

When your work is ready, you open a Pull Request from your branch into
`dev`. Another team member reviews it. Once approved, it gets merged
into `dev`. This is how everyone's code comes together.

```
  ✓  silvana  → PR → dev      (correct)
  ✓  timo     → PR → dev      (correct)
  ✗  silvana  → PR → main     (WRONG — never PR to main)
  ✗  ritchie  → PR → main     (WRONG — never PR to main)
```

**Rule 3: Only Cecil merges `dev` into `main`.**

Once features have been merged into `dev` and tested together, Cecil
(and only Cecil) merges `dev` into `main`. This ensures `main` always
contains stable, working code. No one else touches `main`.

```
  ✓  Cecil:   git checkout main → git merge dev → git push
  ✗  Anyone else: git checkout main → (STOP — you don't do this)
```

### Why This Strategy?

```
                    YOUR BRANCH          DEV              MAIN
                    (your work)     (team integration)  (stable release)
                         │                 │                │
  You write code ───────▶●                 │                │
  You commit ───────────▶●                 │                │
  You push ─────────────▶●                 │                │
                         │                 │                │
  You open a PR ─────────┼────── PR ──────▶│                │
  Team reviews ──────────┼────────────────▶●                │
  PR merged ─────────────┼────────────────▶●                │
                         │                 │                │
  Others merge too ──────┼────────────────▶●                │
  Team tests together ───┼────────────────▶●                │
                         │                 │                │
  Cecil merges to main ──┼─────────────────┼────── merge ──▶●
                         │                 │                │
```

This protects `main` from broken code. If something goes wrong in `dev`,
the team fixes it there without affecting `main`. The `dev` branch is
the "testing ground" where everyone's code is combined before it's
promoted to `main`.

### Keeping Your Branch Up to Date

As other people's PRs get merged into `dev`, you should periodically
pull those changes into your own branch so you're building on the latest
code:

```bash
# Switch to your branch
git checkout <your-name>

# Pull latest changes from dev into your branch
git merge dev

# If there are merge conflicts, resolve them, then:
git add .
git commit -m "Merge dev into <your-name>"
```

Do this regularly (at least once a day when the team is active) to avoid
painful merge conflicts later.

### Branch Protection (for the repo admin / Cecil)

If your repository host supports it (GitHub, GitLab, Bitbucket), Cecil
should set up these protections:

**On `main`:**
- Require PR to merge (no direct pushes)
- Only Cecil can approve and merge PRs
- Require all CI checks to pass (if you set up CI later)

**On `dev`:**
- Require PR to merge (no direct pushes)
- Require at least 1 reviewer approval
- Allow any team member to approve

This makes the rules enforced by the platform, not just by trust.

---

## Git Workflow

### First Time Setup (Every Team Member)

```bash
# Clone the repo
git clone <repo-url>
cd motion-eye-detector

# Switch to your personal branch
git checkout <your-name>
# Example: git checkout silvana

# Set up virtual environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Daily Workflow

```bash
# 1. Make sure you're on YOUR branch
git checkout <your-name>

# 2. Pull the latest from dev to stay current
git merge dev

# 3. Do your work — edit your file(s)

# 4. Check what changed
git status

# 5. Stage your changes
git add src/<your-file>.py

# 6. Commit with a descriptive message
git commit -m "Implement _preprocess method with grayscale and blur"

# 7. Push to remote
git push origin <your-name>
```

### Submitting Your Work (Pull Request)

When a feature or fix is ready:

1. Push all your commits: `git push origin <your-name>`
2. Go to your repository on GitHub/GitLab
3. Click "New Pull Request" or "Create Merge Request"
4. Set the **source** to your branch (e.g., `silvana`)
5. Set the **target** to `dev` (NOT `main`)
6. Write a description of what you did and what to test
7. Request a review from at least one teammate
8. After approval, merge the PR

### Cecil's Workflow (Merging dev → main)

Once features are merged into `dev` and the team confirms they work:

```bash
# Switch to main
git checkout main

# Pull latest main (in case of remote changes)
git pull origin main

# Merge dev into main
git merge dev

# Push the updated main
git push origin main
```

Cecil should only do this when `dev` is in a stable, tested state.

### Recommended PR Merge Order

Since modules have dependencies, merge PRs into `dev` in this order:

```
1. silvana  (camera)      — no dependencies
2. timo     (EAR logic)   — no dependencies
3. melanie  (motion math) — depends on camera
4. stan     (motion draw) — depends on melanie's work
5. deno     (face/eye)    — depends on camera
6. ritchie  (alerts)      — depends on motion + EAR
```

After all are merged into `dev` and tested, Cecil merges `dev` → `main`.

---

## How to Run the Program

### Option A: Test Individual Modules (During Development)

Each module can run independently for testing. You don't need the full
system working to test your own part.

```bash
# Make sure you're in the project directory with venv activated
cd motion-eye-detector
source venv/bin/activate

# Test each module:
python src/camera.py        # Silvana: should show live webcam feed
python src/motion.py        # Melanie & Stan: should show motion boxes
python src/face_eye.py      # Deno: should show eye landmark dots
python src/ear_logic.py     # Timo: should print math test results
python src/alerts.py        # Ritchie: should show flashing alert borders
```

Each module opens a camera window (except Timo's, which is math-only).
Press **q** to close the window and exit.

### Option B: Run the Full Integrated System

This requires all modules to be complete and merged into `dev` (or `main`).

**Terminal 1 — Start the buzzer listener on the Pi (via SSH):**

```bash
ssh <username>@<pi-ip-address>
python3 pi_buzzer_listener.py
```

You should see:
```
Pi Buzzer Listener starting...
[MQTT] Connected (code 0)
[MQTT] Subscribed to alerts/#
```

**Terminal 2 — Start the detection system on your laptop:**

```bash
cd motion-eye-detector
source venv/bin/activate
python src/main.py
```

You should see:
```
Initializing modules...
  [✓] Camera initialized
  [✓] Motion detector initialized
  [✓] Face/eye detector initialized
  [✓] EAR calculator initialized
  [✓] Alert manager initialized

All modules ready. Press 'q' to quit.
```

A camera window will open showing your webcam feed with real-time
detection overlays.

### What You'll See During Operation

| Scenario | On Laptop Screen | On Pi Terminal |
|----------|-----------------|----------------|
| No motion, eyes open | Clean feed, "ALL CLEAR" status | No output |
| Motion detected | Red border, "MOTION DETECTED", green boxes | `[HH:MM:SS] MOTION detected!` + buzzer |
| Eyes closed briefly (blink) | Yellow eye dots briefly dim | No output (too brief) |
| Eyes closed >0.5s (drowsy) | Orange border, "DROWSINESS ALERT!" | `[HH:MM:SS] EYES CLOSED for N frames!` + long buzzer |
| Both motion + drowsy | Red + orange borders, both texts | Both alerts printed |
| No face visible | "No face detected" text | No drowsiness alerts |

### Keyboard Controls

| Key | Action |
|-----|--------|
| `q` | Quit the application cleanly |

---

## File Structure

```
motion-eye-detector/
│
├── src/                          # All source code lives here
│   ├── __init__.py               # Makes src a Python package
│   ├── camera.py                 # Silvana  — webcam capture & display
│   ├── motion.py                 # Mel+Stan — frame differencing & contours
│   ├── face_eye.py               # Deno     — dlib face detection & eye landmarks
│   ├── ear_logic.py              # Timo     — EAR formula & drowsiness state
│   ├── alerts.py                 # Ritchie  — visual overlays & MQTT publishing
│   └── main.py                   # Everyone — integration pipeline
│
├── tests/                        # Unit tests (encouraged but optional)
├── docs/                         # Additional documentation or diagrams
│
├── pi_buzzer_listener.py         # Runs on the Raspberry Pi (not in src/)
├── shape_predictor_68_face_landmarks.dat  # dlib model (downloaded, NOT in git)
│
├── requirements.txt              # Python package dependencies
├── .gitignore                    # Files excluded from version control
└── README.md                     # This file
```

---

## Troubleshooting

### Common Issues and Fixes

| Problem | Likely Cause | Solution |
|---------|-------------|----------|
| `Could not open camera source: 0` | Another app is using the camera | Close Zoom, Teams, browser tabs using camera |
| `pip install dlib` fails | cmake not installed | Run `sudo apt install cmake` or `brew install cmake` first |
| `FileNotFoundError: shape_predictor...` | Model file not downloaded | Run the wget + bunzip2 commands from Step 4 |
| `[MQTT] Could not connect to Pi` | Pi offline, wrong IP, or Mosquitto not running | Check IP, run `sudo systemctl status mosquitto` on Pi |
| Everything triggers as motion | Threshold too low | Increase `threshold` in MotionDetector (try 8000-10000) |
| Motion is never detected | Threshold too high | Decrease `threshold` (try 2000-3000) |
| Drowsiness never triggers | Threshold too strict | Lower `ear_threshold` (try 0.22) or `consec_frames` (try 10) |
| Drowsiness triggers too easily | Threshold too loose | Raise `ear_threshold` (try 0.18) or `consec_frames` (try 20) |
| System runs slowly on Pi 3 | Pi 3 has limited CPU/RAM | Run detection on laptop only; Pi just handles buzzer |
| `ModuleNotFoundError` when running main.py | Virtual environment not activated | Run `source venv/bin/activate` |
| Camera feed is laggy | Processing too heavy | Reduce resolution in CameraStream (try 320x240) |
| Face detection is intermittent | Face too far or at angle | Move closer to camera, face it directly |
| Merge conflicts when pulling dev | Multiple people edited same file | Coordinate with the other person, resolve manually |
| Pushed to wrong branch | Committed on dev or main by accident | Use `git reset` and `git stash`, switch to your branch |

### Getting Help

If you're stuck on your module:
1. Check the detailed comments in your source file — every method has step-by-step instructions
2. Run your module's standalone test (`python src/<your-file>.py`) to isolate the issue
3. Ask a teammate — the dependency map above shows who works on related modules
4. Check OpenCV and dlib documentation online