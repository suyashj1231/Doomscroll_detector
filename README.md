# Doomscroll Detector

Webcam-based focus monitor that notices when you've stopped looking at your screen and started looking at your phone. It runs two vision models side by side on every frame, a face-mesh head-pose and iris tracker plus a YOLOv8 phone tracker, all on your machine. No frame is saved or sent anywhere.

## How it decides you're looking down

**Head pose from six landmarks.** MediaPipe Face Mesh (with `refine_landmarks` for the iris points) gives 478 facial landmarks. Six of them, the nose tip, chin, eye corners and mouth corners, are matched against a generic 3D face model and passed to OpenCV's `solvePnP`. The resulting rotation is decomposed (`Rodrigues` → `RQDecomp3x3`) into pitch, yaw and roll, and a projected nose vector is drawn so you can see the estimate.

**Iris position for subtle glances.** People often glance down at a phone with their eyes, barely moving their head. So the tracker also measures where the iris sits between the upper and lower eyelid (landmarks 468 and 159/145) as a 0-1 ratio.

**Hybrid rule.**

| Condition | Verdict |
|---|---|
| Head pitch > 10° | looking down |
| Head pitch > 2° **and** iris ratio > 0.55 | looking down (subtle glance) |
| otherwise | looking at the screen |

**Phone tracking.** Ultralytics YOLOv8n runs in tracking mode (`persist=True`) restricted to the COCO `cell phone` class, so each phone keeps a stable ID across frames. It runs on Apple's Metal backend (`device='mps'`); change `object_detector.py` for CUDA or CPU.

The overlay shows the live pitch, the iris ratio, tracked phones with their IDs and confidence scores, and a `LOOKING DOWN` / `LOOKING UP` status.

## Run

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python main.py        # press q to quit
```

`yolov8n.pt` is included. Your terminal needs camera permission on macOS.

## Code map

| File | Role |
|---|---|
| `main.py` | Capture loop, runs both detectors, draws the overlay |
| `gaze_detector.py` | Face mesh, PnP head pose, iris ratio, hybrid looking-down rule |
| `object_detector.py` | YOLOv8 phone tracking |
| `implementation_plan.md` | Build plan, phase by phase |

## Next

- Trigger when a phone is tracked **and** gaze is down for a sustained window
- Spoken alert through `pyttsx3`, with a cooldown so it doesn't nag
- Run YOLO on every Nth frame and reuse tracks in between, to cut CPU/GPU load
