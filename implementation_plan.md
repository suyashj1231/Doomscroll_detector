# Doomscroll Detector Implementation Plan

## Phase 1: Environment & Boilerplate
- [ ] Create `requirements.txt` with dependencies (opencv-python, mediapipe, ultralytics, pyttsx3, numpy).
- [ ] Create `main.py` with a basic webcam loop to verify video capture works.

## Phase 2: Phone Detection (The "Doomscroll" Object)
- [ ] Initialize YOLOv8 (`yolov8n.pt`) in `main.py`.
- [ ] Run inference on video frames to detect the "cell phone" class.
- [ ] Draw bounding boxes around phones for visual debugging.

## Phase 3: Gaze & Face Tracking
- [ ] Integrate MediaPipe Face Mesh.
- [ ] Implement logic to estimate head pose (looking at screen vs. looking down).
- [ ] Refine with iris tracking if necessary.

## Phase 4: The Logic Core
- [ ] Define the "Trigger Condition" (e.g., Phone detected AND Gaze down).
- [ ] Create a `Detector` class to manage state and logic.

## Phase 5: Intervention (The "Stop" Signal)
- [ ] Implement `pyttsx3` for audio alerts.
- [ ] Implement cooldown logic to prevent continuous spamming of alerts.

## Phase 6: Polish
- [ ] Add visual notifications (e.g., screen overlay).
- [ ] Optimize performance.