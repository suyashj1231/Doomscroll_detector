# Doomscroll Detector

**Doomscroll Detector** is an intelligent background utility designed to keep you focused and productive. By leveraging real-time computer vision, it detects when your attention drifts from your screen to your phone and vocally reminds you to get back on track.

> "As soon as you stop looking at the screen, we say STOP."

##  How It Works

The application uses your webcam to monitor two key behaviors in real-time:
1.  **Gaze & Head Pose**: Analyses your face landmarks to determine if you are looking at the monitor or looking down/away.
2.  **Object Detection**: Specifically scans the frame for mobile phones.

If the system detects that your eyes are glued to a phone or your focus has left the primary screen for too long, it triggers an audio alert to break the "doomscrolling" loop.

##  Tech Stack

We have chosen a robust Python-based stack for performance and ease of access to hardware acceleration.

-   **Language**: [Python 3.10+](https://www.python.org/)
-   **Computer Vision**: [OpenCV](https://opencv.org/) (Video Capture & Processing)
-   **Face/Eye Tracking**: [MediaPipe](https://developers.google.com/mediapipe) (High-fidelity Face Mesh & Iris Tracking)
-   **Object Detection**: [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) (State-of-the-art detection for identifying 'cell phone')
-   **Audio**: `pyttsx3` (Offline Text-to-Speech engine) or `playsound`

##  Features

-   **Real-time Monitoring**: Low-latency video processing.
-   **Phone Detection**: Identifies when a phone enters the frame and face interaction occurs.
-   **Gaze Tracking**: Estimates where the user is looking (Screen vs. Down/Side).
-   **Instant Feedback**: Audio cues to interrupt distraction immediately.
-   **Privacy Focused**: All processing happens locally on your machine. No video is saved or sent to the cloud.

##  Getting Started

### Prerequisites

-   Python 3.10 or higher installed.
-   A webcam.

### Installation

1.  Clone the repo:
    ```bash
    git clone https://github.com/suyashj1231/Doomscroll_detector.git
    cd Doomscroll_detector
    ```

2.  Create a virtual environment:
    ```bash
    python -m venv venv
    # Windows
    .\venv\Scripts\activate
    # Mac/Linux
    source venv/bin/activate
    ```

3.  Install dependencies:
    ```bash
    pip install opencv-python mediapipe ultralytics pyttsx3 numpy
    ```

4.  Download YOLO weights (Auto-downloaded on first run, but you can grab `yolov8n.pt` for speed).

### Usage

Run the main detector script:

```bash
python main.py
```

Press `q` to quit the application window.

##  Roadmap

-   [ ] **Productivity Logging**: Track stats on how often you get distracted per hour.
-   [ ] **Custom Alerts**: Allow users to record their own "Stop!" messages.
-   [ ] **Silent Mode**: Visual notification instead of audio.
-   [ ] **Pomodoro Integration**: Only run detection during work intervals.

##  Contributing

Contributions are welcome! detection accuracy improvements are always needed.

##  License

Distributed under the MIT License.
