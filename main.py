import cv2
import sys
from object_detector import ObjectDetector
from gaze_detector import GazeDetector

def main():
    # Initialize webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        sys.exit(1)

    # Initialize Detectors
    object_detector = ObjectDetector(model_path='yolov8n.pt', target_classes=['cell phone'])
    gaze_detector = GazeDetector()

    print("Webcam started. Press 'q' to quit.")

    while True:
        # Read frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image.")
            break

        # 1. Gaze Detection (Draws on frame directly)
        gaze_result = gaze_detector.detect(frame)
        
        # 2. Object Detection
        results = object_detector.detect(frame)

        # Process Object Results
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Bounding box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Get Track ID if available
                track_id = int(box.id[0]) if box.id is not None else 0

                # Draw rectangle
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"Phone #{track_id} {box.conf[0]:.2f}", (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Display Interaction Status
        status_color = (0, 0, 255) if gaze_result["is_looking_down"] else (0, 255, 0)
        status_text = "LOOKING DOWN" if gaze_result["is_looking_down"] else "LOOKING UP"
        cv2.putText(frame, f"Status: {status_text}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)

        # Display the frame
        cv2.imshow('Doomscroll Detector - Phase 3', frame)

        # Quit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
