import cv2
import sys
from ultralytics import YOLO

def main():
    # Initialize webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        sys.exit(1)

    # Initialize YOLOv8 model
    print("Loading YOLOv8 model...")
    model = YOLO('yolov8n.pt')
    print("Model loaded.")

    print("Webcam started. Press 'q' to quit.")

    while True:
        # Read frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image.")
            break

        # Run inference
        results = model(frame, verbose=False)

        # Process results
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Class ID and Check if it's a cell phone
                cls = int(box.cls[0])
                class_name = model.names[cls]
                
                if class_name == 'cell phone':
                    # Bounding box coordinates
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    # Draw rectangle
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"Phone {box.conf[0]:.2f}", (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Display the frame
        cv2.imshow('Doomscroll Detector - Phase 2', frame)

        # Quit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
