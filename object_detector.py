from ultralytics import YOLO
import cv2

class ObjectDetector:
    def __init__(self, model_path='yolov8n.pt', target_classes=None):
        """
        Initialize the YOLOv8 object detector.
        
        Args:
            model_path (str): Path to the YOLO model file.
            target_classes (list): List of class names to detect (e.g., ['cell phone']). 
                                   If None, detects all classes.
        """
        print(f"Loading YOLOv8 model from {model_path}...")
        self.model = YOLO(model_path)
        self.device = 'mps'  # Hardcoded optimization for this setup
        print("Model loaded.")
        
        self.target_class_ids = None
        if target_classes:
            self.target_class_ids = [
                k for k, v in self.model.names.items() 
                if v in target_classes
            ]
            print(f"Filtering for classes: {target_classes} (IDs: {self.target_class_ids})")

    def detect(self, frame):
        """
        Run object tracking on the frame.
        
        Args:
            frame: The video frame to process.
            
        Returns:
            list: List of detected objects (or the raw results object).
        """
        # Run inference with tracking (persist=True)
        # verbose=False keeps the terminal clean
        results = self.model.track(
            frame, 
            persist=True, 
            verbose=False, 
            device=self.device, 
            classes=self.target_class_ids
        )
        return results
