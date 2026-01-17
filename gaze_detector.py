import cv2
import mediapipe as mp
import numpy as np
import time

class GazeDetector:
    def __init__(self):
        """
        Initialize MediaPipe Face Mesh for gaze/head pose tracking.
        """
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            refine_landmarks=True
        )
        
        # Camera matrix placeholders (will be set on first frame)
        self.cam_matrix = None
        self.dist_coeffs = np.zeros((4, 1))
        
        # 3D model points for PnP (Generic human face)
        self.model_points = np.array([
            (0.0, 0.0, 0.0),             # Nose tip
            (0.0, -330.0, -65.0),        # Chin
            (-225.0, 170.0, -135.0),     # Left eye left corner
            (225.0, 170.0, -135.0),      # Right eye right corner
            (-150.0, -150.0, -125.0),    # Left Mouth corner
            (150.0, -150.0, -125.0)      # Right mouth corner
        ])

    def detect(self, image):
        """
        Estimate head pose to detect if the user is looking down.
        
        Args:
            image: BGR image from webcam.
            
        Returns:
            dict: {
                "looking_down": bool,
                "pitch": float,
                "yaw": float,
                "roll": float,
                "landmarks": list (optional)
            }
        """
        start = time.time()
        img_h, img_w, _ = image.shape
        
        # Initialize camera matrix if first run
        if self.cam_matrix is None:
            focal_length = 1 * img_w
            self.cam_matrix = np.array([
                [focal_length, 0, img_w / 2],
                [0, focal_length, img_h / 2],
                [0, 0, 1]
            ])

        # Convert to RGB for MediaPipe
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        rgb_image.flags.writeable = False # Optimization
        
        results = self.face_mesh.process(rgb_image)
        
        rgb_image.flags.writeable = True
        
        output = {
            "is_looking_down": False,
            "pitch": 0,
            "yaw": 0,
            "roll": 0
        }

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # 2D Image Points
                image_points = np.array([
                    (face_landmarks.landmark[1].x * img_w, face_landmarks.landmark[1].y * img_h),     # Nose tip
                    (face_landmarks.landmark[152].x * img_w, face_landmarks.landmark[152].y * img_h), # Chin
                    (face_landmarks.landmark[263].x * img_w, face_landmarks.landmark[263].y * img_h), # Left eye
                    (face_landmarks.landmark[33].x * img_w, face_landmarks.landmark[33].y * img_h),   # Right eye
                    (face_landmarks.landmark[291].x * img_w, face_landmarks.landmark[291].y * img_h), # Left mouth
                    (face_landmarks.landmark[61].x * img_w, face_landmarks.landmark[61].y * img_h)    # Right mouth
                ], dtype="double")

                # Solve PnP
                (success, rotation_vector, translation_vector) = cv2.solvePnP(
                    self.model_points, 
                    image_points, 
                    self.cam_matrix, 
                    self.dist_coeffs, 
                    flags=cv2.SOLVEPNP_ITERATIVE
                )

                if success:
                    # Get rotational matrix
                    rmat, jac = cv2.Rodrigues(rotation_vector)
                    
                    # Get angles
                    angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)
                    
                    pitch = angles[0] * 360
                    yaw = angles[1] * 360
                    roll = angles[2] * 360
                    
                    output["pitch"] = pitch
                    output["yaw"] = yaw
                    output["roll"] = roll
                    
                    # Heuristic for "Looking Down"
                    # Pitch is usually positive when looking down with this coord system? 
                    # Need to verify experimentally, but usually pitch > 10-15 degrees is down.
                    # Or < -10 depending on axis definition. 
                    # Based on standard MP/OpenCV: Looking down typically increases Pitch positive or negative depending on points.
                    # Let's assume a threshold for now and print it to debug.
                    if pitch > 10: 
                        output["is_looking_down"] = True
                    
                    # Draw nose direction
                    nose_end_point2D, jacobian = cv2.projectPoints(
                        np.array([(0.0, 0.0, 1000.0)]), 
                        rotation_vector, 
                        translation_vector, 
                        self.cam_matrix, 
                        self.dist_coeffs
                    )
                    
                    p1 = (int(image_points[0][0]), int(image_points[0][1]))
                    p2 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))
                    
                    cv2.line(image, p1, p2, (255, 0, 0), 2)
                    cv2.putText(image, f"Pitch: {int(pitch)}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # Only process the first face
                face_2d = image_points
                face_3d = self.model_points
                
                # --- Iris Tracking ---
                # Left Eye Indices: 33 (Left), 133 (Right), 159 (Top), 145 (Bottom)
                # Right Eye Indices: 362 (Left), 263 (Right), 386 (Top), 374 (Bottom)
                # Left Iris: 468, Right Iris: 473
                
                left_iris = face_landmarks.landmark[468]
                right_iris = face_landmarks.landmark[473]
                
                left_eye_top = face_landmarks.landmark[159]
                left_eye_bottom = face_landmarks.landmark[145]
                
                # Height of left eye
                eye_height = abs(left_eye_top.y - left_eye_bottom.y)
                # Distance from top eyelid to iris center
                iris_to_top = abs(left_iris.y - left_eye_top.y)
                
                # Vertical Ratio (0.0 = Top, 1.0 = Bottom)
                # If ratio > 0.5, eye is looking lower.
                vertical_ratio = iris_to_top / eye_height if eye_height > 0 else 0.5
                
                output["iris_tracked"] = True
                output["gaze_ratio"] = vertical_ratio
                
                # Hybrid Logic: 
                # 1. Head Pitch > 10 ("True" Looking Down)
                # 2. Head Pitch > 2 AND Eyes Looking Down (Ratio > 0.6) - Subtle Look
                
                if pitch > 10:
                    output["is_looking_down"] = True
                elif pitch > 2 and vertical_ratio > 0.55:
                    output["is_looking_down"] = True
                
                # Debug Visuals for Gaze
                cv2.putText(image, f"Eye Ratio: {vertical_ratio:.2f}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                
                # Draw Iris for confirmation
                ix, iy = int(left_iris.x * img_w), int(left_iris.y * img_h)
                cv2.circle(image, (ix, iy), 3, (0, 255, 255), -1)

                break
        
        return output
