try:
    import cv2
except ImportError:
    print("Warning: OpenCV not installed. Using mock hand detection.")
    cv2 = None

import mediapipe as mp
import numpy as np
from typing import List, Tuple, Optional, Dict
import logging

logger = logging.getLogger(__name__)

class HandDetector:
    """Hand detection and landmark extraction using MediaPipe"""
    
    def __init__(self, 
                 static_image_mode: bool = False,
                 max_num_hands: int = 2,
                 min_detection_confidence: float = 0.7,
                 min_tracking_confidence: float = 0.5):
        """
        Initialize the hand detector
        
        Args:
            static_image_mode: Whether to treat input as static images
            max_num_hands: Maximum number of hands to detect
            min_detection_confidence: Minimum confidence for hand detection
            min_tracking_confidence: Minimum confidence for hand tracking
        """
        if cv2 is None:
            logger.warning("OpenCV not available, using mock hand detection")
            self.mock_mode = True
        else:
            self.mock_mode = False
            
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        if not self.mock_mode:
            self.hands = self.mp_hands.Hands(
                static_image_mode=static_image_mode,
                max_num_hands=max_num_hands,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence
            )
        
        self.hand_landmarks_connections = self.mp_hands.HAND_CONNECTIONS
        
    async def detect_hands(self, image_data):
        """
        Detect hands in the image and return landmarks
        
        Args:
            image_data: Input image data (base64 string or numpy array)
            
        Returns:
            List of hand landmarks or empty list if no hands detected
        """
        if self.mock_mode:
            # Return mock landmarks for testing
            return self._get_mock_landmarks()
        
        try:
            # Convert base64 to numpy array if needed
            if isinstance(image_data, str):
                import base64
                image_bytes = base64.b64decode(image_data.split(',')[1])
                image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
            else:
                image = image_data
            
            # Convert BGR to RGB
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Process the image
            results = self.hands.process(rgb_image)
            
            if results.multi_hand_landmarks:
                # Return landmarks from the first detected hand
                landmarks = results.multi_hand_landmarks[0]
                return self._extract_landmarks(landmarks, image.shape)
            
            return []
            
        except Exception as e:
            logger.error(f"Error in hand detection: {e}")
            return []
    
    def _get_mock_landmarks(self):
        """Return mock landmarks for testing"""
        # Mock 21 hand landmarks (MediaPipe format)
        mock_landmarks = []
        for i in range(21):
            mock_landmarks.append([0.5 + i * 0.01, 0.5 + i * 0.01, 0.0])
        return mock_landmarks
    
    def _extract_landmarks(self, hand_landmarks, image_shape: Tuple[int, int, int]) -> List[Dict]:
        """
        Extract hand landmarks with normalized and pixel coordinates
        
        Args:
            hand_landmarks: MediaPipe hand landmarks
            image_shape: Shape of the input image (height, width, channels)
            
        Returns:
            List of landmark dictionaries
        """
        landmarks = []
        height, width = image_shape[:2]
        
        for i, landmark in enumerate(hand_landmarks.landmark):
            landmarks.append({
                "id": i,
                "name": self._get_landmark_name(i),
                "x": landmark.x,  # Normalized coordinate (0-1)
                "y": landmark.y,  # Normalized coordinate (0-1)
                "z": landmark.z,  # Relative depth
                "pixel_x": int(landmark.x * width),
                "pixel_y": int(landmark.y * height),
                "visibility": getattr(landmark, 'visibility', 1.0)
            })
            
        return landmarks
    
    def _get_landmark_name(self, landmark_id: int) -> str:
        """Get the name of a landmark by its ID"""
        landmark_names = [
            "WRIST",
            "THUMB_CMC", "THUMB_MCP", "THUMB_IP", "THUMB_TIP",
            "INDEX_FINGER_MCP", "INDEX_FINGER_PIP", "INDEX_FINGER_DIP", "INDEX_FINGER_TIP",
            "MIDDLE_FINGER_MCP", "MIDDLE_FINGER_PIP", "MIDDLE_FINGER_DIP", "MIDDLE_FINGER_TIP",
            "RING_FINGER_MCP", "RING_FINGER_PIP", "RING_FINGER_DIP", "RING_FINGER_TIP",
            "PINKY_MCP", "PINKY_PIP", "PINKY_DIP", "PINKY_TIP"
        ]
        
        return landmark_names[landmark_id] if landmark_id < len(landmark_names) else f"LANDMARK_{landmark_id}"
    
    def _calculate_bounding_box(self, landmarks: List[Dict]) -> Dict:
        """
        Calculate bounding box for detected hand
        
        Args:
            landmarks: List of hand landmarks
            
        Returns:
            Bounding box dictionary
        """
        if not landmarks:
            return {}
            
        x_coords = [lm["pixel_x"] for lm in landmarks]
        y_coords = [lm["pixel_y"] for lm in landmarks]
        
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        
        # Add some padding
        padding = 20
        width = x_max - x_min + 2 * padding
        height = y_max - y_min + 2 * padding
        
        return {
            "x": max(0, x_min - padding),
            "y": max(0, y_min - padding),
            "width": width,
            "height": height,
            "center_x": x_min + (x_max - x_min) // 2,
            "center_y": y_min + (y_max - y_min) // 2
        }
    
    def draw_landmarks(self, image: np.ndarray, detection_result: Dict) -> np.ndarray:
        """
        Draw hand landmarks on the image
        
        Args:
            image: Input image
            detection_result: Result from detect_hands method
            
        Returns:
            Image with drawn landmarks
        """
        if not detection_result["hands_detected"]:
            return image
            
        annotated_image = image.copy()
        
        # Draw landmarks for each hand
        for i, landmarks in enumerate(detection_result["hand_landmarks"]):
            # Draw connections
            self._draw_connections(annotated_image, landmarks)
            
            # Draw landmark points
            for landmark in landmarks:
                cv2.circle(
                    annotated_image,
                    (landmark["pixel_x"], landmark["pixel_y"]),
                    5,
                    (0, 255, 0),  # Green color
                    -1
                )
                
            # Draw bounding box
            if i < len(detection_result["bounding_boxes"]):
                bbox = detection_result["bounding_boxes"][i]
                cv2.rectangle(
                    annotated_image,
                    (bbox["x"], bbox["y"]),
                    (bbox["x"] + bbox["width"], bbox["y"] + bbox["height"]),
                    (255, 0, 0),  # Blue color
                    2
                )
                
                # Add hand classification label
                if i < len(detection_result["hand_classifications"]):
                    label = detection_result["hand_classifications"][i]["label"]
                    score = detection_result["hand_classifications"][i]["score"]
                    text = f"{label} ({score:.2f})"
                    
                    cv2.putText(
                        annotated_image,
                        text,
                        (bbox["x"], bbox["y"] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 0, 0),
                        2
                    )
        
        return annotated_image
    
    def _draw_connections(self, image: np.ndarray, landmarks: List[Dict]):
        """Draw connections between hand landmarks"""
        connections = [
            # Thumb
            (0, 1), (1, 2), (2, 3), (3, 4),
            # Index finger
            (0, 5), (5, 6), (6, 7), (7, 8),
            # Middle finger
            (0, 9), (9, 10), (10, 11), (11, 12),
            # Ring finger
            (0, 13), (13, 14), (14, 15), (15, 16),
            # Pinky
            (0, 17), (17, 18), (18, 19), (19, 20)
        ]
        
        for start_idx, end_idx in connections:
            if start_idx < len(landmarks) and end_idx < len(landmarks):
                start_point = (landmarks[start_idx]["pixel_x"], landmarks[start_idx]["pixel_y"])
                end_point = (landmarks[end_idx]["pixel_x"], landmarks[end_idx]["pixel_y"])
                
                cv2.line(image, start_point, end_point, (0, 255, 255), 2)  # Yellow lines
    
    def get_hand_features(self, landmarks: List[Dict]) -> Dict:
        """
        Extract geometric features from hand landmarks for ML model
        
        Args:
            landmarks: List of hand landmarks
            
        Returns:
            Dictionary of extracted features
        """
        if not landmarks or len(landmarks) != 21:
            return {}
            
        features = {}
        
        # Finger tip positions relative to wrist
        wrist = landmarks[0]
        finger_tips = [landmarks[4], landmarks[8], landmarks[12], landmarks[16], landmarks[20]]  # Thumb, Index, Middle, Ring, Pinky
        
        # Calculate distances from wrist to fingertips
        for i, tip in enumerate(finger_tips):
            finger_name = ["thumb", "index", "middle", "ring", "pinky"][i]
            distance = np.sqrt((tip["x"] - wrist["x"])**2 + (tip["y"] - wrist["y"])**2)
            features[f"{finger_name}_distance"] = distance
        
        # Calculate angles between fingers
        for i in range(len(finger_tips) - 1):
            finger1 = ["thumb", "index", "middle", "ring"][i]
            finger2 = ["index", "middle", "ring", "pinky"][i]
            
            # Vector from wrist to each fingertip
            v1 = np.array([finger_tips[i]["x"] - wrist["x"], finger_tips[i]["y"] - wrist["y"]])
            v2 = np.array([finger_tips[i+1]["x"] - wrist["x"], finger_tips[i+1]["y"] - wrist["y"]])
            
            # Calculate angle
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
            features[f"{finger1}_{finger2}_angle"] = angle
        
        # Hand orientation (angle of line from wrist to middle finger MCP)
        middle_mcp = landmarks[9]
        hand_vector = np.array([middle_mcp["x"] - wrist["x"], middle_mcp["y"] - wrist["y"]])
        hand_angle = np.arctan2(hand_vector[1], hand_vector[0])
        features["hand_orientation"] = hand_angle
        
        # Hand size (distance from wrist to middle finger tip)
        middle_tip = landmarks[12]
        hand_size = np.sqrt((middle_tip["x"] - wrist["x"])**2 + (middle_tip["y"] - wrist["y"])**2)
        features["hand_size"] = hand_size
        
        return features
    
    def close(self):
        """Clean up resources"""
        if hasattr(self, 'hands') and not self.mock_mode:
            self.hands.close()
