import cv2
import numpy as np
from typing import List, Tuple, Optional
import mediapipe as mp

class ImagePreprocessor:
    """Utility class for preprocessing images for sign detection"""
    
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        
    def resize_image(self, image: np.ndarray, target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
        """Resize image to target size while maintaining aspect ratio"""
        h, w = image.shape[:2]
        target_w, target_h = target_size
        
        # Calculate scaling factor
        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        # Resize image
        resized = cv2.resize(image, (new_w, new_h))
        
        # Create canvas with target size
        canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        
        # Center the resized image
        y_offset = (target_h - new_h) // 2
        x_offset = (target_w - new_w) // 2
        canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
        
        return canvas
    
    def normalize_image(self, image: np.ndarray) -> np.ndarray:
        """Normalize image pixel values to [0, 1] range"""
        return image.astype(np.float32) / 255.0
    
    def apply_histogram_equalization(self, image: np.ndarray) -> np.ndarray:
        """Apply histogram equalization to improve contrast"""
        if len(image.shape) == 3:
            # Convert to LAB color space
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # Apply CLAHE to L channel
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            
            # Merge channels and convert back
            lab = cv2.merge([l, a, b])
            return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        else:
            # Grayscale image
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            return clahe.apply(image)
    
    def extract_hand_region(self, image: np.ndarray, landmarks: List) -> np.ndarray:
        """Extract hand region from image based on landmarks"""
        if not landmarks or len(landmarks) < 21:
            return image
        
        # Get bounding box of hand landmarks
        x_coords = [landmark[0] for landmark in landmarks]
        y_coords = [landmark[1] for landmark in landmarks]
        
        x_min, x_max = int(min(x_coords)), int(max(x_coords))
        y_min, y_max = int(min(y_coords)), int(max(y_coords))
        
        # Add padding
        padding = 20
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(image.shape[1], x_max + padding)
        y_max = min(image.shape[0], y_max + padding)
        
        # Extract hand region
        hand_region = image[y_min:y_max, x_min:x_max]
        
        return hand_region
    
    def create_landmark_image(self, landmarks: List, image_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
        """Create a visualization of hand landmarks"""
        canvas = np.zeros((image_size[1], image_size[0], 3), dtype=np.uint8)
        
        if not landmarks:
            return canvas
        
        # Draw landmarks
        for i, landmark in enumerate(landmarks):
            x, y = int(landmark[0] * image_size[0]), int(landmark[1] * image_size[1])
            cv2.circle(canvas, (x, y), 3, (0, 255, 0), -1)
            
            # Draw connections between landmarks
            if i < len(landmarks) - 1:
                next_x, next_y = int(landmarks[i + 1][0] * image_size[0]), int(landmarks[i + 1][1] * image_size[1])
                cv2.line(canvas, (x, y), (next_x, next_y), (255, 0, 0), 1)
        
        return canvas
    
    def augment_image(self, image: np.ndarray) -> List[np.ndarray]:
        """Apply data augmentation to create multiple training samples"""
        augmented = []
        
        # Original image
        augmented.append(image)
        
        # Horizontal flip
        flipped = cv2.flip(image, 1)
        augmented.append(flipped)
        
        # Rotation (small angles)
        for angle in [-15, 15]:
            h, w = image.shape[:2]
            center = (w // 2, h // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(image, rotation_matrix, (w, h))
            augmented.append(rotated)
        
        # Brightness adjustment
        for alpha in [0.8, 1.2]:
            adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=0)
            augmented.append(adjusted)
        
        return augmented
    
    def preprocess_for_model(self, image: np.ndarray, target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
        """Complete preprocessing pipeline for model input"""
        # Resize
        resized = self.resize_image(image, target_size)
        
        # Normalize
        normalized = self.normalize_image(resized)
        
        # Ensure correct shape for model input
        if len(normalized.shape) == 3:
            # Add batch dimension
            return np.expand_dims(normalized, axis=0)
        else:
            return normalized 