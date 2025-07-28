import numpy as np
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

class AccuracyScorer:
    """Service for calculating accuracy scores for sign detection"""
    
    def __init__(self):
        self.confidence_threshold = 0.7
        self.landmark_weights = self._initialize_landmark_weights()
        
    def _initialize_landmark_weights(self) -> Dict[int, float]:
        """Initialize weights for different hand landmarks"""
        # Weights based on importance for sign recognition
        weights = {}
        
        # Fingertips (most important for sign recognition)
        for i in [4, 8, 12, 16, 20]:  # Thumb tip, index tip, middle tip, ring tip, pinky tip
            weights[i] = 1.0
        
        # Finger PIP joints (second most important)
        for i in [3, 7, 11, 15, 19]:  # Thumb IP, index PIP, middle PIP, ring PIP, pinky PIP
            weights[i] = 0.8
        
        # Finger MCP joints
        for i in [2, 6, 10, 14, 18]:  # Thumb MCP, index MCP, middle MCP, ring MCP, pinky MCP
            weights[i] = 0.6
        
        # Wrist and palm landmarks
        for i in [0, 1, 5, 9, 13, 17]:  # Wrist, thumb CMC, index CMC, middle CMC, ring CMC, pinky CMC
            weights[i] = 0.4
            
        return weights
    
    def calculate_landmark_accuracy(self, detected_landmarks: List, expected_landmarks: List) -> float:
        """Calculate accuracy based on landmark positions"""
        if not detected_landmarks or not expected_landmarks:
            return 0.0
        
        if len(detected_landmarks) != len(expected_landmarks):
            return 0.0
        
        total_error = 0.0
        total_weight = 0.0
        
        for i, (detected, expected) in enumerate(zip(detected_landmarks, expected_landmarks)):
            weight = self.landmark_weights.get(i, 0.5)
            
            # Calculate Euclidean distance between landmarks
            error = np.sqrt(
                (detected[0] - expected[0])**2 + 
                (detected[1] - expected[1])**2
            )
            
            total_error += error * weight
            total_weight += weight
        
        if total_weight == 0:
            return 0.0
        
        # Normalize error and convert to accuracy score
        avg_error = total_error / total_weight
        accuracy = max(0.0, 1.0 - avg_error)
        
        return accuracy
    
    def calculate_confidence_score(self, model_confidence: float) -> float:
        """Calculate confidence score based on model output"""
        if model_confidence < 0:
            return 0.0
        if model_confidence > 1:
            return 1.0
        return model_confidence
    
    def calculate_overall_accuracy(self, 
                                 detected_sign: str, 
                                 expected_sign: str,
                                 model_confidence: float,
                                 landmark_accuracy: float = None) -> Dict[str, float]:
        """Calculate overall accuracy score for sign detection"""
        
        # Base accuracy from model confidence
        confidence_score = self.calculate_confidence_score(model_confidence)
        
        # Sign matching accuracy
        sign_match = 1.0 if detected_sign == expected_sign else 0.0
        
        # Combine scores
        if landmark_accuracy is not None:
            # Weighted combination of all factors
            overall_accuracy = (
                0.4 * confidence_score +
                0.4 * sign_match +
                0.2 * landmark_accuracy
            )
        else:
            # Simplified scoring without landmark accuracy
            overall_accuracy = (
                0.6 * confidence_score +
                0.4 * sign_match
            )
        
        return {
            "overall_accuracy": round(overall_accuracy, 3),
            "confidence_score": round(confidence_score, 3),
            "sign_match": round(sign_match, 3),
            "landmark_accuracy": round(landmark_accuracy, 3) if landmark_accuracy is not None else None
        }
    
    def evaluate_detection_quality(self, landmarks: List, confidence: float) -> Dict[str, Any]:
        """Evaluate the quality of hand detection"""
        if not landmarks or len(landmarks) < 21:
            return {
                "quality": "poor",
                "score": 0.0,
                "issues": ["insufficient_landmarks"]
            }
        
        issues = []
        score = 1.0
        
        # Check landmark visibility
        visible_landmarks = sum(1 for lm in landmarks if lm[2] > 0.5)  # z-coordinate visibility
        visibility_ratio = visible_landmarks / len(landmarks)
        
        if visibility_ratio < 0.8:
            issues.append("poor_visibility")
            score *= 0.7
        
        # Check confidence threshold
        if confidence < self.confidence_threshold:
            issues.append("low_confidence")
            score *= 0.8
        
        # Check landmark distribution (ensure hand is properly detected)
        x_coords = [lm[0] for lm in landmarks]
        y_coords = [lm[1] for lm in landmarks]
        
        x_range = max(x_coords) - min(x_coords)
        y_range = max(y_coords) - min(y_coords)
        
        if x_range < 0.1 or y_range < 0.1:
            issues.append("compressed_landmarks")
            score *= 0.6
        
        # Determine quality level
        if score >= 0.8:
            quality = "excellent"
        elif score >= 0.6:
            quality = "good"
        elif score >= 0.4:
            quality = "fair"
        else:
            quality = "poor"
        
        return {
            "quality": quality,
            "score": round(score, 3),
            "issues": issues,
            "visibility_ratio": round(visibility_ratio, 3)
        }
    
    def get_accuracy_feedback(self, accuracy_score: float) -> Dict[str, Any]:
        """Generate feedback based on accuracy score"""
        if accuracy_score >= 0.9:
            feedback = {
                "message": "Excellent! Your sign is very accurate.",
                "suggestion": "Keep up the great work!",
                "color": "green"
            }
        elif accuracy_score >= 0.7:
            feedback = {
                "message": "Good! Your sign is mostly accurate.",
                "suggestion": "Try to make the gesture a bit more precise.",
                "color": "yellow"
            }
        elif accuracy_score >= 0.5:
            feedback = {
                "message": "Fair accuracy. There's room for improvement.",
                "suggestion": "Check your hand position and finger placement.",
                "color": "orange"
            }
        else:
            feedback = {
                "message": "Low accuracy. Let's work on this sign.",
                "suggestion": "Review the correct hand position and try again.",
                "color": "red"
            }
        
        return feedback 