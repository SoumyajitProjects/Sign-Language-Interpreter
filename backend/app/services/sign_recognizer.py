import numpy as np
try:
    import tensorflow as tf
    from tensorflow import keras
    TENSORFLOW_AVAILABLE = True
except ImportError:
    print("Warning: TensorFlow not installed. Using mock sign recognition.")
    TENSORFLOW_AVAILABLE = False

try:
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    print("Warning: Scikit-learn not installed. Using mock preprocessing.")
    SKLEARN_AVAILABLE = False

from typing import Dict, List, Tuple, Optional
import pickle
import os
import logging
import json
from datetime import datetime

from .hand_detector import HandDetector
from .accuracy_scorer import AccuracyScorer

logger = logging.getLogger(__name__)

class SignRecognizer:
    """Sign language recognition using machine learning"""
    
    def __init__(self, model_path: str = "data/models/sign_classifier.h5"):
        """
        Initialize the sign recognizer
        
        Args:
            model_path: Path to the trained model file
        """
        self.model_path = model_path
        self.model = None
        self.scaler = None
        self.label_encoder = None
        self.hand_detector = HandDetector()
        self.accuracy_scorer = AccuracyScorer()
        
        # ASL alphabet mapping
        self.asl_labels = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        self.num_classes = len(self.asl_labels)
        
        # Feature names for consistent ordering
        self.feature_names = [
            'thumb_distance', 'index_distance', 'middle_distance', 'ring_distance', 'pinky_distance',
            'thumb_index_angle', 'index_middle_angle', 'middle_ring_angle', 'ring_pinky_angle',
            'hand_orientation', 'hand_size'
        ]
        
    async def initialize(self):
        """Initialize the model and preprocessors"""
        try:
            if os.path.exists(self.model_path):
                logger.info(f"Loading existing model from {self.model_path}")
                self.model = keras.models.load_model(self.model_path)
                
                # Load scaler if exists
                scaler_path = self.model_path.replace('.h5', '_scaler.pkl')
                if os.path.exists(scaler_path):
                    with open(scaler_path, 'rb') as f:
                        self.scaler = pickle.load(f)
                else:
                    logger.warning("Scaler not found, creating new one")
                    self.scaler = StandardScaler()
            else:
                logger.info("No existing model found, creating new model")
                await self._create_model()
                
        except Exception as e:
            logger.error(f"Error initializing sign recognizer: {e}")
            await self._create_model()
    
    async def _create_model(self):
        """Create and compile a new neural network model"""
        try:
            # Create neural network architecture
            self.model = keras.Sequential([
                keras.layers.Input(shape=(len(self.feature_names),)),
                keras.layers.Dense(128, activation='relu'),
                keras.layers.Dropout(0.3),
                keras.layers.Dense(64, activation='relu'),
                keras.layers.Dropout(0.2),
                keras.layers.Dense(32, activation='relu'),
                keras.layers.Dense(self.num_classes, activation='softmax')
            ])
            
            # Compile model
            self.model.compile(
                optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            
            # Initialize scaler
            self.scaler = StandardScaler()
            
            # Generate synthetic training data for initial model
            await self._generate_synthetic_data()
            
            logger.info("New model created and trained with synthetic data")
            
        except Exception as e:
            logger.error(f"Error creating model: {e}")
            raise
    
    async def _generate_synthetic_data(self):
        """Generate synthetic training data for initial model training"""
        try:
            # Generate synthetic hand landmark features for each ASL letter
            X_synthetic = []
            y_synthetic = []
            
            np.random.seed(42)  # For reproducible results
            
            for label_idx, letter in enumerate(self.asl_labels):
                # Generate multiple samples per letter
                for _ in range(50):  # 50 samples per letter
                    features = self._generate_synthetic_features_for_letter(letter)
                    X_synthetic.append(features)
                    y_synthetic.append(label_idx)
            
            X_synthetic = np.array(X_synthetic)
            y_synthetic = np.array(y_synthetic)
            
            # Fit scaler and transform features
            X_scaled = self.scaler.fit_transform(X_synthetic)
            
            # Train the model
            self.model.fit(
                X_scaled, y_synthetic,
                epochs=50,
                batch_size=32,
                validation_split=0.2,
                verbose=0
            )
            
            # Save the model and scaler
            await self._save_model()
            
            logger.info(f"Model trained on {len(X_synthetic)} synthetic samples")
            
        except Exception as e:
            logger.error(f"Error generating synthetic data: {e}")
            raise
    
    def _generate_synthetic_features_for_letter(self, letter: str) -> List[float]:
        """Generate synthetic hand features for a specific ASL letter"""
        # This is a simplified approach - in a real implementation,
        # you would use actual hand landmark data from ASL datasets
        
        base_features = {
            'A': [0.15, 0.18, 0.20, 0.18, 0.15, 0.3, 0.2, 0.2, 0.3, 0.0, 0.18],
            'B': [0.25, 0.30, 0.32, 0.30, 0.28, 0.1, 0.1, 0.1, 0.1, 0.2, 0.30],
            'C': [0.20, 0.25, 0.27, 0.25, 0.22, 1.5, 0.3, 0.3, 0.3, 0.5, 0.25],
            'D': [0.18, 0.28, 0.25, 0.20, 0.18, 0.5, 0.8, 0.3, 0.3, 0.3, 0.26],
            'E': [0.12, 0.15, 0.17, 0.16, 0.14, 0.2, 0.2, 0.2, 0.2, 0.1, 0.15],
            # Add more letters with distinctive feature patterns
        }
        
        # Get base features for the letter, or use default
        if letter in base_features:
            features = base_features[letter].copy()
        else:
            # Default features with some variation
            features = [0.20, 0.25, 0.27, 0.25, 0.22, 0.5, 0.4, 0.4, 0.4, 0.2, 0.25]
        
        # Add some random noise to make data more realistic
        noise_scale = 0.05
        for i in range(len(features)):
            features[i] += np.random.normal(0, noise_scale)
            features[i] = max(0.05, min(1.0, features[i]))  # Clamp values
        
        return features
    
    async def predict_sign(self, image: np.ndarray, target_sign: str = None) -> Dict:
        """
        Predict ASL sign from image
        
        Args:
            image: Input image as numpy array
            target_sign: Optional target sign for accuracy calculation
            
        Returns:
            Dictionary with prediction results
        """
        try:
            start_time = datetime.now()
            
            # Detect hands in the image
            detection_result = self.hand_detector.detect_hands(image)
            
            prediction_result = {
                "hands_detected": detection_result["hands_detected"],
                "num_hands": detection_result["num_hands"],
                "predicted_sign": None,
                "confidence": 0.0,
                "accuracy_score": 0.0,
                "hand_landmarks": detection_result["hand_landmarks"],
                "processing_time_ms": 0,
                "error": None
            }
            
            if not detection_result["hands_detected"]:
                prediction_result["error"] = "No hands detected in image"
                return prediction_result
            
            # Use the first detected hand for prediction
            hand_landmarks = detection_result["hand_landmarks"][0]
            
            # Extract features from hand landmarks
            features = self.hand_detector.get_hand_features(hand_landmarks)
            
            if not features:
                prediction_result["error"] = "Could not extract features from hand landmarks"
                return prediction_result
            
            # Convert features to numpy array in correct order
            feature_vector = np.array([features.get(name, 0.0) for name in self.feature_names])
            feature_vector = feature_vector.reshape(1, -1)
            
            # Scale features
            if self.scaler:
                feature_vector = self.scaler.transform(feature_vector)
            
            # Make prediction
            predictions = self.model.predict(feature_vector, verbose=0)
            predicted_class = np.argmax(predictions[0])
            confidence = float(predictions[0][predicted_class])
            
            predicted_sign = self.asl_labels[predicted_class]
            prediction_result["predicted_sign"] = predicted_sign
            prediction_result["confidence"] = confidence
            
            # Calculate accuracy if target sign is provided
            if target_sign:
                accuracy_score = self.accuracy_scorer.calculate_accuracy(
                    predicted_sign, target_sign, confidence, hand_landmarks
                )
                prediction_result["accuracy_score"] = accuracy_score
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            prediction_result["processing_time_ms"] = processing_time
            
            return prediction_result
            
        except Exception as e:
            logger.error(f"Error in sign prediction: {e}")
            return {
                "hands_detected": False,
                "num_hands": 0,
                "predicted_sign": None,
                "confidence": 0.0,
                "accuracy_score": 0.0,
                "hand_landmarks": [],
                "processing_time_ms": 0,
                "error": str(e)
            }
    
    async def train_with_new_data(self, features: List[List[float]], labels: List[str]):
        """
        Retrain model with new data
        
        Args:
            features: List of feature vectors
            labels: List of corresponding labels
        """
        try:
            if not features or not labels or len(features) != len(labels):
                raise ValueError("Invalid training data provided")
            
            # Convert labels to indices
            label_indices = [self.asl_labels.index(label) for label in labels if label in self.asl_labels]
            valid_features = [features[i] for i, label in enumerate(labels) if label in self.asl_labels]
            
            if not valid_features:
                raise ValueError("No valid training data found")
            
            X_new = np.array(valid_features)
            y_new = np.array(label_indices)
            
            # Scale new features
            X_scaled = self.scaler.transform(X_new)
            
            # Fine-tune the model
            self.model.fit(
                X_scaled, y_new,
                epochs=10,
                batch_size=16,
                verbose=0
            )
            
            # Save updated model
            await self._save_model()
            
            logger.info(f"Model retrained with {len(X_new)} new samples")
            
        except Exception as e:
            logger.error(f"Error retraining model: {e}")
            raise
    
    async def _save_model(self):
        """Save the trained model and scaler"""
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            
            # Save model
            self.model.save(self.model_path)
            
            # Save scaler
            scaler_path = self.model_path.replace('.h5', '_scaler.pkl')
            with open(scaler_path, 'wb') as f:
                pickle.dump(self.scaler, f)
            
            logger.info(f"Model and scaler saved to {self.model_path}")
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            raise
    
    def get_model_info(self) -> Dict:
        """Get information about the current model"""
        return {
            "model_path": self.model_path,
            "num_classes": self.num_classes,
            "supported_signs": self.asl_labels,
            "feature_names": self.feature_names,
            "model_loaded": self.model is not None,
            "scaler_loaded": self.scaler is not None
        }
    
    def close(self):
        """Clean up resources"""
        if self.hand_detector:
            self.hand_detector.close()
