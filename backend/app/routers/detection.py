from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
import logging
from typing import Dict, Any
import json
from datetime import datetime

from ..services.hand_detector import HandDetector
from ..services.sign_recognizer import SignRecognizer
from ..models.database import Session, SignDetection

logger = logging.getLogger(__name__)
router = APIRouter()

# Global instances (will be initialized in main.py)
hand_detector = None
sign_recognizer = None

@router.post("/detect-sign")
async def detect_sign(frame_data: Dict[str, Any]):
    """Process a video frame and detect signs"""
    try:
        # Extract frame data
        image_data = frame_data.get("image")
        session_id = frame_data.get("session_id")
        
        if not image_data:
            raise HTTPException(status_code=400, detail="No image data provided")
        
        # Process frame with hand detection
        hand_landmarks = await hand_detector.detect_hands(image_data)
        
        if not hand_landmarks:
            return {
                "detected_sign": None,
                "confidence": 0.0,
                "accuracy": 0.0,
                "hand_landmarks": [],
                "message": "No hands detected"
            }
        
        # Recognize sign
        sign_result = await sign_recognizer.recognize_sign(hand_landmarks)
        
        # Calculate accuracy
        accuracy = sign_recognizer.calculate_accuracy(hand_landmarks, sign_result["sign"])
        
        # Store detection in database
        detection = SignDetection(
            session_id=session_id,
            detected_sign=sign_result["sign"],
            confidence=sign_result["confidence"],
            accuracy=accuracy,
            hand_landmarks=json.dumps(hand_landmarks),
            timestamp=datetime.utcnow()
        )
        
        return {
            "detected_sign": sign_result["sign"],
            "confidence": sign_result["confidence"],
            "accuracy": accuracy,
            "hand_landmarks": hand_landmarks,
            "timestamp": detection.timestamp.isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error in sign detection: {str(e)}")
        raise HTTPException(status_code=500, detail="Sign detection failed")

@router.get("/sessions/{session_id}/stats")
async def get_session_stats(session_id: str):
    """Get statistics for a detection session"""
    try:
        # Get session data from database
        session = Session.get_by_id(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Calculate statistics
        detections = SignDetection.get_by_session(session_id)
        
        total_detections = len(detections)
        if total_detections == 0:
            return {
                "session_id": session_id,
                "total_detections": 0,
                "average_accuracy": 0.0,
                "most_detected_sign": None,
                "session_duration": 0
            }
        
        avg_accuracy = sum(d.accuracy for d in detections) / total_detections
        sign_counts = {}
        for detection in detections:
            sign_counts[detection.detected_sign] = sign_counts.get(detection.detected_sign, 0) + 1
        
        most_detected = max(sign_counts.items(), key=lambda x: x[1])[0] if sign_counts else None
        
        return {
            "session_id": session_id,
            "total_detections": total_detections,
            "average_accuracy": round(avg_accuracy, 3),
            "most_detected_sign": most_detected,
            "session_duration": (session.end_time - session.start_time).total_seconds() if session.end_time else 0
        }
        
    except Exception as e:
        logger.error(f"Error getting session stats: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get session statistics")

@router.post("/sessions")
async def create_session():
    """Create a new detection session"""
    try:
        session = Session.create()
        return {
            "session_id": session.id,
            "start_time": session.start_time.isoformat(),
            "status": "active"
        }
    except Exception as e:
        logger.error(f"Error creating session: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to create session") 