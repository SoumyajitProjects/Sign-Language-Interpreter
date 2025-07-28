from fastapi import APIRouter, HTTPException
import logging
from typing import List, Dict, Any

from ..models.database import Sign, SignDetection

logger = logging.getLogger(__name__)
router = APIRouter()

@router.get("/signs")
async def get_signs():
    """Get list of all supported signs"""
    try:
        signs = Sign.get_all()
        return {
            "signs": [
                {
                    "id": sign.id,
                    "letter": sign.letter,
                    "description": sign.description,
                    "category": sign.category,
                    "difficulty": sign.difficulty
                }
                for sign in signs
            ],
            "total": len(signs)
        }
    except Exception as e:
        logger.error(f"Error getting signs: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve signs")

@router.get("/signs/{sign_id}")
async def get_sign(sign_id: str):
    """Get detailed information about a specific sign"""
    try:
        sign = Sign.get_by_id(sign_id)
        if not sign:
            raise HTTPException(status_code=404, detail="Sign not found")
        
        return {
            "id": sign.id,
            "letter": sign.letter,
            "description": sign.description,
            "category": sign.category,
            "difficulty": sign.difficulty,
            "hand_position": sign.hand_position,
            "fingers_extended": sign.fingers_extended,
            "created_at": sign.created_at.isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting sign {sign_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve sign")

@router.get("/signs/letter/{letter}")
async def get_sign_by_letter(letter: str):
    """Get sign information by letter"""
    try:
        sign = Sign.get_by_letter(letter.upper())
        if not sign:
            raise HTTPException(status_code=404, detail=f"Sign for letter '{letter}' not found")
        
        return {
            "id": sign.id,
            "letter": sign.letter,
            "description": sign.description,
            "category": sign.category,
            "difficulty": sign.difficulty,
            "hand_position": sign.hand_position,
            "fingers_extended": sign.fingers_extended
        }
    except Exception as e:
        logger.error(f"Error getting sign for letter {letter}: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve sign")

@router.get("/signs/category/{category}")
async def get_signs_by_category(category: str):
    """Get all signs in a specific category"""
    try:
        signs = Sign.get_by_category(category)
        return {
            "category": category,
            "signs": [
                {
                    "id": sign.id,
                    "letter": sign.letter,
                    "description": sign.description,
                    "difficulty": sign.difficulty
                }
                for sign in signs
            ],
            "total": len(signs)
        }
    except Exception as e:
        logger.error(f"Error getting signs for category {category}: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve signs")

@router.get("/signs/stats")
async def get_sign_statistics():
    """Get statistics about sign detection"""
    try:
        # Get detection statistics
        total_detections = SignDetection.get_total_count()
        recent_detections = SignDetection.get_recent_detections(limit=100)
        
        # Calculate most detected signs
        sign_counts = {}
        for detection in recent_detections:
            sign_counts[detection.detected_sign] = sign_counts.get(detection.detected_sign, 0) + 1
        
        most_detected = sorted(sign_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        
        return {
            "total_detections": total_detections,
            "recent_detections": len(recent_detections),
            "most_detected_signs": [
                {"sign": sign, "count": count} for sign, count in most_detected
            ],
            "supported_signs": len(Sign.get_all())
        }
    except Exception as e:
        logger.error(f"Error getting sign statistics: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve sign statistics") 