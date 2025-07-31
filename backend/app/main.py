from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import logging
import cv2
import mediapipe as mp
import numpy as np
import base64
from datetime import datetime
import math

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Advanced Sign Language Detection",
    description="Real-time hand detection and sign recognition with phrases",
    version="2.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Global hands detector
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_personal_message(self, message: dict, websocket: WebSocket):
        await websocket.send_json(message)

manager = ConnectionManager()

@app.websocket("/ws/detection")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time hand detection"""
    await manager.connect(websocket)
    logger.info("WebSocket client connected")
    
    try:
        while True:
            # Receive image data from client
            data = await websocket.receive_json()
            
            if data.get("type") == "frame":
                # Get base64 image data
                image_data = data.get("image", "")
                
                if image_data:
                    # Decode base64 image
                    image_bytes = base64.b64decode(image_data.split(',')[1])
                    image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
                    
                    # Convert BGR to RGB
                    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    
                    # Process with MediaPipe
                    results = hands.process(rgb_image)
                    
                    # Prepare response
                    response = {
                        "type": "detection_result",
                        "hands_detected": False,
                        "landmarks": [],
                        "sign": None,
                        "phrase": None,
                        "confidence": 0.0,
                        "accuracy": 0.0,
                        "timestamp": datetime.now().isoformat()
                    }
                    
                    if results.multi_hand_landmarks:
                        response["hands_detected"] = True
                        
                        # Get landmarks from first hand
                        hand_landmarks = results.multi_hand_landmarks[0]
                        
                        # Convert landmarks to list format
                        landmarks = []
                        for landmark in hand_landmarks.landmark:
                            landmarks.append({
                                "x": landmark.x,
                                "y": landmark.y,
                                "z": landmark.z
                            })
                        
                        response["landmarks"] = landmarks
                        
                        # Advanced sign detection
                        sign, phrase, confidence, accuracy = detect_sign_and_phrase(landmarks)
                        response["sign"] = sign
                        response["phrase"] = phrase
                        response["confidence"] = confidence
                        response["accuracy"] = accuracy
                    
                    # Send response back
                    await manager.send_personal_message(response, websocket)
                
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        logger.info("WebSocket connection closed")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)

def calculate_distance(point1, point2):
    """Calculate Euclidean distance between two points"""
    return math.sqrt((point1["x"] - point2["x"])**2 + (point1["y"] - point2["y"])**2)

def calculate_angle(point1, point2, point3):
    """Calculate angle between three points"""
    a = calculate_distance(point1, point2)
    b = calculate_distance(point2, point3)
    c = calculate_distance(point1, point3)
    
    if a == 0 or b == 0:
        return 0
    
    cos_angle = (a**2 + b**2 - c**2) / (2 * a * b)
    cos_angle = max(-1, min(1, cos_angle))  # Clamp to [-1, 1]
    return math.degrees(math.acos(cos_angle))

def detect_sign_and_phrase(landmarks):
    """Advanced sign detection with phrase recognition"""
    if not landmarks or len(landmarks) < 21:
        return None, None, 0.0, 0.0
    
    # Get key landmark positions
    wrist = landmarks[0]
    thumb_tip = landmarks[4]
    thumb_ip = landmarks[3]
    thumb_mcp = landmarks[2]
    index_tip = landmarks[8]
    index_pip = landmarks[6]
    index_mcp = landmarks[5]
    middle_tip = landmarks[12]
    middle_pip = landmarks[10]
    middle_mcp = landmarks[9]
    ring_tip = landmarks[16]
    ring_pip = landmarks[14]
    ring_mcp = landmarks[13]
    pinky_tip = landmarks[20]
    pinky_pip = landmarks[18]
    pinky_mcp = landmarks[17]
    
    # Calculate finger extension status
    def is_finger_extended(tip, pip, mcp):
        tip_y = tip["y"]
        pip_y = pip["y"]
        mcp_y = mcp["y"]
        return tip_y < pip_y < mcp_y
    
    def is_thumb_extended():
        # Thumb is extended if tip is above IP joint
        return thumb_tip["y"] < thumb_ip["y"]
    
    # Check finger extensions
    index_extended = is_finger_extended(index_tip, index_pip, index_mcp)
    middle_extended = is_finger_extended(middle_tip, middle_pip, middle_mcp)
    ring_extended = is_finger_extended(ring_tip, ring_pip, ring_mcp)
    pinky_extended = is_finger_extended(pinky_tip, pinky_pip, pinky_mcp)
    thumb_extended = is_thumb_extended()
    
    # Count extended fingers
    extended_fingers = [index_extended, middle_extended, ring_extended, pinky_extended]
    num_extended = sum(extended_fingers)
    
    # Calculate hand orientation and position
    palm_center_x = (index_mcp["x"] + middle_mcp["x"] + ring_mcp["x"] + pinky_mcp["x"]) / 4
    palm_center_y = (index_mcp["y"] + middle_mcp["y"] + ring_mcp["y"] + pinky_mcp["y"]) / 4
    
    # Calculate distances for more precise detection
    thumb_index_distance = calculate_distance(thumb_tip, index_tip)
    index_middle_distance = calculate_distance(index_tip, middle_tip)
    
    # Advanced sign detection logic
    sign, phrase, confidence, accuracy = detect_advanced_signs(
        extended_fingers, thumb_extended, num_extended,
        thumb_index_distance, index_middle_distance,
        landmarks
    )
    
    return sign, phrase, confidence, accuracy

def detect_advanced_signs(extended_fingers, thumb_extended, num_extended, 
                         thumb_index_distance, index_middle_distance, landmarks):
    """Advanced sign detection with multiple criteria"""
    
    [index_extended, middle_extended, ring_extended, pinky_extended] = extended_fingers
    
    # Most Specific Signs First - Check these FIRST to avoid conflicts
    if detect_i_love_you(landmarks):
        return "I", "I Love You", 0.95, 0.94
    
    elif detect_ok_sign(landmarks):
        return "👌", "OK", 0.9, 0.87
    
    elif detect_rock_on(landmarks):
        return "🤘", "Rock On", 0.85, 0.83
    
    elif detect_peace_sign(landmarks):
        return "✌️", "Peace", 0.95, 0.92
    
    # Number Detection - More specific checks
    elif detect_number_one(landmarks):
        return "1️⃣", "Number One", 0.9, 0.88
    
    elif detect_number_two(landmarks):
        return "2️⃣", "Number Two", 0.9, 0.87
    
    elif detect_number_three(landmarks):
        return "3️⃣", "Number Three", 0.9, 0.86
    
    elif detect_number_four(landmarks):
        return "4️⃣", "Number Four", 0.9, 0.85
    
    # Simple gestures last (most likely to cause false positives)
    elif detect_thumbs_up(landmarks):
        return "👍", "Thumbs Up", 0.9, 0.88
    
    # ASL Alphabet Detection - More comprehensive
    if num_extended == 0 and not thumb_extended:
        return "A", "A", 0.9, 0.85  # Fist
    
    elif num_extended == 1 and index_extended and not thumb_extended:
        return "B", "B", 0.9, 0.88  # Index finger up
    
    elif num_extended == 1 and middle_extended and not index_extended:
        return "D", "D", 0.85, 0.82  # Middle finger up
    
    elif num_extended == 2 and index_extended and middle_extended:
        return "V", "V", 0.9, 0.88  # V sign
    
    elif num_extended == 3 and index_extended and middle_extended and ring_extended:
        return "W", "W", 0.9, 0.87  # Three fingers
    
    elif num_extended == 4 and not thumb_extended:
        return "5", "Five", 0.95, 0.93  # All fingers extended
    
    elif thumb_extended and not any(extended_fingers):
        return "L", "L", 0.85, 0.83  # Thumb and index form L
    
    elif num_extended == 1 and pinky_extended and not thumb_extended:
        return "I", "I", 0.8, 0.78  # Pinky up (only if thumb not extended)
    
    elif num_extended == 2 and index_extended and pinky_extended and not thumb_extended:
        return "Y", "Y", 0.85, 0.82  # Index and pinky (only if thumb not extended)
    
    elif num_extended == 3 and index_extended and middle_extended and pinky_extended:
        return "3", "Three", 0.8, 0.77  # Three specific fingers
    
    elif num_extended == 2 and middle_extended and ring_extended:
        return "U", "U", 0.8, 0.76  # Middle and ring
    
    elif num_extended == 1 and ring_extended:
        return "R", "R", 0.75, 0.72  # Ring finger only
    
    elif num_extended == 2 and index_extended and ring_extended:
        return "H", "H", 0.8, 0.76  # Index and ring
    
    elif num_extended == 2 and index_extended and pinky_extended:
        return "N", "N", 0.8, 0.76  # Index and pinky
    
    elif num_extended == 3 and index_extended and middle_extended and ring_extended:
        return "M", "M", 0.8, 0.77  # Three fingers down
    
    elif num_extended == 1 and thumb_extended and index_extended:
        return "F", "F", 0.8, 0.76  # Thumb and index
    
    elif num_extended == 2 and thumb_extended and index_extended:
        return "G", "G", 0.8, 0.76  # Thumb and index extended
    
    elif num_extended == 3 and thumb_extended and index_extended and middle_extended:
        return "H", "H", 0.8, 0.76  # Thumb, index, middle
    
    elif num_extended == 4 and thumb_extended:
        return "4", "Four", 0.85, 0.82  # All fingers including thumb
    
    elif num_extended == 5:
        return "5", "Five", 0.95, 0.93  # All fingers
    
    else:
        # Calculate confidence based on how close we are to known signs
        confidence = calculate_confidence_score(extended_fingers, thumb_extended, landmarks)
        return "Unknown", "Unknown", confidence, confidence * 0.8

def detect_i_love_you(landmarks):
    """Detect 'I Love You' sign (thumb, index, and pinky extended)"""
    if len(landmarks) < 21:
        return False
    
    thumb_tip = landmarks[4]
    thumb_ip = landmarks[3]
    thumb_mcp = landmarks[2]
    index_tip = landmarks[8]
    index_pip = landmarks[6]
    index_mcp = landmarks[5]
    middle_tip = landmarks[12]
    middle_pip = landmarks[10]
    middle_mcp = landmarks[9]
    ring_tip = landmarks[16]
    ring_pip = landmarks[14]
    ring_mcp = landmarks[13]
    pinky_tip = landmarks[20]
    pinky_pip = landmarks[18]
    pinky_mcp = landmarks[17]
    
    # More precise finger extension detection
    def is_finger_extended(tip, pip, mcp):
        tip_y = tip["y"]
        pip_y = pip["y"]
        mcp_y = mcp["y"]
        return tip_y < pip_y < mcp_y
    
    # Check if thumb, index, and pinky are extended, middle and ring are NOT extended
    thumb_extended = thumb_tip["y"] < thumb_ip["y"]
    index_extended = is_finger_extended(index_tip, index_pip, index_mcp)
    middle_extended = is_finger_extended(middle_tip, middle_pip, middle_mcp)
    ring_extended = is_finger_extended(ring_tip, ring_pip, ring_mcp)
    pinky_extended = is_finger_extended(pinky_tip, pinky_pip, pinky_mcp)
    
    # Additional check: thumb should be clearly extended and pointing outward
    thumb_clearly_extended = thumb_tip["y"] < thumb_ip["y"] and thumb_tip["x"] > thumb_mcp["x"]
    
    return (thumb_clearly_extended and 
            index_extended and 
            not middle_extended and 
            not ring_extended and 
            pinky_extended)

def detect_thumbs_up(landmarks):
    """Detect thumbs up gesture"""
    if len(landmarks) < 21:
        return False
    
    thumb_tip = landmarks[4]
    thumb_ip = landmarks[3]
    thumb_mcp = landmarks[2]
    wrist = landmarks[0]
    
    # Get other finger positions
    index_tip = landmarks[8]
    index_pip = landmarks[6]
    index_mcp = landmarks[5]
    middle_tip = landmarks[12]
    middle_pip = landmarks[10]
    middle_mcp = landmarks[9]
    ring_tip = landmarks[16]
    ring_pip = landmarks[14]
    ring_mcp = landmarks[13]
    pinky_tip = landmarks[20]
    pinky_pip = landmarks[18]
    pinky_mcp = landmarks[17]
    
    def is_finger_folded(tip, pip, mcp):
        # For a folded finger, tip should be below or close to pip
        return tip["y"] >= pip["y"] - 0.02
    
    # Thumb should be pointing up (y coordinate decreasing)
    thumb_extended = thumb_tip["y"] < thumb_ip["y"] < thumb_mcp["y"]
    thumb_above_wrist = thumb_tip["y"] < wrist["y"]
    
    # Other fingers should be folded down
    index_folded = is_finger_folded(index_tip, index_pip, index_mcp)
    middle_folded = is_finger_folded(middle_tip, middle_pip, middle_mcp)
    ring_folded = is_finger_folded(ring_tip, ring_pip, ring_mcp)
    pinky_folded = is_finger_folded(pinky_tip, pinky_pip, pinky_mcp)
    
    return (thumb_extended and 
            thumb_above_wrist and 
            index_folded and 
            middle_folded and 
            ring_folded and 
            pinky_folded)

def detect_ok_sign(landmarks):
    """Detect OK sign (thumb and index form circle)"""
    if len(landmarks) < 21:
        return False
    
    thumb_tip = landmarks[4]
    index_tip = landmarks[8]
    middle_tip = landmarks[12]
    ring_tip = landmarks[16]
    pinky_tip = landmarks[20]
    
    # Calculate distances
    thumb_index_dist = calculate_distance(thumb_tip, index_tip)
    thumb_middle_dist = calculate_distance(thumb_tip, middle_tip)
    thumb_ring_dist = calculate_distance(thumb_tip, ring_tip)
    thumb_pinky_dist = calculate_distance(thumb_tip, pinky_tip)
    
    # Thumb and index should be close (forming circle), others should be far
    return (thumb_index_dist < 0.05 and 
            thumb_middle_dist > 0.1 and 
            thumb_ring_dist > 0.1 and 
            thumb_pinky_dist > 0.1)

def detect_peace_sign(landmarks):
    """Detect peace sign (index and middle fingers close together)"""
    if len(landmarks) < 21:
        return False
    
    index_tip = landmarks[8]
    index_pip = landmarks[6]
    index_mcp = landmarks[5]
    middle_tip = landmarks[12]
    middle_pip = landmarks[10]
    middle_mcp = landmarks[9]
    ring_tip = landmarks[16]
    ring_pip = landmarks[14]
    ring_mcp = landmarks[13]
    pinky_tip = landmarks[20]
    pinky_pip = landmarks[18]
    pinky_mcp = landmarks[17]
    
    def is_finger_extended(tip, pip, mcp):
        tip_y = tip["y"]
        pip_y = pip["y"]
        mcp_y = mcp["y"]
        return tip_y < pip_y < mcp_y
    
    # Check if index and middle are extended, others are not
    index_extended = is_finger_extended(index_tip, index_pip, index_mcp)
    middle_extended = is_finger_extended(middle_tip, middle_pip, middle_mcp)
    ring_extended = is_finger_extended(ring_tip, ring_pip, ring_mcp)
    pinky_extended = is_finger_extended(pinky_tip, pinky_pip, pinky_mcp)
    
    # Calculate distance between index and middle fingertips
    index_middle_distance = calculate_distance(index_tip, middle_tip)
    
    # Peace sign: index and middle extended, close together, others not extended
    return (index_extended and 
            middle_extended and 
            not ring_extended and 
            not pinky_extended and 
            index_middle_distance < 0.08)  # Fingers should be close together

def detect_rock_on(landmarks):
    """Detect rock on gesture (index and pinky extended)"""
    if len(landmarks) < 21:
        return False
    
    thumb_tip = landmarks[4]
    thumb_ip = landmarks[3]
    thumb_mcp = landmarks[2]
    index_tip = landmarks[8]
    index_pip = landmarks[6]
    index_mcp = landmarks[5]
    middle_tip = landmarks[12]
    middle_pip = landmarks[10]
    middle_mcp = landmarks[9]
    ring_tip = landmarks[16]
    ring_pip = landmarks[14]
    ring_mcp = landmarks[13]
    pinky_tip = landmarks[20]
    pinky_pip = landmarks[18]
    pinky_mcp = landmarks[17]
    
    def is_finger_extended(tip, pip, mcp):
        tip_y = tip["y"]
        pip_y = pip["y"]
        mcp_y = mcp["y"]
        return tip_y < pip_y < mcp_y
    
    def is_finger_folded(tip, pip, mcp):
        # For a folded finger, tip should be below or close to pip
        return tip["y"] >= pip["y"] - 0.02
    
    def is_thumb_folded():
        # Thumb is folded if tip is not significantly above IP joint
        return thumb_tip["y"] >= thumb_ip["y"] - 0.02
    
    # Check if index and pinky are extended, middle and ring are folded
    index_extended = is_finger_extended(index_tip, index_pip, index_mcp)
    middle_folded = is_finger_folded(middle_tip, middle_pip, middle_mcp)
    ring_folded = is_finger_folded(ring_tip, ring_pip, ring_mcp)
    pinky_extended = is_finger_extended(pinky_tip, pinky_pip, pinky_mcp)
    thumb_folded = is_thumb_folded()
    
    # Calculate distance between extended fingers to ensure they're properly separated
    finger_separation = calculate_distance(index_tip, pinky_tip)
    
    return (index_extended and 
            pinky_extended and 
            middle_folded and 
            ring_folded and 
            thumb_folded and
            finger_separation > 0.08)  # Ensure fingers are properly separated

def detect_number_one(landmarks):
    """Detect number 1 (index finger extended)"""
    if len(landmarks) < 21:
        return False
    
    index_tip = landmarks[8]
    index_pip = landmarks[6]
    index_mcp = landmarks[5]
    middle_tip = landmarks[12]
    middle_pip = landmarks[10]
    middle_mcp = landmarks[9]
    ring_tip = landmarks[16]
    ring_pip = landmarks[14]
    ring_mcp = landmarks[13]
    pinky_tip = landmarks[20]
    pinky_pip = landmarks[18]
    pinky_mcp = landmarks[17]
    thumb_tip = landmarks[4]
    thumb_ip = landmarks[3]
    
    def is_finger_extended(tip, pip, mcp):
        return tip["y"] < pip["y"] < mcp["y"]
    
    def is_thumb_extended():
        return thumb_tip["y"] < thumb_ip["y"]
    
    # Only index finger should be extended
    index_extended = is_finger_extended(index_tip, index_pip, index_mcp)
    middle_extended = is_finger_extended(middle_tip, middle_pip, middle_mcp)
    ring_extended = is_finger_extended(ring_tip, ring_pip, ring_mcp)
    pinky_extended = is_finger_extended(pinky_tip, pinky_pip, pinky_mcp)
    thumb_extended = is_thumb_extended()
    
    return (index_extended and 
            not middle_extended and 
            not ring_extended and 
            not pinky_extended and 
            not thumb_extended)

def detect_number_two(landmarks):
    """Detect number 2 (index and middle fingers extended)"""
    if len(landmarks) < 21:
        return False
    
    index_tip = landmarks[8]
    index_pip = landmarks[6]
    index_mcp = landmarks[5]
    middle_tip = landmarks[12]
    middle_pip = landmarks[10]
    middle_mcp = landmarks[9]
    ring_tip = landmarks[16]
    ring_pip = landmarks[14]
    ring_mcp = landmarks[13]
    pinky_tip = landmarks[20]
    pinky_pip = landmarks[18]
    pinky_mcp = landmarks[17]
    thumb_tip = landmarks[4]
    thumb_ip = landmarks[3]
    
    def is_finger_extended(tip, pip, mcp):
        return tip["y"] < pip["y"] < mcp["y"]
    
    def is_thumb_extended():
        return thumb_tip["y"] < thumb_ip["y"]
    
    # Index and middle should be extended, others not
    index_extended = is_finger_extended(index_tip, index_pip, index_mcp)
    middle_extended = is_finger_extended(middle_tip, middle_pip, middle_mcp)
    ring_extended = is_finger_extended(ring_tip, ring_pip, ring_mcp)
    pinky_extended = is_finger_extended(pinky_tip, pinky_pip, pinky_mcp)
    thumb_extended = is_thumb_extended()
    
    # Check if fingers are reasonably separated (not too close together)
    finger_distance = calculate_distance(index_tip, middle_tip)
    
    return (index_extended and 
            middle_extended and 
            not ring_extended and 
            not pinky_extended and 
            not thumb_extended and
            finger_distance > 0.03)  # Fingers should be separated

def detect_number_three(landmarks):
    """Detect number 3 (index, middle, and ring fingers extended)"""
    if len(landmarks) < 21:
        return False
    
    index_tip = landmarks[8]
    index_pip = landmarks[6]
    index_mcp = landmarks[5]
    middle_tip = landmarks[12]
    middle_pip = landmarks[10]
    middle_mcp = landmarks[9]
    ring_tip = landmarks[16]
    ring_pip = landmarks[14]
    ring_mcp = landmarks[13]
    pinky_tip = landmarks[20]
    pinky_pip = landmarks[18]
    pinky_mcp = landmarks[17]
    thumb_tip = landmarks[4]
    thumb_ip = landmarks[3]
    
    def is_finger_extended(tip, pip, mcp):
        return tip["y"] < pip["y"] < mcp["y"]
    
    def is_thumb_extended():
        return thumb_tip["y"] < thumb_ip["y"]
    
    # Index, middle, and ring should be extended, pinky and thumb not
    index_extended = is_finger_extended(index_tip, index_pip, index_mcp)
    middle_extended = is_finger_extended(middle_tip, middle_pip, middle_mcp)
    ring_extended = is_finger_extended(ring_tip, ring_pip, ring_mcp)
    pinky_extended = is_finger_extended(pinky_tip, pinky_pip, pinky_mcp)
    thumb_extended = is_thumb_extended()
    
    return (index_extended and 
            middle_extended and 
            ring_extended and 
            not pinky_extended and 
            not thumb_extended)

def detect_number_four(landmarks):
    """Detect number 4 (all fingers except thumb extended)"""
    if len(landmarks) < 21:
        return False
    
    index_tip = landmarks[8]
    index_pip = landmarks[6]
    index_mcp = landmarks[5]
    middle_tip = landmarks[12]
    middle_pip = landmarks[10]
    middle_mcp = landmarks[9]
    ring_tip = landmarks[16]
    ring_pip = landmarks[14]
    ring_mcp = landmarks[13]
    pinky_tip = landmarks[20]
    pinky_pip = landmarks[18]
    pinky_mcp = landmarks[17]
    thumb_tip = landmarks[4]
    thumb_ip = landmarks[3]
    
    def is_finger_extended(tip, pip, mcp):
        return tip["y"] < pip["y"] < mcp["y"]
    
    def is_thumb_extended():
        return thumb_tip["y"] < thumb_ip["y"]
    
    # All fingers except thumb should be extended
    index_extended = is_finger_extended(index_tip, index_pip, index_mcp)
    middle_extended = is_finger_extended(middle_tip, middle_pip, middle_mcp)
    ring_extended = is_finger_extended(ring_tip, ring_pip, ring_mcp)
    pinky_extended = is_finger_extended(pinky_tip, pinky_pip, pinky_mcp)
    thumb_extended = is_thumb_extended()
    
    return (index_extended and 
            middle_extended and 
            ring_extended and 
            pinky_extended and 
            not thumb_extended)

def calculate_confidence_score(extended_fingers, thumb_extended, landmarks):
    """Calculate confidence score based on landmark positions"""
    if not landmarks:
        return 0.0
    
    # Base confidence on how clear the hand position is
    confidence = 0.5
    
    # Add confidence for clear finger positions
    if len(extended_fingers) == 4:
        confidence += 0.3
    elif len(extended_fingers) == 3:
        confidence += 0.25
    elif len(extended_fingers) == 2:
        confidence += 0.2
    elif len(extended_fingers) == 1:
        confidence += 0.15
    
    # Add confidence for thumb position
    if thumb_extended:
        confidence += 0.1
    
    return min(confidence, 1.0)

@app.get("/")
async def root():
    return {"message": "Advanced Sign Language Detection API"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8001,
        reload=True,
        log_level="info"
    )
