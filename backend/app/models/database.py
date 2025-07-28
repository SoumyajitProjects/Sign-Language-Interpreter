from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean, Text, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime
import os

# Database configuration
import os

# Ensure data directory exists
os.makedirs("data", exist_ok=True)
DATABASE_URL = "sqlite:///./data/signs.db"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

class Sign(Base):
    """Model for ASL signs"""
    __tablename__ = "signs"

    id = Column(Integer, primary_key=True, index=True)
    letter = Column(String(1), unique=True, index=True, nullable=False)
    name = Column(String(50), nullable=False)
    description = Column(Text)
    difficulty_level = Column(Integer, default=1)  # 1-5 scale
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    detection_sessions = relationship("DetectionSession", back_populates="target_sign")

class DetectionSession(Base):
    """Model for detection sessions"""
    __tablename__ = "detection_sessions"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String(50), unique=True, index=True, nullable=False)
    target_sign_id = Column(Integer, ForeignKey("signs.id"))
    user_id = Column(String(50), nullable=True)  # Optional user identification
    start_time = Column(DateTime, default=datetime.utcnow)
    end_time = Column(DateTime, nullable=True)
    total_attempts = Column(Integer, default=0)
    successful_attempts = Column(Integer, default=0)
    average_accuracy = Column(Float, default=0.0)
    best_accuracy = Column(Float, default=0.0)
    is_active = Column(Boolean, default=True)
    
    # Relationships
    target_sign = relationship("Sign", back_populates="detection_sessions")
    detections = relationship("Detection", back_populates="session")

class Detection(Base):
    """Model for individual sign detections"""
    __tablename__ = "detections"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("detection_sessions.id"))
    detected_sign = Column(String(1), nullable=False)
    confidence = Column(Float, nullable=False)
    accuracy_score = Column(Float, nullable=False)
    hand_landmarks = Column(Text)  # JSON string of hand landmarks
    timestamp = Column(DateTime, default=datetime.utcnow)
    processing_time_ms = Column(Float)
    
    # Relationships
    session = relationship("DetectionSession", back_populates="detections")

class UserStats(Base):
    """Model for user statistics"""
    __tablename__ = "user_stats"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String(50), unique=True, index=True, nullable=False)
    total_sessions = Column(Integer, default=0)
    total_detections = Column(Integer, default=0)
    average_accuracy = Column(Float, default=0.0)
    favorite_sign = Column(String(1), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_active = Column(DateTime, default=datetime.utcnow)

# Additional classes for router compatibility
class Session:
    """Session model for router compatibility"""
    def __init__(self, id=None, start_time=None, end_time=None):
        self.id = id
        self.start_time = start_time
        self.end_time = end_time
    
    @staticmethod
    def create():
        """Create a new session"""
        import uuid
        session_id = str(uuid.uuid4())
        db = SessionLocal()
        try:
            session = DetectionSession(
                session_id=session_id,
                start_time=datetime.utcnow()
            )
            db.add(session)
            db.commit()
            db.refresh(session)
            return Session(id=session_id, start_time=session.start_time)
        finally:
            db.close()
    
    @staticmethod
    def get_by_id(session_id):
        """Get session by ID"""
        db = SessionLocal()
        try:
            session = db.query(DetectionSession).filter(DetectionSession.session_id == session_id).first()
            if session:
                return Session(id=session.session_id, start_time=session.start_time, end_time=session.end_time)
            return None
        finally:
            db.close()

class SignDetection:
    """SignDetection model for router compatibility"""
    def __init__(self, session_id=None, detected_sign=None, confidence=None, accuracy=None, hand_landmarks=None, timestamp=None):
        self.session_id = session_id
        self.detected_sign = detected_sign
        self.confidence = confidence
        self.accuracy = accuracy
        self.hand_landmarks = hand_landmarks
        self.timestamp = timestamp
    
    @staticmethod
    def get_by_session(session_id):
        """Get detections by session ID"""
        db = SessionLocal()
        try:
            detections = db.query(Detection).join(DetectionSession).filter(DetectionSession.session_id == session_id).all()
            return [SignDetection(
                session_id=d.session_id,
                detected_sign=d.detected_sign,
                confidence=d.confidence,
                accuracy=d.accuracy_score,
                hand_landmarks=d.hand_landmarks,
                timestamp=d.timestamp
            ) for d in detections]
        finally:
            db.close()
    
    @staticmethod
    def get_total_count():
        """Get total number of detections"""
        db = SessionLocal()
        try:
            return db.query(Detection).count()
        finally:
            db.close()
    
    @staticmethod
    def get_recent_detections(limit=100):
        """Get recent detections"""
        db = SessionLocal()
        try:
            detections = db.query(Detection).order_by(Detection.timestamp.desc()).limit(limit).all()
            return [SignDetection(
                session_id=d.session_id,
                detected_sign=d.detected_sign,
                confidence=d.confidence,
                accuracy=d.accuracy_score,
                hand_landmarks=d.hand_landmarks,
                timestamp=d.timestamp
            ) for d in detections]
        finally:
            db.close()

def create_tables():
    """Create all database tables"""
    Base.metadata.create_all(bind=engine)
    init_default_signs()  # Initialize with default signs

# Add methods to Sign class for router compatibility
def get_all_signs():
    """Get all signs"""
    db = SessionLocal()
    try:
        return db.query(Sign).all()
    finally:
        db.close()

def get_sign_by_id(sign_id):
    """Get sign by ID"""
    db = SessionLocal()
    try:
        return db.query(Sign).filter(Sign.id == sign_id).first()
    finally:
        db.close()

def get_sign_by_letter(letter):
    """Get sign by letter"""
    db = SessionLocal()
    try:
        return db.query(Sign).filter(Sign.letter == letter).first()
    finally:
        db.close()

def get_signs_by_category(category):
    """Get signs by category (placeholder - all signs for now)"""
    db = SessionLocal()
    try:
        return db.query(Sign).all()
    finally:
        db.close()

# Add these methods to Sign class
Sign.get_all = staticmethod(get_all_signs)
Sign.get_by_id = staticmethod(get_sign_by_id)
Sign.get_by_letter = staticmethod(get_sign_by_letter)
Sign.get_by_category = staticmethod(get_signs_by_category)

def get_db():
    """Dependency to get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def init_default_signs():
    """Initialize the database with ASL alphabet signs"""
    db = SessionLocal()
    
    # Check if signs already exist
    if db.query(Sign).count() > 0:
        db.close()
        return
    
    # ASL alphabet signs with descriptions
    asl_signs = [
        ("A", "Letter A", "Closed fist with thumb pointing up"),
        ("B", "Letter B", "Four fingers pointing up, thumb folded across palm"),
        ("C", "Letter C", "Curved hand forming the letter C shape"),
        ("D", "Letter D", "Index finger pointing up, other fingers and thumb form O"),
        ("E", "Letter E", "All fingers curled down, touching thumb"),
        ("F", "Letter F", "Index and thumb form circle, other fingers point up"),
        ("G", "Letter G", "Index finger and thumb point horizontally"),
        ("H", "Letter H", "Index and middle finger point horizontally"),
        ("I", "Letter I", "Pinky finger pointing up, other fingers in fist"),
        ("J", "Letter J", "Pinky finger draws J shape in air"),
        ("K", "Letter K", "Index and middle finger up, thumb between them"),
        ("L", "Letter L", "Thumb and index finger form L shape"),
        ("M", "Letter M", "Three fingers over thumb"),
        ("N", "Letter N", "Two fingers over thumb"),
        ("O", "Letter O", "All fingers form circle with thumb"),
        ("P", "Letter P", "Same as K but pointing down"),
        ("Q", "Letter Q", "Thumb and index finger point down"),
        ("R", "Letter R", "Index and middle finger crossed"),
        ("S", "Letter S", "Closed fist with thumb over fingers"),
        ("T", "Letter T", "Thumb between index and middle finger"),
        ("U", "Letter U", "Index and middle finger pointing up together"),
        ("V", "Letter V", "Index and middle finger apart, pointing up"),
        ("W", "Letter W", "Three fingers pointing up"),
        ("X", "Letter X", "Index finger curled, pointing sideways"),
        ("Y", "Letter Y", "Thumb and pinky extended, other fingers folded"),
        ("Z", "Letter Z", "Index finger draws Z shape in air"),
    ]
    
    for letter, name, description in asl_signs:
        sign = Sign(
            letter=letter,
            name=name,
            description=description,
            difficulty_level=1 if letter in "AEIOU" else 2  # Vowels are easier
        )
        db.add(sign)
    
    db.commit()
    db.close()

if __name__ == "__main__":
    create_tables()
    init_default_signs()
    print("Database initialized with ASL signs!")
