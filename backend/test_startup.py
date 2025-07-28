#!/usr/bin/env python3
"""
Test script to identify startup issues
"""

import sys
import os

def test_imports():
    """Test if all required modules can be imported"""
    print("Testing imports...")
    
    try:
        import fastapi
        print("✅ FastAPI imported successfully")
    except ImportError as e:
        print(f"❌ FastAPI import failed: {e}")
        return False
    
    try:
        import uvicorn
        print("✅ Uvicorn imported successfully")
    except ImportError as e:
        print(f"❌ Uvicorn import failed: {e}")
        return False
    
    try:
        import cv2
        print("✅ OpenCV imported successfully")
    except ImportError as e:
        print(f"❌ OpenCV import failed: {e}")
        return False
    
    try:
        import mediapipe
        print("✅ MediaPipe imported successfully")
    except ImportError as e:
        print(f"❌ MediaPipe import failed: {e}")
        return False
    
    try:
        import tensorflow
        print("✅ TensorFlow imported successfully")
    except ImportError as e:
        print(f"❌ TensorFlow import failed: {e}")
        return False
    
    try:
        import numpy
        print("✅ NumPy imported successfully")
    except ImportError as e:
        print(f"❌ NumPy import failed: {e}")
        return False
    
    return True

def test_database():
    """Test database setup"""
    print("\nTesting database...")
    
    try:
        from app.models.database import create_tables, DATABASE_URL
        print(f"✅ Database URL: {DATABASE_URL}")
        
        # Create data directory if it doesn't exist
        os.makedirs("data", exist_ok=True)
        print("✅ Data directory created/verified")
        
        # Test table creation
        create_tables()
        print("✅ Database tables created successfully")
        
    except Exception as e:
        print(f"❌ Database setup failed: {e}")
        return False
    
    return True

def test_app_import():
    """Test if the app can be imported"""
    print("\nTesting app import...")
    
    try:
        from app.main import app
        print("✅ App imported successfully")
    except Exception as e:
        print(f"❌ App import failed: {e}")
        return False
    
    return True

def main():
    """Run all tests"""
    print("🔍 Sign Language Detector - Startup Test")
    print("=" * 50)
    
    # Test imports
    if not test_imports():
        print("\n❌ Import test failed. Please install missing dependencies.")
        sys.exit(1)
    
    # Test database
    if not test_database():
        print("\n❌ Database test failed.")
        sys.exit(1)
    
    # Test app import
    if not test_app_import():
        print("\n❌ App import test failed.")
        sys.exit(1)
    
    print("\n✅ All tests passed! The app should start successfully.")
    print("\nTo start the server, run:")
    print("uvicorn app.main:app --reload --host 0.0.0.0 --port 8000")

if __name__ == "__main__":
    main() 