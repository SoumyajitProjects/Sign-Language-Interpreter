# 🖐️ Real-Time Sign Language Detection System

A full-stack, real-time sign language recognition system that uses computer vision and machine learning to detect and classify hand gestures captured via webcam. The system translates sign language into text and displays both the meaning and model confidence in real time.

---

## 🚀 Features

- Live webcam-based sign language detection
- Hand tracking using MediaPipe
- Deep learning-based sign classifier with TensorFlow/Keras
- Real-time accuracy feedback and predictions
- Modern React frontend with video feed and live output
- FastAPI backend with RESTful endpoints
- Local database (SQLite) to log recognized signs and sessions

---

## 🧰 Tech Stack

- **Frontend**: React, JavaScript, WebRTC, Axios
- **Backend**: Python, FastAPI, Uvicorn
- **Computer Vision**: OpenCV, MediaPipe
- **Machine Learning**: TensorFlow, Keras
- **Database**: SQLite (via SQLModel or SQLAlchemy)
- **Others**: Virtualenv, CORS, Pydantic


### 1. Backend Setup

```bash
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Start the backend server:
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8001
```

---

### 2. Frontend Setup

```bash
cd frontend
npm install
npm start
```

The app will open at `http://localhost:3000`.

## 📦 Requirements

### Backend

- Python 3.9+
- OpenCV
- MediaPipe
- FastAPI
- Uvicorn
- TensorFlow/Keras
- SQLModel or SQLAlchemy

### Frontend

- Node.js v18+
- React
- Axios
- WebRTC-compatible browser (Chrome/Firefox)

