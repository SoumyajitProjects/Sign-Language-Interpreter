import React, { useState, useEffect, useRef } from 'react';
import Webcam from 'react-webcam';
import './App.css';

function App() {
  const webcamRef = useRef(null);
  const canvasRef = useRef(null);
  const [isDetecting, setIsDetecting] = useState(false);
  const [detectionResult, setDetectionResult] = useState(null);
  const [isConnected, setIsConnected] = useState(false);
  const wsRef = useRef(null);
  const animationRef = useRef(null);

  useEffect(() => {
    connectWebSocket();
    
    // Initialize canvas when component mounts
    if (canvasRef.current) {
      const canvas = canvasRef.current;
      const ctx = canvas.getContext('2d');
      ctx.clearRect(0, 0, canvas.width, canvas.height);
    }
    
    // Set up periodic canvas clearing to prevent stuck landmarks
    const canvasClearInterval = setInterval(() => {
      if (!isDetecting) {
        clearCanvas();
      }
    }, 1000); // Clear every second when not detecting
    
    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
        animationRef.current = null;
      }
      clearInterval(canvasClearInterval);
    };
  }, [isDetecting]);

  const connectWebSocket = () => {
    const ws = new WebSocket('ws://localhost:8001/ws/detection');
    
    ws.onopen = () => {
      console.log('WebSocket connected');
      setIsConnected(true);
    };
    
    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        if (data.type === 'detection_result') {
          setDetectionResult(data);
          
          // Always clear canvas first
          clearCanvas();
          
          // Only draw if hands are detected and landmarks exist
          if (data.hands_detected && data.landmarks && data.landmarks.length > 0) {
            drawLandmarks(data.landmarks, data.phrase);
          }
          // If no hands detected, canvas stays cleared
        }
      } catch (error) {
        console.error('Error parsing WebSocket message:', error);
        // Clear canvas on error too
        clearCanvas();
      }
    };
    
    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
      setIsConnected(false);
    };
    
    ws.onclose = () => {
      console.log('WebSocket disconnected');
      setIsConnected(false);
      // Try to reconnect after a delay
      setTimeout(() => {
        if (isDetecting) {
          connectWebSocket();
        }
      }, 2000);
    };
    
    wsRef.current = ws;
  };

  const drawLandmarks = (landmarks, phrase = null) => {
    if (!canvasRef.current) return;
    
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    const video = webcamRef.current.video;
    
    if (!video) {
      // If no video, still clear the canvas
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      return;
    }
    
    // Set canvas size to match video
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    
    // Clear canvas completely - use the full canvas dimensions
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // Only draw landmarks if they exist and are valid
    if (landmarks && landmarks.length > 0) {
      // Draw landmarks
      landmarks.forEach((landmark, index) => {
        if (landmark && typeof landmark.x === 'number' && typeof landmark.y === 'number') {
          const x = landmark.x * canvas.width;
          const y = landmark.y * canvas.height;
          
          // Draw landmark point
          ctx.beginPath();
          ctx.arc(x, y, 4, 0, 2 * Math.PI);
          ctx.fillStyle = '#FF0000'; // Red color
          ctx.fill();
          
          // Draw landmark number (only for key points to reduce clutter)
          if (index === 4 || index === 8 || index === 12 || index === 16 || index === 20) {
            ctx.fillStyle = '#FFFFFF';
            ctx.font = '10px Arial';
            ctx.fillText(index.toString(), x + 6, y - 6);
          }
        }
      });
      
      // Draw connections between landmarks
      const connections = [
        [0, 1], [1, 2], [2, 3], [3, 4], // Thumb
        [0, 5], [5, 6], [6, 7], [7, 8], // Index finger
        [0, 9], [9, 10], [10, 11], [11, 12], // Middle finger
        [0, 13], [13, 14], [14, 15], [15, 16], // Ring finger
        [0, 17], [17, 18], [18, 19], [19, 20], // Pinky
        [0, 5], [5, 9], [9, 13], [13, 17] // Palm connections
      ];
      
      ctx.strokeStyle = '#00FF00'; // Green color
      ctx.lineWidth = 2;
      
      connections.forEach(([start, end]) => {
        if (landmarks[start] && landmarks[end] && 
            typeof landmarks[start].x === 'number' && typeof landmarks[start].y === 'number' &&
            typeof landmarks[end].x === 'number' && typeof landmarks[end].y === 'number') {
          const startX = landmarks[start].x * canvas.width;
          const startY = landmarks[start].y * canvas.height;
          const endX = landmarks[end].x * canvas.width;
          const endY = landmarks[end].y * canvas.height;
          
          ctx.beginPath();
          ctx.moveTo(startX, startY);
          ctx.lineTo(endX, endY);
          ctx.stroke();
        }
      });
    }
    
    // Draw phrase on camera if detected
    if (phrase) {
      // Create a semi-transparent background for the phrase
      ctx.fillStyle = 'rgba(0, 0, 0, 0.8)';
      ctx.fillRect(10, 10, 320, 70);
      
      // Draw phrase text
      ctx.fillStyle = '#FF6B6B';
      ctx.font = 'bold 28px Arial';
      ctx.textAlign = 'left';
      ctx.fillText(phrase, 20, 50);
      
      // Draw a border around the phrase box
      ctx.strokeStyle = '#FF6B6B';
      ctx.lineWidth = 3;
      ctx.strokeRect(10, 10, 320, 70);
    }
  };

  const getAccuracyColor = (accuracy) => {
    if (accuracy >= 0.9) return '#00FF00'; // Green
    if (accuracy >= 0.7) return '#FFFF00'; // Yellow
    if (accuracy >= 0.5) return '#FFA500'; // Orange
    return '#FF0000'; // Red
  };

  const clearCanvas = () => {
    if (!canvasRef.current) return;
    
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    
    // Clear the entire canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);
  };

  const getConfidenceColor = (confidence) => {
    if (confidence >= 0.9) return '#00FF00'; // Green
    if (confidence >= 0.7) return '#FFFF00'; // Yellow
    if (confidence >= 0.5) return '#FFA500'; // Orange
    return '#FF0000'; // Red
  };

  const toggleDetection = () => {
    if (isDetecting) {
      setIsDetecting(false);
      // Stop animation and clear canvas when stopping detection
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
        animationRef.current = null;
      }
      clearCanvas();
    } else {
      setIsDetecting(true);
      startDetection();
    }
  };

  const startDetection = () => {
    if (!isDetecting) return;
    
    let lastFrameTime = 0;
    const frameInterval = 1000 / 15; // 15 FPS for better performance
    
    const captureFrame = (currentTime) => {
      if (!isDetecting) {
        // Stop the animation loop when detection is disabled
        if (animationRef.current) {
          cancelAnimationFrame(animationRef.current);
          animationRef.current = null;
        }
        return;
      }
      
      // Limit frame rate to prevent overwhelming the backend
      if (currentTime - lastFrameTime >= frameInterval) {
        if (webcamRef.current && wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
          const imageSrc = webcamRef.current.getScreenshot();
          if (imageSrc) {
            wsRef.current.send(JSON.stringify({
              type: 'frame',
              image: imageSrc
            }));
          }
        }
        lastFrameTime = currentTime;
      }
      
      // Continue the animation loop only if still detecting
      if (isDetecting) {
        animationRef.current = requestAnimationFrame(captureFrame);
      }
    };
    
    animationRef.current = requestAnimationFrame(captureFrame);
  };

  useEffect(() => {
    if (isDetecting) {
      startDetection();
    } else {
      // Clear canvas when detection stops
      clearCanvas();
    }
    
    // Cleanup function to ensure detection stops when component unmounts
    return () => {
      if (isDetecting) {
        setIsDetecting(false);
        clearCanvas();
      }
    };
  }, [isDetecting]);

  return (
    <div className="App">
      <header className="App-header">
        <h1>Sign Language Detection</h1>
        <div className="status">
          Status: {isConnected ? '🟢 Connected' : '🔴 Disconnected'}
        </div>
      </header>
      
      <main className="App-main">
        <div className="camera-container">
          <div className="video-wrapper">
            <Webcam
              ref={webcamRef}
              audio={false}
              screenshotFormat="image/jpeg"
              videoConstraints={{
                width: 640,
                height: 480,
                facingMode: 'user'
              }}
            />
            <canvas
              ref={canvasRef}
              className="landmarks-canvas"
              style={{
                position: 'absolute',
                top: 0,
                left: 0,
                pointerEvents: 'none'
              }}
            />
          </div>
          
          <div className="controls">
            <button 
              onClick={toggleDetection}
              className={`detection-btn ${isDetecting ? 'active' : ''}`}
            >
              {isDetecting ? 'Stop Detection' : 'Start Detection'}
            </button>
            <button 
              onClick={clearCanvas}
              className="clear-btn"
              style={{ marginLeft: '10px' }}
            >
              Clear Canvas
            </button>
          </div>
        </div>
        
        <div className="results">
          <h2>Detection Status</h2>
          {detectionResult ? (
            <div className="result-content">
              <div className="result-item">
                <strong>Status:</strong> {detectionResult.hands_detected ? '🟢 Detecting' : '🔴 No Hands'}
              </div>
              
              {detectionResult.confidence > 0 && (
                <div className="result-item">
                  <strong>Confidence:</strong> 
                  <span style={{ color: getConfidenceColor(detectionResult.confidence) }}>
                    {(detectionResult.confidence * 100).toFixed(1)}%
                  </span>
                </div>
              )}
              
              {detectionResult.accuracy > 0 && (
                <div className="result-item">
                  <strong>Accuracy:</strong> 
                  <span style={{ color: getAccuracyColor(detectionResult.accuracy) }}>
                    {(detectionResult.accuracy * 100).toFixed(1)}%
                  </span>
                </div>
              )}
            </div>
          ) : (
            <p>Start detection to see real-time results on camera.</p>
          )}
        </div>
        
        <div className="instructions">
          <h3>Try These Signs:</h3>
          <div className="sign-examples">
            <div className="sign-example">
              <strong>A:</strong> Make a fist
            </div>
            <div className="sign-example">
              <strong>B:</strong> Point with index finger
            </div>
            <div className="sign-example">
              <strong>V:</strong> Peace sign (index + middle)
            </div>
            <div className="sign-example">
              <strong>W:</strong> Three fingers up
            </div>
            <div className="sign-example">
              <strong>5:</strong> All fingers extended
            </div>
            <div className="sign-example">
              <strong>I Love You:</strong> Thumb + index + pinky
            </div>
            <div className="sign-example">
              <strong>👍:</strong> Thumbs up
            </div>
            <div className="sign-example">
              <strong>👌:</strong> OK sign (thumb + index circle)
            </div>
            <div className="sign-example">
              <strong>🤘:</strong> Rock on (index + pinky)
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}

export default App; 