import React, { useRef, useCallback, useEffect, useState } from 'react';
import Webcam from 'react-webcam';
import { Camera as CameraIcon, Video, VideoOff, RotateCcw } from 'lucide-react';

const Camera = ({ sessionId, onDetectionResult, onConnectionStatus }) => {
  const webcamRef = useRef(null);
  const [isStreaming, setIsStreaming] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [error, setError] = useState(null);
  const [facingMode, setFacingMode] = useState('user');
  const [ws, setWs] = useState(null);

  // WebSocket connection
  useEffect(() => {
    if (sessionId) {
      connectWebSocket();
    }
    
    return () => {
      if (ws) {
        ws.close();
      }
    };
  }, [sessionId]);

  const connectWebSocket = () => {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//localhost:8001/ws/detection`;
    
    console.log('Attempting to connect to WebSocket:', wsUrl);
    
    const websocket = new WebSocket(wsUrl);
    
    websocket.onopen = () => {
      console.log('WebSocket connected successfully');
      onConnectionStatus(true);
      setError(null);
    };
    
    websocket.onmessage = (event) => {
      console.log('Received WebSocket message:', event.data);
      const data = JSON.parse(event.data);
      if (data.type === 'detection_result') {
        console.log('Processing detection result:', data);
        onDetectionResult(data);
      }
    };
    
    websocket.onerror = (error) => {
      console.error('WebSocket error:', error);
      onConnectionStatus(false);
      setError('Connection error');
    };
    
    websocket.onclose = () => {
      console.log('WebSocket disconnected');
      onConnectionStatus(false);
    };
    
    setWs(websocket);
  };

  const captureFrame = useCallback(() => {
    if (webcamRef.current && ws && ws.readyState === WebSocket.OPEN) {
      const imageSrc = webcamRef.current.getScreenshot();
      if (imageSrc) {
        console.log('Capturing frame and sending to WebSocket');
        // Send frame data to WebSocket
        const frameData = {
          type: 'frame',
          image: imageSrc,
          session_id: sessionId
        };
        console.log('Sending frame data:', { type: frameData.type, session_id: frameData.session_id });
        ws.send(JSON.stringify(frameData));
      } else {
        console.log('No image captured from webcam');
      }
    } else {
      console.log('WebSocket not ready or webcam not available');
    }
  }, [ws, sessionId]);

  // Start/stop streaming
  const toggleStreaming = () => {
    if (isStreaming) {
      setIsStreaming(false);
    } else {
      setIsStreaming(true);
      startStreaming();
    }
  };

  const startStreaming = () => {
    if (!isStreaming) return;
    
    captureFrame();
    
    // Continue streaming
    setTimeout(() => {
      if (isStreaming) {
        startStreaming();
      }
    }, 100); // 10 FPS
  };

  const switchCamera = () => {
    setFacingMode(facingMode === 'user' ? 'environment' : 'user');
  };

  const videoConstraints = {
    width: 640,
    height: 480,
    facingMode: facingMode,
    aspectRatio: 4/3
  };

  const handleUserMediaError = (error) => {
    console.error('Camera error:', error);
    setError('Camera access denied. Please allow camera permissions.');
  };

  return (
    <div className="space-y-4">
      {/* Camera Controls */}
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-4">
          <button
            onClick={toggleStreaming}
            className={`flex items-center space-x-2 px-4 py-2 rounded-lg font-medium transition-colors ${
              isStreaming 
                ? 'bg-red-600 hover:bg-red-700 text-white' 
                : 'bg-green-600 hover:bg-green-700 text-white'
            }`}
          >
            {isStreaming ? (
              <>
                <VideoOff size={20} />
                <span>Stop Detection</span>
              </>
            ) : (
              <>
                <Video size={20} />
                <span>Start Detection</span>
              </>
            )}
          </button>
          
          <button
            onClick={switchCamera}
            className="flex items-center space-x-2 px-4 py-2 bg-gray-200 hover:bg-gray-300 text-gray-700 rounded-lg font-medium transition-colors"
          >
            <RotateCcw size={20} />
            <span>Switch Camera</span>
          </button>
        </div>
        
        <div className="flex items-center space-x-2">
          <div className={`w-3 h-3 rounded-full ${isStreaming ? 'bg-green-500' : 'bg-gray-400'}`}></div>
          <span className="text-sm text-gray-600">
            {isStreaming ? 'Detecting' : 'Idle'}
          </span>
        </div>
      </div>

      {/* Camera Feed */}
      <div className="relative bg-black rounded-lg overflow-hidden">
        <Webcam
          ref={webcamRef}
          audio={false}
          screenshotFormat="image/jpeg"
          videoConstraints={videoConstraints}
          onUserMediaError={handleUserMediaError}
          className="w-full h-auto"
        />
        
        {/* Overlay */}
        <div className="absolute inset-0 pointer-events-none">
          {/* Detection area indicator */}
          <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2">
            <div className="w-48 h-48 border-2 border-primary-500 rounded-lg opacity-50"></div>
          </div>
          
          {/* Processing indicator */}
          {isProcessing && (
            <div className="absolute top-4 right-4 bg-black/50 text-white px-3 py-1 rounded-lg text-sm">
              Processing...
            </div>
          )}
        </div>
      </div>

      {/* Error Display */}
      {error && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-4">
          <div className="flex items-center">
            <div className="flex-shrink-0">
              <CameraIcon className="h-5 w-5 text-red-400" />
            </div>
            <div className="ml-3">
              <p className="text-sm text-red-800">{error}</p>
            </div>
          </div>
        </div>
      )}

      {/* Instructions */}
      <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
        <h4 className="text-sm font-medium text-blue-900 mb-2">Camera Tips:</h4>
        <ul className="text-sm text-blue-800 space-y-1">
          <li>• Ensure your hand is clearly visible in the center area</li>
          <li>• Maintain good lighting for better detection</li>
          <li>• Keep your hand steady while making signs</li>
          <li>• Position your hand at arm's length from the camera</li>
        </ul>
      </div>
    </div>
  );
};

export default Camera; 