import axios from 'axios';

// Create axios instance with base configuration
const api = axios.create({
  baseURL: process.env.REACT_APP_API_URL || 'http://localhost:8001',
  timeout: 10000,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor
api.interceptors.request.use(
  (config) => {
    // Add any auth tokens here if needed
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Response interceptor
api.interceptors.response.use(
  (response) => {
    return response;
  },
  (error) => {
    console.error('API Error:', error);
    return Promise.reject(error);
  }
);

// API functions
export const apiService = {
  // Session management
  createSession: async () => {
    const response = await api.post('/api/sessions');
    return response.data;
  },

  getSessionStats: async (sessionId) => {
    const response = await api.get(`/api/sessions/${sessionId}/stats`);
    return response.data;
  },

  // Sign detection
  detectSign: async (frameData) => {
    const response = await api.post('/api/detect-sign', frameData);
    return response.data;
  },

  // Sign information
  getSigns: async () => {
    const response = await api.get('/api/signs');
    return response.data;
  },

  getSign: async (signId) => {
    const response = await api.get(`/api/signs/${signId}`);
    return response.data;
  },

  getSignByLetter: async (letter) => {
    const response = await api.get(`/api/signs/letter/${letter}`);
    return response.data;
  },

  getSignsByCategory: async (category) => {
    const response = await api.get(`/api/signs/category/${category}`);
    return response.data;
  },

  getSignStats: async () => {
    const response = await api.get('/api/signs/stats');
    return response.data;
  },

  // Health check
  healthCheck: async () => {
    const response = await api.get('/health');
    return response.data;
  },
};

// WebSocket connection helper
export const createWebSocket = (sessionId) => {
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  const wsUrl = `${protocol}//${window.location.host}/ws/detection`;
  
  const ws = new WebSocket(wsUrl);
  
  ws.onopen = () => {
    console.log('WebSocket connected');
  };
  
  ws.onerror = (error) => {
    console.error('WebSocket error:', error);
  };
  
  ws.onclose = () => {
    console.log('WebSocket disconnected');
  };
  
  return ws;
};

// Utility functions
export const formatTimestamp = (timestamp) => {
  return new Date(timestamp).toLocaleString();
};

export const formatConfidence = (confidence) => {
  return Math.round(confidence * 100);
};

export const formatAccuracy = (accuracy) => {
  return Math.round(accuracy * 100);
};

export default apiService; 