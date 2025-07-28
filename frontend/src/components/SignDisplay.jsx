import React from 'react';
import { Hand, AlertCircle, CheckCircle } from 'lucide-react';

const SignDisplay = ({ result }) => {
  if (!result) {
    return (
      <div className="text-center py-8">
        <Hand className="mx-auto h-12 w-12 text-gray-400 mb-4" />
        <p className="text-gray-500">No sign detected yet</p>
        <p className="text-sm text-gray-400 mt-2">Start detection to see results</p>
      </div>
    );
  }

  const { detected_sign, confidence, accuracy, message } = result;

  // Determine status and styling
  const getStatusInfo = () => {
    if (message && message.includes('No hands detected')) {
      return {
        icon: AlertCircle,
        color: 'text-yellow-500',
        bgColor: 'bg-yellow-50',
        borderColor: 'border-yellow-200',
        status: 'No Hands Detected'
      };
    }
    
    if (confidence >= 0.8) {
      return {
        icon: CheckCircle,
        color: 'text-green-500',
        bgColor: 'bg-green-50',
        borderColor: 'border-green-200',
        status: 'High Confidence'
      };
    } else if (confidence >= 0.6) {
      return {
        icon: Hand,
        color: 'text-blue-500',
        bgColor: 'bg-blue-50',
        borderColor: 'border-blue-200',
        status: 'Medium Confidence'
      };
    } else {
      return {
        icon: AlertCircle,
        color: 'text-orange-500',
        bgColor: 'bg-orange-50',
        borderColor: 'border-orange-200',
        status: 'Low Confidence'
      };
    }
  };

  const statusInfo = getStatusInfo();
  const StatusIcon = statusInfo.icon;

  return (
    <div className="space-y-4">
      {/* Main Sign Display */}
      <div className={`${statusInfo.bgColor} ${statusInfo.borderColor} border rounded-lg p-6 text-center`}>
        <StatusIcon className={`mx-auto h-16 w-16 ${statusInfo.color} mb-4`} />
        
        {detected_sign ? (
          <div className="space-y-2">
            <div className="text-6xl font-bold text-gray-900 mb-2">
              {detected_sign}
            </div>
            <p className="text-lg text-gray-600">
              American Sign Language
            </p>
            <p className="text-sm text-gray-500">
              Letter {detected_sign}
            </p>
          </div>
        ) : (
          <div className="space-y-2">
            <p className="text-lg text-gray-600">
              {message || 'Processing...'}
            </p>
          </div>
        )}
      </div>

      {/* Confidence Score */}
      {detected_sign && (
        <div className="bg-gray-50 rounded-lg p-4">
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm font-medium text-gray-700">Confidence</span>
            <span className="text-sm font-bold text-gray-900">
              {Math.round(confidence * 100)}%
            </span>
          </div>
          
          <div className="w-full bg-gray-200 rounded-full h-2">
            <div 
              className={`h-2 rounded-full transition-all duration-300 ${
                confidence >= 0.8 ? 'bg-green-500' : 
                confidence >= 0.6 ? 'bg-blue-500' : 'bg-orange-500'
              }`}
              style={{ width: `${confidence * 100}%` }}
            ></div>
          </div>
          
          <div className="flex items-center justify-between mt-2">
            <span className="text-xs text-gray-500">Low</span>
            <span className="text-xs text-gray-500">High</span>
          </div>
        </div>
      )}

      {/* Status Information */}
      <div className="flex items-center justify-center">
        <span className={`inline-flex items-center px-3 py-1 rounded-full text-sm font-medium ${statusInfo.bgColor} ${statusInfo.color}`}>
          <StatusIcon className="h-4 w-4 mr-1" />
          {statusInfo.status}
        </span>
      </div>

      {/* Additional Info */}
      {result.timestamp && (
        <div className="text-center text-xs text-gray-500">
          Detected at {new Date(result.timestamp).toLocaleTimeString()}
        </div>
      )}
    </div>
  );
};

export default SignDisplay; 