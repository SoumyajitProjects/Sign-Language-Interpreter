import React from 'react';
import { TrendingUp, Target, Award, AlertTriangle } from 'lucide-react';

const AccuracyMeter = ({ result }) => {
  if (!result || !result.accuracy) {
    return (
      <div className="text-center py-8">
        <Target className="mx-auto h-12 w-12 text-gray-400 mb-4" />
        <p className="text-gray-500">No accuracy data yet</p>
        <p className="text-sm text-gray-400 mt-2">Start detection to see accuracy scores</p>
      </div>
    );
  }

  const { accuracy, confidence } = result;
  const accuracyPercent = Math.round(accuracy * 100);

  // Determine accuracy level and styling
  const getAccuracyInfo = () => {
    if (accuracy >= 0.9) {
      return {
        level: 'Excellent',
        color: 'text-green-600',
        bgColor: 'bg-green-100',
        borderColor: 'border-green-200',
        icon: Award,
        iconColor: 'text-green-500',
        message: 'Perfect form! Keep it up!',
        suggestion: 'You\'re doing great with this sign.'
      };
    } else if (accuracy >= 0.7) {
      return {
        level: 'Good',
        color: 'text-blue-600',
        bgColor: 'bg-blue-100',
        borderColor: 'border-blue-200',
        icon: TrendingUp,
        iconColor: 'text-blue-500',
        message: 'Good accuracy!',
        suggestion: 'Try to make the gesture a bit more precise.'
      };
    } else if (accuracy >= 0.5) {
      return {
        level: 'Fair',
        color: 'text-orange-600',
        bgColor: 'bg-orange-100',
        borderColor: 'border-orange-200',
        icon: Target,
        iconColor: 'text-orange-500',
        message: 'Fair accuracy. Room for improvement.',
        suggestion: 'Check your hand position and finger placement.'
      };
    } else {
      return {
        level: 'Poor',
        color: 'text-red-600',
        bgColor: 'bg-red-100',
        borderColor: 'border-red-200',
        icon: AlertTriangle,
        iconColor: 'text-red-500',
        message: 'Low accuracy. Let\'s work on this.',
        suggestion: 'Review the correct hand position and try again.'
      };
    }
  };

  const accuracyInfo = getAccuracyInfo();
  const AccuracyIcon = accuracyInfo.icon;

  return (
    <div className="space-y-4">
      {/* Accuracy Score Display */}
      <div className={`${accuracyInfo.bgColor} ${accuracyInfo.borderColor} border rounded-lg p-6 text-center`}>
        <AccuracyIcon className={`mx-auto h-12 w-12 ${accuracyInfo.iconColor} mb-4`} />
        
        <div className="space-y-2">
          <div className="text-4xl font-bold text-gray-900">
            {accuracyPercent}%
          </div>
          <p className={`text-lg font-medium ${accuracyInfo.color}`}>
            {accuracyInfo.level}
          </p>
          <p className="text-sm text-gray-600">
            {accuracyInfo.message}
          </p>
        </div>
      </div>

      {/* Accuracy Meter */}
      <div className="space-y-3">
        <div className="flex items-center justify-between">
          <span className="text-sm font-medium text-gray-700">Accuracy Score</span>
          <span className="text-sm font-bold text-gray-900">
            {accuracyPercent}%
          </span>
        </div>
        
        <div className="relative">
          <div className="w-full bg-gray-200 rounded-full h-3">
            <div 
              className={`h-3 rounded-full transition-all duration-500 ${
                accuracy >= 0.9 ? 'bg-green-500' : 
                accuracy >= 0.7 ? 'bg-blue-500' : 
                accuracy >= 0.5 ? 'bg-orange-500' : 'bg-red-500'
              }`}
              style={{ width: `${accuracyPercent}%` }}
            ></div>
          </div>
          
          {/* Accuracy level markers */}
          <div className="flex justify-between mt-1">
            <span className="text-xs text-gray-500">0%</span>
            <span className="text-xs text-gray-500">50%</span>
            <span className="text-xs text-gray-500">70%</span>
            <span className="text-xs text-gray-500">90%</span>
            <span className="text-xs text-gray-500">100%</span>
          </div>
        </div>
      </div>

      {/* Feedback Section */}
      <div className="bg-gray-50 rounded-lg p-4">
        <h4 className="text-sm font-medium text-gray-900 mb-2">Suggestion:</h4>
        <p className="text-sm text-gray-600">
          {accuracyInfo.suggestion}
        </p>
      </div>

      {/* Detailed Metrics */}
      <div className="grid grid-cols-2 gap-4">
        <div className="bg-white border border-gray-200 rounded-lg p-3">
          <div className="text-xs text-gray-500 mb-1">Confidence</div>
          <div className="text-lg font-bold text-gray-900">
            {Math.round(confidence * 100)}%
          </div>
        </div>
        
        <div className="bg-white border border-gray-200 rounded-lg p-3">
          <div className="text-xs text-gray-500 mb-1">Accuracy</div>
          <div className="text-lg font-bold text-gray-900">
            {accuracyPercent}%
          </div>
        </div>
      </div>

      {/* Progress Indicators */}
      <div className="space-y-2">
        <div className="flex items-center justify-between text-sm">
          <span className="text-gray-600">Detection Quality</span>
          <span className={`font-medium ${
            accuracy >= 0.9 ? 'text-green-600' : 
            accuracy >= 0.7 ? 'text-blue-600' : 
            accuracy >= 0.5 ? 'text-orange-600' : 'text-red-600'
          }`}>
            {accuracyInfo.level}
          </span>
        </div>
        
        <div className="flex items-center justify-between text-sm">
          <span className="text-gray-600">Model Confidence</span>
          <span className={`font-medium ${
            confidence >= 0.8 ? 'text-green-600' : 
            confidence >= 0.6 ? 'text-blue-600' : 
            confidence >= 0.4 ? 'text-orange-600' : 'text-red-600'
          }`}>
            {Math.round(confidence * 100)}%
          </span>
        </div>
      </div>
    </div>
  );
};

export default AccuracyMeter; 