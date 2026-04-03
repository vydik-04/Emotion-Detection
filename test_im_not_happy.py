#!/usr/bin/env python3
"""
Direct test of "im not happy" with the emotion detection model
"""

from ultimate_emotion_detector import UltimateEmotionDetector

def test_im_not_happy():
    print("🎭 TESTING: 'im not happy'")
    print("=" * 40)
    
    # Initialize the detector
    detector = UltimateEmotionDetector()
    
    if detector.model is None:
        print("❌ Model failed to load!")
        return
    
    # Test the exact phrase
    test_text = "im not happy"
    
    print(f"📝 Input: '{test_text}'")
    print()
    
    # Get the analysis
    result = detector.analyze_text(test_text)
    
    print(f"🎯 PREDICTED EMOTION: {result['most_likely_emotion'].upper()}")
    print()
    
    print("📊 TOP 3 EMOTIONS:")
    for i, emotion_data in enumerate(result['top_3_emotions'], 1):
        emotion = emotion_data['emotion']
        probability = emotion_data['probability']
        
        # Create visual bar
        bar_length = int(probability * 30)
        bar = "█" * bar_length + "░" * (30 - bar_length)
        
        # Add confidence indicator
        if probability > 0.7:
            confidence = "🔥 HIGH"
        elif probability > 0.4:
            confidence = "💪 GOOD"
        else:
            confidence = "👍 LOW"
        
        print(f"  {i}. {emotion.replace('_', ' ').title():<20} {bar} {probability:.1%} {confidence}")
    
    print()
    
    # Check if correct
    if result['most_likely_emotion'] == 'sadness':
        print("✅ SUCCESS! Model correctly detected SADNESS")
        print("🎉 Your negation handling is working perfectly!")
    else:
        print("❌ FAILED! Model did not detect sadness")
        print(f"   Expected: sadness")
        print(f"   Got: {result['most_likely_emotion']}")

if __name__ == "__main__":
    test_im_not_happy()
