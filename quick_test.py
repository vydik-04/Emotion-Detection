#!/usr/bin/env python3
"""
Quick test to verify sadness detection for "im not happy"
"""

from ultimate_emotion_detector import UltimateEmotionDetector

def quick_test():
    print("🧪 Testing: 'im not happy'")
    print("=" * 40)
    
    # Initialize detector
    detector = UltimateEmotionDetector()
    
    if detector.model is None:
        print("❌ Model failed to load!")
        return
    
    # Test the specific phrase
    test_text = "im not happy"
    
    print(f"📝 Input: '{test_text}'")
    print()
    
    # Get predictions
    result = detector.analyze_text(test_text)
    predicted_emotion = result['most_likely_emotion']
    
    print(f"🎭 Predicted Emotion: {predicted_emotion}")
    print(f"🎯 Expected: sadness")
    print()
    
    # Check if correct
    if predicted_emotion == 'sadness':
        print("✅ SUCCESS! Model correctly detected sadness")
    else:
        print("❌ INCORRECT! Model did not detect sadness")
    
    print("\n📊 Top 3 Emotions:")
    for i, emotion_data in enumerate(result['top_3_emotions'], 1):
        emotion = emotion_data['emotion']
        prob = emotion_data['probability']
        print(f"  {i}. {emotion}: {prob:.2%}")
    
    print("\n" + "=" * 40)
    
    # Test a few more negation cases
    more_tests = [
        "I'm not happy",
        "not feeling good", 
        "never feeling great",
        "not ok"
    ]
    
    print("🔄 Testing more negation cases:")
    print("-" * 40)
    
    for text in more_tests:
        emotion = detector.predict_single_emotion(text)
        status = "✅" if emotion == 'sadness' else "❌"
        print(f"{status} '{text}' → {emotion}")

if __name__ == "__main__":
    quick_test()
