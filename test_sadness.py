#!/usr/bin/env python3
"""
Test script to verify sadness detection for negation phrases like "im not happy"
"""

from ultimate_emotion_detector import UltimateEmotionDetector
import sys

def test_sadness_detection():
    """Test the model with sadness-related phrases"""
    
    print("🧪 Testing Emotion Detection Model for Sadness")
    print("=" * 50)
    
    # Initialize the detector
    detector = UltimateEmotionDetector()
    
    if detector.model is None:
        print("❌ Model failed to load!")
        return False
    
    # Test phrases that should be detected as sadness
    test_phrases = [
        "im not happy",
        "I'm not happy",
        "i am not happy",
        "not feeling good",
        "never feeling great",
        "not ok",
        "I'm sad",
        "feeling down",
        "so depressed",
        "heartbroken",
        "crying",
        "breakup hurt me"
    ]
    
    print("\n🎯 Testing phrases for sadness detection:")
    print("-" * 50)
    
    correct_predictions = 0
    total_tests = len(test_phrases)
    
    for i, phrase in enumerate(test_phrases, 1):
        try:
            # Get prediction
            result = detector.analyze_text(phrase)
            predicted_emotion = result['predicted_emotion']
            confidence = result['confidence']
            
            # Check if sadness is detected
            is_correct = predicted_emotion == 'sadness'
            if is_correct:
                correct_predictions += 1
                status = "✅ CORRECT"
            else:
                status = "❌ INCORRECT"
            
            print(f"{i:2d}. '{phrase}'")
            print(f"    → Predicted: {predicted_emotion} ({confidence:.2%}) {status}")
            
            # Show top 3 emotions for context
            top_3_str = ', '.join([f"{e['emotion']}({e['probability']:.1%})" for e in result['top_3_emotions']])
            print(f"    → Top 3: {top_3_str}")
            print()
            
        except Exception as e:
            print(f"❌ Error testing '{phrase}': {e}")
            print()
    
    # Summary
    accuracy = (correct_predictions / total_tests) * 100
    print("=" * 50)
    print(f"📊 RESULTS SUMMARY:")
    print(f"   Correct predictions: {correct_predictions}/{total_tests}")
    print(f"   Accuracy: {accuracy:.1f}%")
    
    if accuracy >= 80:
        print(f"🎉 EXCELLENT! Model correctly detects sadness in most cases")
    elif accuracy >= 60:
        print(f"👍 GOOD! Model detects sadness reasonably well")
    else:
        print(f"⚠️  NEEDS IMPROVEMENT! Model needs better sadness detection")
    
    return accuracy >= 60

if __name__ == "__main__":
    success = test_sadness_detection()
    sys.exit(0 if success else 1)
