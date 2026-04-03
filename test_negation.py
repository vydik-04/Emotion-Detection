#!/usr/bin/env python3
"""
Quick test script to debug negation handling
"""

from ultimate_emotion_detector import UltimateEmotionDetector

def test_negation():
    print("🔍 Testing Negation Handling...")
    
    # Initialize detector
    detector = UltimateEmotionDetector()
    
    if detector.model is None:
        print("❌ Model failed to load!")
        return
    
    # Test phrases
    test_phrases = [
        "not happy",
        "I'm not happy", 
        "not feeling good",
        "never happy",
        "I am happy",  # Control - should be joy
        "feeling sad"  # Control - should be sadness
    ]
    
    print("\n📋 Test Results:")
    print("=" * 50)
    
    for phrase in test_phrases:
        try:
            result = detector.analyze_text(phrase)
            prediction = result['most_likely_emotion']
            top_3 = result['top_3_emotions']
            
            print(f"\n📝 Text: '{phrase}'")
            print(f"🎯 Prediction: {prediction}")
            print("📊 Top 3:")
            for i, item in enumerate(top_3, 1):
                emotion = item['emotion']
                prob = item['probability']
                print(f"   {i}. {emotion}: {prob:.2%}")
                
        except Exception as e:
            print(f"❌ Error with '{phrase}': {e}")

if __name__ == "__main__":
    test_negation()
