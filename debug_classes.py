#!/usr/bin/env python3
"""
Debug script to check model classes and sadness index
"""

from ultimate_emotion_detector import UltimateEmotionDetector
import numpy as np

def debug_model_classes():
    print("🔍 DEBUGGING MODEL CLASSES AND SADNESS INDEX")
    print("=" * 60)
    
    # Initialize detector
    detector = UltimateEmotionDetector()
    
    if detector.model is None:
        print("❌ Model failed to load!")
        return
    
    print(f"📋 Model Classes ({len(detector.model.classes_)}):")
    for i, emotion in enumerate(detector.model.classes_):
        print(f"  {i:2d}. {emotion}")
    
    print(f"\n🎭 Expected Emotions ({len(detector.emotions)}):")
    for i, emotion in enumerate(detector.emotions):
        print(f"  {i:2d}. {emotion}")
    
    # Check if sadness is in model classes
    if 'sadness' in detector.model.classes_:
        sadness_idx = list(detector.model.classes_).index('sadness')
        print(f"\n✅ 'sadness' found at index {sadness_idx}")
    else:
        print(f"\n❌ 'sadness' NOT found in model classes!")
        return
    
    # Test a simple prediction
    test_text = "im not happy"
    print(f"\n🧪 Testing: '{test_text}'")
    
    # Get raw ML prediction
    processed_text = detector._negation_aware_preprocess(test_text)
    print(f"📝 Processed text: '{processed_text}'")
    
    ml_prediction = detector.model.predict([processed_text])[0]
    probabilities = detector.model.predict_proba([processed_text])[0]
    
    print(f"🤖 ML Prediction: {ml_prediction}")
    print(f"📊 ML Probabilities (top 5):")
    
    # Show top 5 ML predictions
    top_5_indices = np.argsort(probabilities)[-5:][::-1]
    for i, idx in enumerate(top_5_indices):
        emotion = detector.model.classes_[idx]
        prob = probabilities[idx]
        print(f"  {i+1}. {emotion}: {prob:.4f} ({prob*100:.2f}%)")
    
    # Check sadness probability specifically
    sadness_prob = probabilities[sadness_idx]
    print(f"\n🎯 Sadness probability: {sadness_prob:.4f} ({sadness_prob*100:.2f}%)")
    
    # Test rule matching manually
    print(f"\n🔧 Testing Rule Application:")
    text_lower = test_text.lower()
    sadness_patterns = detector.keyword_rules['sadness']
    
    matches_found = []
    for i, pattern in enumerate(sadness_patterns):
        import re
        if re.search(pattern, text_lower, re.IGNORECASE):
            rule_strength = detector._calculate_rule_strength('sadness', pattern, text_lower, sadness_prob)
            matches_found.append((i, pattern, rule_strength))
            print(f"  ✅ Pattern {i+1}: {pattern} (strength: {rule_strength:.3f})")
    
    if not matches_found:
        print("  ❌ No patterns matched!")
    else:
        print(f"\n📈 Best match strength: {max(match[2] for match in matches_found):.3f}")
        
        # Test the full rule application
        final_prediction, final_probabilities = detector.apply_keyword_rules(test_text, ml_prediction, probabilities)
        print(f"\n🎯 Final prediction: {final_prediction}")
        print(f"📊 Final sadness probability: {final_probabilities[sadness_idx]:.4f} ({final_probabilities[sadness_idx]*100:.2f}%)")

if __name__ == "__main__":
    debug_model_classes()
