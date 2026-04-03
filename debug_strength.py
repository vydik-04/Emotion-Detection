#!/usr/bin/env python3
"""
Debug rule strength calculation
"""

from ultimate_emotion_detector import UltimateEmotionDetector

def debug_strength():
    print("🔍 DEBUGGING RULE STRENGTH CALCULATION")
    print("=" * 50)
    
    detector = UltimateEmotionDetector()
    
    test_patterns = [
        r'\b(i\'?m?|am|was|were|are)?\s*(not|never|no)\s+(happy|joyful|glad|pleased|satisfied|amused|good|great|fine|well)\b',
        r'\bnot\s+(happy|joyful|glad|pleased|satisfied|amused)\b',
        r'\b(i\'?m?|am)\s+not\s+(happy|okay|ok|fine|good|well)\b'
    ]
    
    text = "im not happy"
    current_prob = 0.0282
    
    for i, pattern in enumerate(test_patterns, 1):
        print(f"\n🧪 Testing Pattern {i}: {pattern}")
        
        # Check individual components
        has_not = 'not ' in pattern
        has_never = 'never ' in pattern  
        has_no = 'no ' in pattern
        
        print(f"  📝 Pattern contains 'not ': {has_not}")
        print(f"  📝 Pattern contains 'never ': {has_never}")
        print(f"  📝 Pattern contains 'no ': {has_no}")
        
        # Test the condition
        is_sadness_negation = has_not or has_never or has_no
        print(f"  🎯 Is sadness negation: {is_sadness_negation}")
        
        # Calculate strength
        strength = detector._calculate_rule_strength('sadness', pattern, text.lower(), current_prob)
        print(f"  📊 Rule strength: {strength:.3f}")
        
        # What should happen
        if is_sadness_negation:
            print(f"  ✅ Should return 1.0 (max strength)")
        else:
            print(f"  ❌ Should follow normal calculation")

if __name__ == "__main__":
    debug_strength()
