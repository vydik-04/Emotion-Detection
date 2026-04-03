#!/usr/bin/env python3
"""
Debug script to test pattern matching for sadness rules
"""

import re
from ultimate_emotion_detector import UltimateEmotionDetector

def test_patterns():
    print("🔍 DEBUGGING SADNESS PATTERN MATCHING")
    print("=" * 50)
    
    # Initialize detector to get the patterns
    detector = UltimateEmotionDetector()
    sadness_patterns = detector.keyword_rules['sadness']
    
    # Test phrases
    test_phrases = [
        "im not happy",
        "I'm not happy", 
        "i am not happy",
        "not happy",
        "not feeling good",
        "never feeling great"
    ]
    
    print(f"\n📋 Sadness Patterns ({len(sadness_patterns)}):")
    for i, pattern in enumerate(sadness_patterns[:10], 1):  # Show first 10
        print(f"  {i:2d}. {pattern}")
    if len(sadness_patterns) > 10:
        print(f"     ... and {len(sadness_patterns) - 10} more")
    
    print(f"\n🧪 Testing Pattern Matches:")
    print("-" * 50)
    
    for phrase in test_phrases:
        print(f"\n📝 Testing: '{phrase}'")
        phrase_lower = phrase.lower()
        
        matches = []
        for i, pattern in enumerate(sadness_patterns):
            try:
                if re.search(pattern, phrase_lower, re.IGNORECASE):
                    matches.append((i+1, pattern))
            except re.error as e:
                print(f"    ❌ Pattern {i+1} error: {e}")
        
        if matches:
            print(f"    ✅ Found {len(matches)} matches:")
            for pattern_num, pattern in matches:
                print(f"       Pattern {pattern_num}: {pattern}")
        else:
            print("    ❌ No pattern matches found")
    
    print("\n" + "=" * 50)
    print("🔧 MANUAL PATTERN TESTING")
    print("=" * 50)
    
    # Test specific patterns manually
    manual_tests = [
        (r'\b(i\'?m?|am|was|were|are)?\s*(not|never|no)\s+(happy|joyful|glad|pleased|satisfied|amused|good|great|fine|well)\b', "im not happy"),
        (r'\bnot\s+(happy|joyful|glad|pleased|satisfied|amused)\b', "im not happy"),
        (r'\bnot\s+(happy|joyful|glad|pleased|satisfied|amused)\b', "not happy"),
        (r'\b(i\'?m?|am)\s+not\s+(happy|okay|ok|fine|good|well)\b', "im not happy"),
        (r'\b(i\'?m?|am)\s+not\s+(happy|okay|ok|fine|good|well)\b', "I'm not happy"),
    ]
    
    for pattern, test_text in manual_tests:
        match = re.search(pattern, test_text.lower(), re.IGNORECASE)
        status = "✅ MATCH" if match else "❌ NO MATCH"
        print(f"{status} '{test_text}' vs pattern: {pattern}")
        if match:
            print(f"         Matched: '{match.group()}'")

if __name__ == "__main__":
    test_patterns()
