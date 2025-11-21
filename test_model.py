"""
Direct test of the tone converter model to verify it's working correctly
"""

import sys
sys.path.append('.')

from models.tone_converter import ToneConverter
import json

def test_conversion(text, mode):
    print(f"\n{'='*60}")
    print(f"Testing: {mode.upper()} mode")
    print(f"{'='*60}")
    print(f"Original: {text}")
    
    converter = ToneConverter()
    result = converter.convert(text, mode)
    
    print(f"\nConverted: {result['converted_text']}")
    print(f"Confidence: {result['confidence']}")
    print(f"Model: {result['model']}")
    
    print(f"\nAlternatives ({len(result['alternatives'])}):")
    for alt in result['alternatives']:
        print(f"  - {alt['mode']}: {alt['text'][:80]}...")
    
    print(f"\nProcessing Notes:")
    for note in result['analysis'].get('processing_notes', []):
        print(f"  • {note}")
    
    return result

def main():
    print("🧪 Testing Tone Converter Model")
    print("="*60)
    
    # Test cases
    test_cases = [
        ("I need this done right now!", "polite"),
        ("Hey, I can't make it to the meeting", "formal"),
        ("We need to fix this problem quickly", "professional"),
        ("Thanks for helping me out", "friendly"),
    ]
    
    for text, mode in test_cases:
        result = test_conversion(text, mode)
        
        # Verify alternatives exist
        if len(result['alternatives']) == 0:
            print("\n⚠️  WARNING: NO ALTERNATIVES GENERATED!")
        else:
            print(f"\n✅ {len(result['alternatives'])} alternatives generated successfully")
    
    print(f"\n{'='*60}")
    print("✅ All tests completed!")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()
