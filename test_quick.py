import sys
sys.path.append('.')
from models.tone_converter import ToneConverter

converter = ToneConverter()
text = "I NEED THIS REPORT TODAY"

print("\n" + "="*60)
print("Testing: " + text)
print("="*60)

# Test Formal
formal = converter._convert_to_formal(text)
print(f"\nFORMAL:\n  Original: {text}\n  Formal:   {formal}")

# Test Informal
informal = converter._convert_to_informal(text)
print(f"\nINFORMAL:\n  Original: {text}\n  Informal: {informal}")

# Test Full conversion with alternatives
print("\n" + "="*60)
result = converter.convert(text, 'polite')
print(f"POLITE MODE:")
print(f"  Converted: {result['converted_text']}")
print(f"  Alternatives: {len(result['alternatives'])}")
for alt in result['alternatives']:
    print(f"    - {alt['mode']}: {alt['text'][:80]}")

print("\n" + "="*60)
