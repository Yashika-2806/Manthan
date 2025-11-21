"""
Quick demo script to test the model.
"""

import torch
from src.model import FactualStyleTransferModel
from src.entity_preserve import EntityPreserver


def main():
    print("="*60)
    print("Factuality-Preserving Style Transfer - Demo")
    print("="*60)
    
    # Initialize model (this will use pretrained BART)
    print("\nInitializing model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = FactualStyleTransferModel(
        model_name="facebook/bart-base",
        style_dim=128,
        num_styles=4
    ).to(device)
    
    model.eval()
    
    # Entity preserver for demonstration
    entity_preserver = EntityPreserver()
    
    # Test examples
    test_examples = [
        {
            "text": "hey john, lets meet at 3pm tomorrow",
            "style": "formal",
            "expected_entities": ["john", "3pm", "tomorrow"]
        },
        {
            "text": "obama was born in hawaii in 1961",
            "style": "formal",
            "expected_entities": ["obama", "hawaii", "1961"]
        },
        {
            "text": "gonna grab lunch at mcdonalds with sarah",
            "style": "formal",
            "expected_entities": ["mcdonalds", "sarah"]
        }
    ]
    
    print("\n" + "="*60)
    print("Running style transfer examples...")
    print("="*60)
    
    for i, example in enumerate(test_examples, 1):
        print(f"\n[Example {i}]")
        print(f"Input (informal): {example['text']}")
        
        # Extract entities
        entities = entity_preserver.extract_entities(example['text'])
        print(f"Detected entities: {[e['text'] for e in entities]}")
        
        # Generate
        with torch.no_grad():
            output = model.generate(
                input_text=example['text'],
                target_style=example['style'],
                preserve_entities=True,
                fact_check=True
            )
        
        print(f"Output (formal): {output}")
        
        # Check entity preservation
        preservation = entity_preserver.check_entity_preservation(
            example['text'],
            output
        )
        
        print(f"Entity preservation rate: {preservation['preservation_rate']:.1%}")
        
        if preservation['missing']:
            print(f"⚠️  Missing entities: {preservation['missing']}")
        else:
            print("✓ All entities preserved!")
        
        if preservation['hallucinated']:
            print(f"⚠️  Hallucinated entities: {preservation['hallucinated']}")
    
    print("\n" + "="*60)
    print("Demo complete!")
    print("="*60)
    
    print("\nNext steps:")
    print("1. Prepare your dataset with scripts/preprocess_data.py")
    print("2. Train the model with train.py --config configs/formality_transfer.yaml")
    print("3. Run inference with inference.py --checkpoint <path> --input <text>")


if __name__ == "__main__":
    main()
