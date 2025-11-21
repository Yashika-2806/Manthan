"""
Inference script for style transfer.
"""

import torch
import argparse
from pathlib import Path

from src.model import FactualStyleTransferModel
from src.entity_preserve import EntityPreserver


def parse_args():
    parser = argparse.ArgumentParser(description="Style transfer inference")
    parser.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint")
    parser.add_argument("--input", type=str, required=True, help="Input text or file")
    parser.add_argument("--target_style", type=str, default="formal", 
                       choices=["formal", "informal", "simple", "complex"],
                       help="Target style")
    parser.add_argument("--output", type=str, default=None, help="Output file (optional)")
    parser.add_argument("--preserve_entities", action="store_true", help="Enforce entity preservation")
    parser.add_argument("--fact_check", action="store_true", help="Apply fact-checking")
    parser.add_argument("--num_beams", type=int, default=4, help="Number of beams for generation")
    parser.add_argument("--max_length", type=int, default=128, help="Maximum generation length")
    return parser.parse_args()


def load_model(checkpoint_path: str, device: torch.device) -> FactualStyleTransferModel:
    """Load model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract config if available
    config = checkpoint.get("config", {})
    model_config = config.get("model", {})
    
    # Create model
    model = FactualStyleTransferModel(
        model_name=model_config.get("backbone", "facebook/bart-base"),
        style_dim=model_config.get("style_dim", 128),
        num_styles=model_config.get("num_styles", 4)
    ).to(device)
    
    # Load weights
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    return model


def main():
    args = parse_args()
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from {args.checkpoint}...")
    model = load_model(args.checkpoint, device)
    
    # Check if input is file or text
    input_path = Path(args.input)
    if input_path.exists() and input_path.is_file():
        # Read from file
        with open(input_path, 'r', encoding='utf-8') as f:
            input_texts = [line.strip() for line in f if line.strip()]
    else:
        # Treat as single text
        input_texts = [args.input]
    
    # Generate outputs
    outputs = []
    entity_preserver = EntityPreserver()
    
    print(f"\nGenerating {len(input_texts)} outputs...")
    for i, input_text in enumerate(input_texts):
        print(f"\n[{i+1}/{len(input_texts)}]")
        print(f"Input: {input_text}")
        
        # Extract entities
        if args.preserve_entities:
            entities = entity_preserver.extract_entities(input_text)
            print(f"Entities: {[e['text'] for e in entities]}")
        
        # Generate
        output_text = model.generate(
            input_text=input_text,
            target_style=args.target_style,
            max_length=args.max_length,
            num_beams=args.num_beams,
            preserve_entities=args.preserve_entities,
            fact_check=args.fact_check
        )
        
        print(f"Output: {output_text}")
        
        # Check entity preservation
        if args.preserve_entities:
            preservation = entity_preserver.check_entity_preservation(input_text, output_text)
            print(f"Entity preservation rate: {preservation['preservation_rate']:.2%}")
            if preservation['missing']:
                print(f"Missing entities: {preservation['missing']}")
        
        outputs.append(output_text)
    
    # Save outputs
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for output in outputs:
                f.write(output + '\n')
        
        print(f"\nOutputs saved to {args.output}")


if __name__ == "__main__":
    main()
