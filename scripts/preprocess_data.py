"""
Data preprocessing script for style transfer datasets.
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict, Tuple
from tqdm import tqdm
import random


def load_gyafc(data_dir: str, split: str = "train") -> List[Dict]:
    """
    Load GYAFC (Grammarly's Yahoo Answers Formality Corpus).
    
    Args:
        data_dir: Path to GYAFC data directory
        split: train/val/test
        
    Returns:
        List of examples
    """
    examples = []
    data_path = Path(data_dir) / split
    
    # GYAFC has informal -> formal mappings
    informal_path = data_path / "informal"
    formal_path = data_path / "formal"
    
    if not informal_path.exists() or not formal_path.exists():
        print(f"Warning: GYAFC {split} data not found at {data_path}")
        return []
    
    # Read informal sentences
    with open(informal_path / "sentences.txt", 'r', encoding='utf-8') as f:
        informal_lines = [line.strip() for line in f]
    
    # Read formal references (multiple per informal)
    formal_refs = []
    for ref_file in sorted(formal_path.glob("ref*.txt")):
        with open(ref_file, 'r', encoding='utf-8') as f:
            formal_refs.append([line.strip() for line in f])
    
    # Create examples
    for i, informal in enumerate(informal_lines):
        # Pick first reference as target
        if formal_refs and i < len(formal_refs[0]):
            formal = formal_refs[0][i]
            
            examples.append({
                "source": informal,
                "target": formal,
                "source_style": "informal",
                "target_style": "formal"
            })
            
            # Also add reverse direction
            examples.append({
                "source": formal,
                "target": informal,
                "source_style": "formal",
                "target_style": "informal"
            })
    
    return examples


def load_yelp(data_dir: str, split: str = "train") -> List[Dict]:
    """
    Load Yelp sentiment dataset.
    
    Args:
        data_dir: Path to Yelp data
        split: train/val/test
        
    Returns:
        List of examples
    """
    examples = []
    data_path = Path(data_dir) / f"{split}.jsonl"
    
    if not data_path.exists():
        print(f"Warning: Yelp {split} data not found at {data_path}")
        return []
    
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            sentiment = "positive" if data.get("label", 1) == 1 else "negative"
            
            examples.append({
                "source": data["text"],
                "target": data["text"],  # Self-reconstruction for unpaired
                "source_style": sentiment,
                "target_style": sentiment
            })
    
    return examples


def load_wiki_simple(data_dir: str, split: str = "train") -> List[Dict]:
    """
    Load Wikipedia simplification dataset.
    
    Args:
        data_dir: Path to wiki simple data
        split: train/val/test
        
    Returns:
        List of examples
    """
    examples = []
    data_path = Path(data_dir) / f"{split}.jsonl"
    
    if not data_path.exists():
        print(f"Warning: Wiki simple {split} data not found at {data_path}")
        return []
    
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            
            examples.append({
                "source": data["complex"],
                "target": data["simple"],
                "source_style": "complex",
                "target_style": "simple"
            })
    
    return examples


def save_jsonl(examples: List[Dict], output_path: str):
    """Save examples to JSONL file."""
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for example in examples:
            f.write(json.dumps(example, ensure_ascii=False) + '\n')
    
    print(f"Saved {len(examples)} examples to {output_path}")


def create_sample_data(output_dir: str, num_samples: int = 100):
    """
    Create sample synthetic data for testing.
    
    Args:
        output_dir: Output directory
        num_samples: Number of samples to generate
    """
    # Sample informal -> formal pairs
    informal_formal_pairs = [
        ("hey whats up", "Hello, how are you?"),
        ("gonna do it later", "I will complete it later."),
        ("thats cool", "That is interesting."),
        ("wanna go out", "Would you like to go out?"),
        ("dunno what to do", "I do not know what to do."),
        ("gotta run", "I must leave now."),
        ("nah im good", "No, thank you."),
        ("yeah sure thing", "Yes, certainly."),
        ("cant make it", "I cannot attend."),
        ("lemme know", "Please inform me."),
    ]
    
    examples = []
    
    for i in range(num_samples):
        # Sample a pair
        informal, formal = random.choice(informal_formal_pairs)
        
        # Add some variation
        if random.random() > 0.5:
            # Informal -> formal
            examples.append({
                "source": informal,
                "target": formal,
                "source_style": "informal",
                "target_style": "formal"
            })
        else:
            # Formal -> informal
            examples.append({
                "source": formal,
                "target": informal,
                "source_style": "formal",
                "target_style": "informal"
            })
    
    # Split into train/val/test
    random.shuffle(examples)
    
    train_split = int(0.7 * len(examples))
    val_split = int(0.85 * len(examples))
    
    train_examples = examples[:train_split]
    val_examples = examples[train_split:val_split]
    test_examples = examples[val_split:]
    
    # Save
    output_path = Path(output_dir)
    save_jsonl(train_examples, output_path / "train.jsonl")
    save_jsonl(val_examples, output_path / "val.jsonl")
    save_jsonl(test_examples, output_path / "test.jsonl")


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess style transfer data")
    parser.add_argument("--dataset", type=str, default="sample",
                       choices=["gyafc", "yelp", "wiki_simple", "sample"],
                       help="Dataset to process")
    parser.add_argument("--data_dir", type=str, default="data/raw",
                       help="Input data directory")
    parser.add_argument("--output_dir", type=str, default="data/processed",
                       help="Output directory")
    parser.add_argument("--num_samples", type=int, default=1000,
                       help="Number of samples for synthetic data")
    return parser.parse_args()


def main():
    args = parse_args()
    
    print(f"Processing {args.dataset} dataset...")
    
    if args.dataset == "sample":
        # Create sample synthetic data
        create_sample_data(args.output_dir, args.num_samples)
    elif args.dataset == "gyafc":
        # Process GYAFC
        for split in ["train", "val", "test"]:
            examples = load_gyafc(args.data_dir, split)
            if examples:
                output_path = Path(args.output_dir) / f"gyafc_{split}.jsonl"
                save_jsonl(examples, output_path)
    elif args.dataset == "yelp":
        # Process Yelp
        for split in ["train", "val", "test"]:
            examples = load_yelp(args.data_dir, split)
            if examples:
                output_path = Path(args.output_dir) / f"yelp_{split}.jsonl"
                save_jsonl(examples, output_path)
    elif args.dataset == "wiki_simple":
        # Process Wiki simplification
        for split in ["train", "val", "test"]:
            examples = load_wiki_simple(args.data_dir, split)
            if examples:
                output_path = Path(args.output_dir) / f"wiki_simple_{split}.jsonl"
                save_jsonl(examples, output_path)
    
    print("Preprocessing complete!")


if __name__ == "__main__":
    main()
