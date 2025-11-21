"""
Dataset classes for style transfer training.
"""

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
from typing import List, Dict, Optional
import json
from pathlib import Path

from .entity_preserve import EntityPreserver


class StyleTransferDataset(Dataset):
    """
    Dataset for style transfer with optional entity extraction.
    
    Expected data format (JSONL):
    {
        "source": "hey how are you",
        "target": "Hello, how are you?",
        "source_style": "informal",
        "target_style": "formal"
    }
    """
    
    def __init__(
        self,
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 128,
        extract_entities: bool = True,
        style_map: Optional[Dict[str, int]] = None
    ):
        """
        Args:
            data_path: Path to JSONL data file
            tokenizer: HuggingFace tokenizer
            max_length: Maximum sequence length
            extract_entities: Whether to extract entities
            style_map: Mapping from style names to indices
        """
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.extract_entities = extract_entities
        
        # Default style mapping
        self.style_map = style_map or {
            "formal": 0,
            "informal": 1,
            "simple": 2,
            "complex": 3
        }
        
        # Entity preserver
        if extract_entities:
            self.entity_preserver = EntityPreserver()
        
        # Load data
        self.examples = self._load_data()
    
    def _load_data(self) -> List[Dict]:
        """Load JSONL data."""
        examples = []
        
        with open(self.data_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                examples.append(data)
        
        return examples
    
    def __len__(self) -> int:
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        example = self.examples[idx]
        
        source_text = example["source"]
        target_text = example["target"]
        target_style = example.get("target_style", "formal")
        
        # Tokenize source
        source_encoded = self.tokenizer(
            source_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # Tokenize target
        target_encoded = self.tokenizer(
            target_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # Prepare decoder input (shift right)
        decoder_input_ids = target_encoded["input_ids"].clone()
        decoder_input_ids[:, 1:] = target_encoded["input_ids"][:, :-1]
        decoder_input_ids[:, 0] = self.tokenizer.bos_token_id
        
        # Labels for loss computation
        labels = target_encoded["input_ids"].clone()
        labels[labels == self.tokenizer.pad_token_id] = -100  # Ignore padding
        
        # Style index
        style_idx = self.style_map.get(target_style, 0)
        
        # Extract entities
        entities = None
        if self.extract_entities:
            entities = self.entity_preserver.extract_entities(source_text)
        
        return {
            "input_ids": source_encoded["input_ids"].squeeze(0),
            "attention_mask": source_encoded["attention_mask"].squeeze(0),
            "decoder_input_ids": decoder_input_ids.squeeze(0),
            "labels": labels.squeeze(0),
            "target_style": torch.tensor(style_idx, dtype=torch.long),
            "entities": entities,
            "source_text": source_text,
            "target_text": target_text
        }


class UnpairedStyleDataset(Dataset):
    """
    Dataset for unpaired style transfer (two separate corpora).
    Used for training with pseudo-parallel data or backtranslation.
    """
    
    def __init__(
        self,
        source_style_path: str,
        target_style_path: str,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 128,
        source_style: str = "informal",
        target_style: str = "formal"
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.source_style = source_style
        self.target_style = target_style
        
        # Load corpora
        self.source_texts = self._load_corpus(source_style_path)
        self.target_texts = self._load_corpus(target_style_path)
        
        # Use minimum length for pairing
        self.length = min(len(self.source_texts), len(self.target_texts))
    
    def _load_corpus(self, path: str) -> List[str]:
        """Load text corpus (one sentence per line)."""
        texts = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                texts.append(line.strip())
        return texts
    
    def __len__(self) -> int:
        return self.length
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        source_text = self.source_texts[idx]
        
        # Tokenize
        source_encoded = self.tokenizer(
            source_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            "input_ids": source_encoded["input_ids"].squeeze(0),
            "attention_mask": source_encoded["attention_mask"].squeeze(0),
            "target_style": torch.tensor(0, dtype=torch.long),  # Placeholder
            "source_text": source_text
        }
