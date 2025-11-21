"""
Entity preservation module using NER and copy mechanisms.
"""

import spacy
from typing import List, Dict, Tuple, Optional
import torch


class EntityPreserver:
    """
    Extract and preserve named entities during style transfer.
    """
    
    def __init__(self, model_name: str = "en_core_web_sm"):
        """
        Initialize with spaCy NER model.
        
        Args:
            model_name: spaCy model name
        """
        try:
            self.nlp = spacy.load(model_name)
        except OSError:
            print(f"Downloading spaCy model: {model_name}")
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", model_name])
            self.nlp = spacy.load(model_name)
    
    def extract_entities(self, text: str) -> List[Dict[str, any]]:
        """
        Extract named entities from text.
        
        Args:
            text: Input text
            
        Returns:
            List of entity dictionaries with keys:
                - text: Entity surface form
                - label: Entity type (PERSON, ORG, GPE, etc.)
                - start: Character start position
                - end: Character end position
        """
        doc = self.nlp(text)
        entities = []
        
        for ent in doc.ents:
            entities.append({
                "text": ent.text,
                "label": ent.label_,
                "start": ent.start_char,
                "end": ent.end_char,
                "start_token": ent.start,
                "end_token": ent.end
            })
        
        # Also extract numeric entities
        for token in doc:
            if token.like_num and not any(
                ent["start"] <= token.idx < ent["end"] for ent in entities
            ):
                entities.append({
                    "text": token.text,
                    "label": "NUMBER",
                    "start": token.idx,
                    "end": token.idx + len(token.text),
                    "start_token": token.i,
                    "end_token": token.i + 1
                })
        
        return entities
    
    def check_entity_preservation(
        self,
        source_text: str,
        generated_text: str,
        fuzzy_match: bool = True
    ) -> Dict[str, any]:
        """
        Check if entities are preserved in generated text.
        
        Args:
            source_text: Original text
            generated_text: Generated text
            fuzzy_match: Allow case-insensitive matching
            
        Returns:
            Dictionary with:
                - preserved: List of preserved entities
                - missing: List of missing entities
                - hallucinated: List of entities in output but not input
                - preservation_rate: Float 0-1
        """
        source_entities = self.extract_entities(source_text)
        generated_entities = self.extract_entities(generated_text)
        
        # Normalize for matching
        def normalize(text):
            return text.lower().strip() if fuzzy_match else text.strip()
        
        source_set = {normalize(e["text"]) for e in source_entities}
        generated_set = {normalize(e["text"]) for e in generated_entities}
        
        preserved = source_set & generated_set
        missing = source_set - generated_set
        hallucinated = generated_set - source_set
        
        preservation_rate = len(preserved) / len(source_set) if source_set else 1.0
        
        return {
            "preserved": list(preserved),
            "missing": list(missing),
            "hallucinated": list(hallucinated),
            "preservation_rate": preservation_rate,
            "num_source": len(source_set),
            "num_generated": len(generated_set)
        }
    
    def get_entity_tokens(
        self,
        text: str,
        tokenizer
    ) -> List[Tuple[int, int]]:
        """
        Get token indices for entities using a given tokenizer.
        
        Args:
            text: Input text
            tokenizer: HuggingFace tokenizer
            
        Returns:
            List of (start_idx, end_idx) token ranges for entities
        """
        entities = self.extract_entities(text)
        entity_ranges = []
        
        # Tokenize text
        tokens = tokenizer(text, return_offsets_mapping=True)
        offset_mapping = tokens["offset_mapping"]
        
        for entity in entities:
            entity_start = entity["start"]
            entity_end = entity["end"]
            
            # Find token indices that overlap with entity
            token_start = None
            token_end = None
            
            for idx, (start, end) in enumerate(offset_mapping):
                if start <= entity_start < end and token_start is None:
                    token_start = idx
                if start < entity_end <= end:
                    token_end = idx + 1
                    break
            
            if token_start is not None and token_end is not None:
                entity_ranges.append((token_start, token_end))
        
        return entity_ranges
    
    def create_entity_mask(
        self,
        text: str,
        tokenizer,
        max_length: int = 128
    ) -> torch.Tensor:
        """
        Create binary mask indicating entity positions.
        
        Args:
            text: Input text
            tokenizer: HuggingFace tokenizer
            max_length: Maximum sequence length
            
        Returns:
            Binary tensor [max_length] with 1 for entity tokens
        """
        entity_ranges = self.get_entity_tokens(text, tokenizer)
        mask = torch.zeros(max_length, dtype=torch.float32)
        
        for start, end in entity_ranges:
            if start < max_length:
                mask[start:min(end, max_length)] = 1.0
        
        return mask
