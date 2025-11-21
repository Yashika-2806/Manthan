"""
Training loss functions for factual style transfer.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Optional


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss for semantic preservation.
    Pull source and generated embeddings together, push apart negatives.
    """
    
    def __init__(self, temperature: float = 0.07, model_name: str = "all-MiniLM-L6-v2"):
        super().__init__()
        self.temperature = temperature
        self.sentence_encoder = SentenceTransformer(model_name)
        self.sentence_encoder.eval()  # Freeze encoder
        
    def forward(
        self,
        source_ids: torch.Tensor,
        target_ids: torch.Tensor,
        tokenizer
    ) -> torch.Tensor:
        """
        Compute InfoNCE-style contrastive loss.
        
        Args:
            source_ids: Source token ids [batch, seq_len]
            target_ids: Target token ids [batch, seq_len]
            tokenizer: Tokenizer to decode ids
            
        Returns:
            Contrastive loss scalar
        """
        batch_size = source_ids.size(0)
        
        # Decode to text
        source_texts = [tokenizer.decode(ids, skip_special_tokens=True) for ids in source_ids]
        target_texts = [tokenizer.decode(ids, skip_special_tokens=True) for ids in target_ids]
        
        # Encode with sentence transformer
        with torch.no_grad():
            source_embeds = self.sentence_encoder.encode(
                source_texts,
                convert_to_tensor=True,
                show_progress_bar=False
            )
            target_embeds = self.sentence_encoder.encode(
                target_texts,
                convert_to_tensor=True,
                show_progress_bar=False
            )
        
        # Normalize embeddings
        source_embeds = F.normalize(source_embeds, dim=-1)
        target_embeds = F.normalize(target_embeds, dim=-1)
        
        # Compute similarity matrix
        sim_matrix = torch.matmul(target_embeds, source_embeds.T) / self.temperature
        
        # Positive pairs are on the diagonal
        labels = torch.arange(batch_size, device=sim_matrix.device)
        
        # InfoNCE loss
        loss = F.cross_entropy(sim_matrix, labels)
        
        return loss


class EntityLoss(nn.Module):
    """
    Loss to penalize missing or altered entities.
    """
    
    def __init__(self, weight: float = 1.0):
        super().__init__()
        self.weight = weight
    
    def forward(
        self,
        logits: torch.Tensor,
        target_ids: torch.Tensor,
        entities: List[Dict],
        tokenizer
    ) -> torch.Tensor:
        """
        Compute entity preservation loss.
        
        Args:
            logits: Model output logits [batch, seq_len, vocab_size]
            target_ids: Target token ids [batch, seq_len]
            entities: List of entity dictionaries per example
            tokenizer: Tokenizer for entity token mapping
            
        Returns:
            Entity loss scalar
        """
        batch_size = logits.size(0)
        total_loss = 0.0
        count = 0
        
        for i in range(batch_size):
            if not entities or i >= len(entities):
                continue
            
            example_entities = entities[i]
            if not example_entities:
                continue
            
            # For each entity, check if it appears in target
            for entity in example_entities:
                entity_text = entity["text"]
                entity_tokens = tokenizer.encode(entity_text, add_special_tokens=False)
                
                # Check if entity tokens are in target
                target_tokens = target_ids[i].tolist()
                
                # Simple check: is entity subsequence in target?
                entity_present = self._is_subsequence(entity_tokens, target_tokens)
                
                if not entity_present:
                    # Penalize missing entity
                    # Simple approach: add constant penalty
                    total_loss += 1.0
                    count += 1
        
        if count > 0:
            return self.weight * total_loss / count
        else:
            return torch.tensor(0.0, device=logits.device)
    
    def _is_subsequence(self, subseq: List[int], seq: List[int]) -> bool:
        """Check if subseq is a subsequence of seq."""
        if not subseq:
            return True
        
        subseq_len = len(subseq)
        seq_len = len(seq)
        
        for i in range(seq_len - subseq_len + 1):
            if seq[i:i+subseq_len] == subseq:
                return True
        
        return False


class CycleLoss(nn.Module):
    """
    Cycle-consistency loss: source -> target_style -> source_style ≈ source
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(
        self,
        original_ids: torch.Tensor,
        reconstructed_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute cycle-consistency loss.
        
        Args:
            original_ids: Original token ids [batch, seq_len]
            reconstructed_ids: Reconstructed token ids after round-trip [batch, seq_len]
            
        Returns:
            Reconstruction loss
        """
        # Simple token-level cross-entropy
        loss = F.cross_entropy(
            reconstructed_ids.view(-1, reconstructed_ids.size(-1)),
            original_ids.view(-1),
            ignore_index=-100
        )
        return loss


class StyleLoss(nn.Module):
    """
    Style classification loss for discriminative training.
    """
    
    def __init__(self, num_styles: int):
        super().__init__()
        self.num_styles = num_styles
        self.criterion = nn.CrossEntropyLoss()
    
    def forward(
        self,
        style_logits: torch.Tensor,
        target_styles: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute style classification loss.
        
        Args:
            style_logits: Style predictions [batch, num_styles]
            target_styles: Target style indices [batch]
            
        Returns:
            Style loss
        """
        return self.criterion(style_logits, target_styles)


class AdversarialLoss(nn.Module):
    """
    Adversarial loss for content-style disentanglement.
    Encourages encoder to produce style-agnostic representations.
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(
        self,
        discriminator_logits: torch.Tensor,
        true_styles: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute adversarial loss to fool discriminator.
        
        Args:
            discriminator_logits: Style predictions from discriminator [batch, num_styles]
            true_styles: True style labels [batch]
            
        Returns:
            Adversarial loss (negative of discriminator loss)
        """
        # Fool discriminator by maximizing entropy (uniform distribution)
        batch_size = discriminator_logits.size(0)
        num_styles = discriminator_logits.size(1)
        
        # Target: uniform distribution over styles
        uniform_target = torch.ones_like(discriminator_logits) / num_styles
        
        # KL divergence between predicted and uniform
        log_probs = F.log_softmax(discriminator_logits, dim=-1)
        loss = F.kl_div(log_probs, uniform_target, reduction='batchmean')
        
        return loss
