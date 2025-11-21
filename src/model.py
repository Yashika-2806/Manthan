"""
Main Model Architecture for Factuality-Preserving Style Transfer
"""

import torch
import torch.nn as nn
from transformers import BartForConditionalGeneration, BartTokenizer
from typing import Dict, List, Optional, Tuple

from .style_module import StyleEmbedding, StyleClassifier
from .entity_preserve import EntityPreserver
from .fact_checker import FactChecker
from .losses import ContrastiveLoss, EntityLoss, CycleLoss


class FactualStyleTransferModel(nn.Module):
    """
    Encoder-Decoder model with factuality preservation for style transfer.
    
    Components:
    - BART/T5 backbone for generation
    - Style conditioning via latent vectors
    - Entity preservation module
    - Fact-checking filters
    - Contrastive semantic loss
    """
    
    def __init__(
        self,
        model_name: str = "facebook/bart-base",
        style_dim: int = 128,
        num_styles: int = 4,  # formal, informal, simple, complex
        entity_preserve_weight: float = 0.2,
        contrastive_weight: float = 0.3,
        temperature: float = 0.07,
    ):
        super().__init__()
        
        # Backbone encoder-decoder
        self.backbone = BartForConditionalGeneration.from_pretrained(model_name)
        self.tokenizer = BartTokenizer.from_pretrained(model_name)
        self.hidden_dim = self.backbone.config.d_model
        
        # Style conditioning
        self.style_embedding = StyleEmbedding(
            num_styles=num_styles,
            style_dim=style_dim,
            hidden_dim=self.hidden_dim
        )
        
        # Style classifier for discriminative signal
        self.style_classifier = StyleClassifier(
            hidden_dim=self.hidden_dim,
            num_styles=num_styles
        )
        
        # Entity preservation module
        self.entity_preserver = EntityPreserver()
        
        # Fact checker (QA + FactCC)
        self.fact_checker = FactChecker()
        
        # Losses
        self.contrastive_loss = ContrastiveLoss(temperature=temperature)
        self.entity_loss = EntityLoss()
        self.cycle_loss = CycleLoss()
        
        # Weights
        self.entity_preserve_weight = entity_preserve_weight
        self.contrastive_weight = contrastive_weight
        
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        target_style: torch.Tensor,
        decoder_input_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        entities: Optional[List[Dict]] = None,
        return_dict: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with multiple loss components.
        
        Args:
            input_ids: Source text token ids [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len]
            target_style: Style indices [batch]
            decoder_input_ids: Target sequence for teacher forcing
            labels: Target labels for loss computation
            entities: List of entity dictionaries per example
            return_dict: Whether to return dictionary
            
        Returns:
            Dictionary with loss components and outputs
        """
        batch_size = input_ids.size(0)
        
        # Encode input
        encoder_outputs = self.backbone.model.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Get style conditioning vector
        style_vector = self.style_embedding(target_style)  # [batch, style_dim]
        
        # Inject style into decoder (prepend to encoder outputs)
        # Method 1: Add style to encoder hidden states
        encoder_hidden_states = encoder_outputs.last_hidden_state
        style_hidden = self.style_embedding.project_to_hidden(style_vector)
        style_hidden = style_hidden.unsqueeze(1)  # [batch, 1, hidden_dim]
        
        # Concatenate style token to encoder outputs
        encoder_hidden_states = torch.cat([style_hidden, encoder_hidden_states], dim=1)
        
        # Adjust attention mask for style token
        style_attention = torch.ones(batch_size, 1, device=attention_mask.device)
        extended_attention_mask = torch.cat([style_attention, attention_mask], dim=1)
        
        # Decode with style conditioning
        if decoder_input_ids is not None:
            outputs = self.backbone(
                attention_mask=extended_attention_mask,
                encoder_outputs=(encoder_hidden_states,),
                decoder_input_ids=decoder_input_ids,
                labels=labels,
                return_dict=True
            )
            gen_loss = outputs.loss
            logits = outputs.logits
        else:
            # Inference mode
            outputs = self.backbone.generate(
                encoder_outputs=(encoder_hidden_states,),
                attention_mask=extended_attention_mask,
                max_length=128,
                num_beams=4,
                early_stopping=True
            )
            gen_loss = None
            logits = None
        
        # Compute additional losses
        loss_dict = {"gen_loss": gen_loss}
        
        # Style classification loss (on decoder outputs)
        if logits is not None:
            # Pool decoder hidden states
            decoder_hidden = outputs.decoder_hidden_states[-1] if hasattr(outputs, 'decoder_hidden_states') else None
            if decoder_hidden is not None:
                pooled_output = decoder_hidden.mean(dim=1)
                style_logits = self.style_classifier(pooled_output)
                style_loss = nn.CrossEntropyLoss()(style_logits, target_style)
                loss_dict["style_loss"] = style_loss
        
        # Entity preservation loss
        if entities is not None and logits is not None:
            entity_loss = self.entity_loss(logits, labels, entities, self.tokenizer)
            loss_dict["entity_loss"] = entity_loss * self.entity_preserve_weight
        
        # Contrastive semantic preservation loss
        if labels is not None:
            # Encode source and target with sentence encoder
            contrastive_loss = self.contrastive_loss(
                input_ids, labels, self.tokenizer
            )
            loss_dict["contrastive_loss"] = contrastive_loss * self.contrastive_weight
        
        # Aggregate total loss
        total_loss = sum(v for v in loss_dict.values() if v is not None)
        loss_dict["total_loss"] = total_loss
        
        if return_dict:
            return {
                "loss": total_loss,
                "loss_dict": loss_dict,
                "logits": logits,
                "outputs": outputs
            }
        else:
            return total_loss
    
    def generate(
        self,
        input_text: str,
        target_style: str,
        max_length: int = 128,
        num_beams: int = 4,
        preserve_entities: bool = True,
        fact_check: bool = True
    ) -> str:
        """
        Generate style-transferred text with factuality preservation.
        
        Args:
            input_text: Source text
            target_style: Target style name (e.g., "formal", "informal")
            max_length: Maximum generation length
            num_beams: Number of beams for beam search
            preserve_entities: Whether to enforce entity preservation
            fact_check: Whether to apply fact-checking filter
            
        Returns:
            Style-transferred text
        """
        self.eval()
        
        # Tokenize input
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128
        )
        input_ids = inputs["input_ids"].to(self.backbone.device)
        attention_mask = inputs["attention_mask"].to(self.backbone.device)
        
        # Extract entities if preservation is enabled
        entities = None
        if preserve_entities:
            entities = self.entity_preserver.extract_entities(input_text)
        
        # Map style name to index
        style_map = {"formal": 0, "informal": 1, "simple": 2, "complex": 3}
        style_idx = torch.tensor([style_map.get(target_style, 0)]).to(self.backbone.device)
        
        # Encode and generate
        with torch.no_grad():
            encoder_outputs = self.backbone.model.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            # Add style conditioning
            style_vector = self.style_embedding(style_idx)
            style_hidden = self.style_embedding.project_to_hidden(style_vector).unsqueeze(1)
            encoder_hidden_states = torch.cat([style_hidden, encoder_outputs.last_hidden_state], dim=1)
            
            # Adjust attention mask
            style_attention = torch.ones(1, 1, device=attention_mask.device)
            extended_attention_mask = torch.cat([style_attention, attention_mask], dim=1)
            
            # Generate with constrained decoding for entities
            if entities and preserve_entities:
                # Custom beam search with entity constraints
                output_ids = self._constrained_generate(
                    encoder_hidden_states,
                    extended_attention_mask,
                    entities,
                    max_length,
                    num_beams
                )
            else:
                output_ids = self.backbone.generate(
                    encoder_outputs=(encoder_hidden_states,),
                    attention_mask=extended_attention_mask,
                    max_length=max_length,
                    num_beams=num_beams,
                    early_stopping=True
                )
        
        # Decode output
        output_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        
        # Apply fact-checking filter
        if fact_check:
            is_factual = self.fact_checker.check_factuality(input_text, output_text)
            if not is_factual:
                print(f"Warning: Factuality check failed. Consider re-generating.")
        
        return output_text
    
    def _constrained_generate(
        self,
        encoder_hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        entities: List[Dict],
        max_length: int,
        num_beams: int
    ) -> torch.Tensor:
        """
        Generate with entity preservation constraints.
        Simplified version - full implementation would use constrained beam search.
        """
        # For now, fall back to standard generation
        # TODO: Implement proper constrained beam search with entity forcing
        output_ids = self.backbone.generate(
            encoder_outputs=(encoder_hidden_states,),
            attention_mask=attention_mask,
            max_length=max_length,
            num_beams=num_beams,
            early_stopping=True
        )
        return output_ids
