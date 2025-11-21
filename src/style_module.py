"""
Style conditioning modules for text style transfer.
"""

import torch
import torch.nn as nn
from typing import Optional


class StyleEmbedding(nn.Module):
    """
    Learnable style embeddings that condition the decoder on target style.
    """
    
    def __init__(self, num_styles: int, style_dim: int, hidden_dim: int):
        super().__init__()
        self.num_styles = num_styles
        self.style_dim = style_dim
        self.hidden_dim = hidden_dim
        
        # Learnable style embeddings
        self.embeddings = nn.Embedding(num_styles, style_dim)
        
        # Project style to hidden dimension for decoder conditioning
        self.style_projection = nn.Sequential(
            nn.Linear(style_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh()
        )
        
    def forward(self, style_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            style_ids: Style indices [batch_size]
        Returns:
            Style embeddings [batch_size, style_dim]
        """
        return self.embeddings(style_ids)
    
    def project_to_hidden(self, style_vectors: torch.Tensor) -> torch.Tensor:
        """
        Project style vectors to hidden dimension.
        
        Args:
            style_vectors: Style vectors [batch_size, style_dim]
        Returns:
            Projected vectors [batch_size, hidden_dim]
        """
        return self.style_projection(style_vectors)


class StyleClassifier(nn.Module):
    """
    Discriminative style classifier for adversarial training and evaluation.
    """
    
    def __init__(self, hidden_dim: int, num_styles: int, dropout: float = 0.1):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_styles)
        )
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Classify style from hidden representations.
        
        Args:
            hidden_states: Hidden states [batch_size, hidden_dim]
        Returns:
            Style logits [batch_size, num_styles]
        """
        return self.classifier(hidden_states)


class StyleDiscriminator(nn.Module):
    """
    Adversarial discriminator to enforce style-content disentanglement.
    Used to make encoder outputs style-agnostic.
    """
    
    def __init__(self, hidden_dim: int, num_styles: int):
        super().__init__()
        self.discriminator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim // 4, num_styles)
        )
        
    def forward(self, encoder_hidden: torch.Tensor) -> torch.Tensor:
        """
        Predict style from encoder outputs (for adversarial loss).
        
        Args:
            encoder_hidden: Encoder hidden states [batch_size, seq_len, hidden_dim]
        Returns:
            Style predictions [batch_size, num_styles]
        """
        # Pool over sequence dimension
        pooled = encoder_hidden.mean(dim=1)  # [batch_size, hidden_dim]
        return self.discriminator(pooled)
