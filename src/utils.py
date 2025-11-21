"""
Utility functions for training and evaluation.
"""

import torch
import random
import numpy as np
from pathlib import Path
from typing import Dict


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_checkpoint(
    state: Dict,
    is_best: bool,
    output_dir: Path,
    filename: str = "checkpoint.pt"
):
    """
    Save model checkpoint.
    
    Args:
        state: State dictionary with model, optimizer, etc.
        is_best: Whether this is the best model so far
        output_dir: Output directory
        filename: Checkpoint filename
    """
    checkpoint_path = output_dir / filename
    torch.save(state, checkpoint_path)
    
    if is_best:
        best_path = output_dir / "best_model.pt"
        torch.save(state, best_path)
        print(f"Saved best model to {best_path}")


def load_checkpoint(checkpoint_path: str) -> Dict:
    """
    Load checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint
        
    Returns:
        State dictionary
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    return checkpoint


def count_parameters(model: torch.nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def format_time(seconds: float) -> str:
    """Format seconds to human-readable time."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"
