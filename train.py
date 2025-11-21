"""
Training script for factual style transfer model.
"""

import torch
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
import yaml
import argparse
from pathlib import Path
from tqdm import tqdm
import wandb
from typing import Dict

from src.model import FactualStyleTransferModel
from src.dataset import StyleTransferDataset
from src.utils import set_seed, save_checkpoint, load_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description="Train factual style transfer model")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--output_dir", type=str, default="checkpoints", help="Output directory")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--wandb", action="store_true", help="Use Weights & Biases logging")
    parser.add_argument("--debug", action="store_true", help="Debug mode with small dataset")
    return parser.parse_args()


def train_epoch(
    model: FactualStyleTransferModel,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
    epoch: int
) -> Dict[str, float]:
    """
    Train for one epoch.
    """
    model.train()
    total_loss = 0.0
    loss_components = {}
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for batch_idx, batch in enumerate(pbar):
        # Move batch to device
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        target_style = batch["target_style"].to(device)
        labels = batch["labels"].to(device)
        decoder_input_ids = batch["decoder_input_ids"].to(device)
        entities = batch.get("entities", None)
        
        # Forward pass
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            target_style=target_style,
            decoder_input_ids=decoder_input_ids,
            labels=labels,
            entities=entities,
            return_dict=True
        )
        
        loss = outputs["loss"]
        loss_dict = outputs["loss_dict"]
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        
        # Accumulate losses
        total_loss += loss.item()
        for key, val in loss_dict.items():
            if val is not None:
                loss_components[key] = loss_components.get(key, 0.0) + val.item()
        
        # Update progress bar
        pbar.set_postfix({"loss": loss.item()})
    
    # Average losses
    num_batches = len(dataloader)
    metrics = {
        "train_loss": total_loss / num_batches,
        **{f"train_{k}": v / num_batches for k, v in loss_components.items()}
    }
    
    return metrics


def evaluate(
    model: FactualStyleTransferModel,
    dataloader: DataLoader,
    device: torch.device
) -> Dict[str, float]:
    """
    Evaluate model on validation set.
    """
    model.eval()
    total_loss = 0.0
    loss_components = {}
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            target_style = batch["target_style"].to(device)
            labels = batch["labels"].to(device)
            decoder_input_ids = batch["decoder_input_ids"].to(device)
            entities = batch.get("entities", None)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                target_style=target_style,
                decoder_input_ids=decoder_input_ids,
                labels=labels,
                entities=entities,
                return_dict=True
            )
            
            loss = outputs["loss"]
            loss_dict = outputs["loss_dict"]
            
            total_loss += loss.item()
            for key, val in loss_dict.items():
                if val is not None:
                    loss_components[key] = loss_components.get(key, 0.0) + val.item()
    
    num_batches = len(dataloader)
    metrics = {
        "val_loss": total_loss / num_batches,
        **{f"val_{k}": v / num_batches for k, v in loss_components.items()}
    }
    
    return metrics


def main():
    args = parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set seed
    set_seed(config.get("seed", 42))
    
    # Initialize wandb
    if args.wandb:
        wandb.init(project="factual-style-transfer", config=config)
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create model
    model = FactualStyleTransferModel(
        model_name=config["model"]["backbone"],
        style_dim=config["model"]["style_dim"],
        num_styles=config["model"]["num_styles"],
        entity_preserve_weight=config["losses"]["entity_weight"],
        contrastive_weight=config["losses"]["contrastive_weight"],
        temperature=config["losses"]["temperature"]
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    
    # Load datasets
    train_dataset = StyleTransferDataset(
        data_path=config["data"]["train_path"],
        tokenizer=model.tokenizer,
        max_length=config["data"]["max_length"],
        extract_entities=config["data"].get("extract_entities", True)
    )
    
    val_dataset = StyleTransferDataset(
        data_path=config["data"]["val_path"],
        tokenizer=model.tokenizer,
        max_length=config["data"]["max_length"],
        extract_entities=config["data"].get("extract_entities", True)
    )
    
    if args.debug:
        train_dataset = torch.utils.data.Subset(train_dataset, range(100))
        val_dataset = torch.utils.data.Subset(val_dataset, range(50))
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=config["training"].get("num_workers", 4)
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=config["training"].get("num_workers", 4)
    )
    
    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"].get("weight_decay", 0.01)
    )
    
    num_training_steps = len(train_loader) * config["training"]["epochs"]
    num_warmup_steps = config["training"].get("warmup_steps", 1000)
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )
    
    # Resume from checkpoint
    start_epoch = 0
    best_val_loss = float('inf')
    
    if args.resume:
        checkpoint = load_checkpoint(args.resume)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_val_loss = checkpoint.get("best_val_loss", float('inf'))
        print(f"Resumed from epoch {start_epoch}")
    
    # Training loop
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for epoch in range(start_epoch, config["training"]["epochs"]):
        print(f"\nEpoch {epoch + 1}/{config['training']['epochs']}")
        
        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, scheduler, device, epoch)
        print(f"Train loss: {train_metrics['train_loss']:.4f}")
        
        # Evaluate
        val_metrics = evaluate(model, val_loader, device)
        print(f"Val loss: {val_metrics['val_loss']:.4f}")
        
        # Log to wandb
        if args.wandb:
            wandb.log({**train_metrics, **val_metrics, "epoch": epoch})
        
        # Save checkpoint
        is_best = val_metrics["val_loss"] < best_val_loss
        if is_best:
            best_val_loss = val_metrics["val_loss"]
        
        save_checkpoint(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "train_loss": train_metrics["train_loss"],
                "val_loss": val_metrics["val_loss"],
                "best_val_loss": best_val_loss,
                "config": config
            },
            is_best=is_best,
            output_dir=output_dir
        )
    
    print(f"\nTraining complete! Best validation loss: {best_val_loss:.4f}")
    
    if args.wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
