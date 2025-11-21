# Factuality-Preserving Text Style Transfer - Setup Guide

## Quick Start

### 1. Environment Setup

```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Linux/Mac)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm
```

### 2. Prepare Sample Data

```bash
# Generate sample synthetic data for testing
python scripts/preprocess_data.py --dataset sample --num_samples 1000 --output_dir data/processed
```

### 3. Run Demo

```bash
# Test the model with sample inputs
python demo.py
```

### 4. Train Model

```bash
# Train on formality transfer
python train.py --config configs/formality_transfer.yaml --output_dir checkpoints/formality

# Train with debug mode (small dataset)
python train.py --config configs/formality_transfer.yaml --debug

# Train with wandb logging
python train.py --config configs/formality_transfer.yaml --wandb
```

### 5. Run Inference

```bash
# Single text input
python inference.py --checkpoint checkpoints/formality/best_model.pt --input "hey whats up bro" --target_style formal

# With entity preservation and fact checking
python inference.py --checkpoint checkpoints/formality/best_model.pt --input "obama was born in 1961" --target_style formal --preserve_entities --fact_check

# Process file
python inference.py --checkpoint checkpoints/formality/best_model.pt --input data/test_inputs.txt --output data/test_outputs.txt --target_style formal
```

## Dataset Preparation

### GYAFC (Formality Transfer)

1. Download GYAFC from: https://github.com/raosudha89/GYAFC-corpus
2. Extract to `data/raw/gyafc/`
3. Process:

```bash
python scripts/preprocess_data.py --dataset gyafc --data_dir data/raw/gyafc --output_dir data/processed
```

### Yelp Reviews (Sentiment)

1. Download Yelp dataset
2. Place in `data/raw/yelp/`
3. Process:

```bash
python scripts/preprocess_data.py --dataset yelp --data_dir data/raw/yelp --output_dir data/processed
```

### Wikipedia Simplification

1. Download WikiLarge or Simple English Wikipedia
2. Place in `data/raw/wiki_simple/`
3. Process:

```bash
python scripts/preprocess_data.py --dataset wiki_simple --data_dir data/raw/wiki_simple --output_dir data/processed
```

## Configuration

Edit YAML configs in `configs/` to customize:
- Model architecture (backbone, dimensions)
- Training hyperparameters (learning rate, batch size)
- Loss weights (style, entity, contrastive, etc.)
- Data paths

Example config structure:
```yaml
model:
  backbone: "facebook/bart-base"
  style_dim: 128
  
training:
  batch_size: 32
  learning_rate: 3e-5
  epochs: 10
  
losses:
  style_weight: 0.5
  entity_weight: 0.2
  contrastive_weight: 0.3
```

## Evaluation

Run evaluation script:

```bash
python scripts/evaluate.py --checkpoint checkpoints/formality/best_model.pt --test_data data/processed/test.jsonl
```

Metrics computed:
- Style accuracy (classifier-based)
- BERTScore (semantic similarity)
- Entity preservation rate
- Factuality score (QA + FactCC)
- BLEU (if references available)

## Troubleshooting

### Out of Memory
- Reduce `batch_size` in config
- Use gradient accumulation
- Use smaller backbone (bart-base instead of bart-large)

### Slow Training
- Use GPU if available
- Reduce `max_length` in config
- Use fewer `num_workers` in dataloader

### Poor Style Transfer
- Increase `style_weight` in config
- Train longer (more epochs)
- Use larger backbone model

### Poor Entity Preservation
- Increase `entity_weight` in config
- Enable `preserve_entities` in inference
- Check NER model quality

## Advanced Usage

### Custom Style Transfer

1. Prepare your dataset in JSONL format:
```json
{"source": "input text", "target": "output text", "source_style": "style1", "target_style": "style2"}
```

2. Update config with your data paths

3. Train model

### Fine-tuning

Resume from checkpoint:
```bash
python train.py --config configs/formality_transfer.yaml --resume checkpoints/formality/checkpoint.pt
```

### Multi-GPU Training

Use PyTorch DistributedDataParallel (modify train.py accordingly)

## Citation

```bibtex
@misc{factual-style-transfer-2025,
  title={Factuality-Preserving Text Style Transfer},
  author={Your Name},
  year={2025},
  url={https://github.com/Yashika-2806/text}
}
```

## Support

For issues or questions:
- Open an issue on GitHub
- Check documentation in README.md
- Review example notebooks (coming soon)
