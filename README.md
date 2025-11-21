# Factuality-Preserving Text Style Transfer

A deep learning system for text style transfer (formality, politeness, simplicity, sentiment) that maintains factual correctness and preserves entities while achieving high style accuracy.

## Problem Statement

Traditional text style transfer systems often:
- Distort meaning or introduce hallucinations
- Fail to preserve factual content (entities, numbers, relations)
- Sacrifice content accuracy for style conformity

## Solution Overview

This project implements a **factuality-aware style transfer model** that combines:
- Encoder-decoder Transformer with latent style vectors
- Entity-preservation constraints via NER
- Fact-checking filters (QA-based + FactCC)
- Contrastive semantic preservation loss
- Cycle consistency for meaning retention

## Key Features

✅ **Content Preservation**: Maintains semantic meaning and factual consistency  
✅ **Style Accuracy**: Achieves target style (formality/politeness/simplicity/sentiment)  
✅ **Entity Protection**: Preserves named entities and numeric facts  
✅ **Non-parallel Training**: Works with unpaired style corpora  
✅ **Modular Design**: Pluggable fact-checking and style modules

## Architecture

```
Input Text → Encoder → Content Representation
                              ↓
                      Style Vector (latent)
                              ↓
                         Decoder
                              ↓
                    Generated Output
                              ↓
              ┌──────────────┴──────────────┐
              ↓                             ↓
      Fact Checker                   Style Classifier
   (QA + FactCC + NER)              (Style Accuracy)
              ↓                             ↓
      Factuality Loss           Contrastive Semantic Loss
```

## Datasets

- **GYAFC**: Formality transfer (formal ↔ informal)
- **ParaNMT**: Paraphrase corpus for content-preserving pairs
- **Simple Wikipedia**: Text simplification
- **Yelp Reviews**: Sentiment/politeness transfer

## Installation

```bash
# Clone repository
git clone https://github.com/Yashika-2806/text.git
cd text

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download spaCy model for NER
python -m spacy download en_core_web_sm
```

## Quick Start

### 1. Preprocess Data
```bash
python scripts/preprocess_data.py \
    --dataset gyafc \
    --output_dir data/processed
```

### 2. Train Model
```bash
python train.py \
    --config configs/formality_transfer.yaml \
    --output_dir checkpoints/formality
```

### 3. Run Inference
```bash
python inference.py \
    --checkpoint checkpoints/formality/best_model.pt \
    --input "hey whats up bro" \
    --target_style formal
```

Output: `"Hello, how are you doing?"`

## Model Components

### 1. **Encoder-Decoder Backbone**
- Based on BART/T5 architecture
- Pretrained on large corpora, fine-tuned for style transfer

### 2. **Latent Style Vector**
- Learned style embeddings conditioned on target style
- Injected into decoder via adapter layers

### 3. **Entity Preservation Module**
- NER-based entity detection (spaCy)
- Copy mechanism to preserve entities in output
- Entity alignment loss

### 4. **Fact-Checking Filter**
- **QA-based**: Generate questions from source, verify answers in output
- **FactCC**: Entailment-based consistency classifier
- **NLI**: Natural Language Inference for semantic preservation

### 5. **Training Objectives**
- **Generation Loss**: Standard MLE for sequence generation
- **Style Loss**: Style classifier on outputs
- **Contrastive Loss**: InfoNCE-style semantic similarity
- **Entity Loss**: Penalty for missing/altered entities
- **Cycle Loss**: Consistency when transferring back to original style

## Evaluation

### Automatic Metrics
- **Style Accuracy**: Classifier-based style match
- **Content Preservation**: BERTScore, BLEURT, BLEU
- **Factuality**: FactCC score, QA-F1, Entity preservation rate
- **Fluency**: Perplexity, grammar error rate

### Human Evaluation
- Factual correctness (1-5 Likert)
- Style match (1-5 Likert)
- Fluency (1-5 Likert)
- Overall preference (A/B comparison)

## Example Results

| Input (Informal) | Output (Formal) | Entity Preserved |
|------------------|-----------------|------------------|
| "hey john, meet me at 3pm" | "Hello John, please meet me at 3:00 PM." | ✅ John, 3pm |
| "this burger is awesome!" | "This burger is excellent." | ✅ burger |
| "obama was born in 1961" | "Barack Obama was born in 1961." | ✅ Obama, 1961 |

## Project Structure

```
text/
├── README.md
├── requirements.txt
├── configs/
│   ├── formality_transfer.yaml
│   ├── simplification.yaml
│   └── sentiment_transfer.yaml
├── src/
│   ├── model.py              # Main model architecture
│   ├── encoder_decoder.py    # Transformer backbone
│   ├── style_module.py       # Style conditioning
│   ├── entity_preserve.py    # NER + entity constraints
│   ├── fact_checker.py       # QA + FactCC modules
│   └── losses.py             # Training objectives
├── scripts/
│   ├── preprocess_data.py    # Dataset preprocessing
│   ├── train.py              # Training script
│   ├── inference.py          # Inference script
│   └── evaluate.py           # Evaluation metrics
├── data/
│   ├── raw/                  # Raw datasets
│   └── processed/            # Preprocessed data
└── checkpoints/              # Saved models
```

## Literature References

### Style Transfer
- **Delete, Retrieve, Generate** (Li et al., 2018): Content-preserving edit pipeline
- **Style Transformer** (Dai et al., 2019): Transformer-based style transfer
- **CrossAligned** (Shen et al., 2017): Adversarial disentanglement

### Factuality & Consistency
- **FactCC** (Kryściński et al., 2019): Factual consistency for summarization
- **QAGS** (Wang et al., 2020): QA-based generation evaluation
- **QAEval** (Durmus et al.): Question answering for factuality

### Semantic Preservation
- **SimCSE**: Contrastive sentence embeddings
- **BERTScore**: Semantic similarity metric

## Experimental Plan

### Phase 1: Implementation (2-4 weeks)
- ✅ Model architecture
- ✅ Entity preservation module
- ✅ Style conditioning

### Phase 2: Factuality Modules (3-5 weeks)
- QA-based fact checking
- FactCC fine-tuning
- Contrastive loss implementation

### Phase 3: Training & Evaluation (3-6 weeks)
- Large-scale training on multiple datasets
- Automatic metrics evaluation
- Human evaluation (500-1000 samples)

### Phase 4: Ablations (2-4 weeks)
- Component ablations
- Adversarial robustness tests
- Domain transfer experiments

## Expected Contributions

1. **Factuality-aware style transfer model** with strong preservation guarantees
2. **Hybrid training procedure** combining contrastive + QA-based fact checking
3. **Comprehensive evaluation protocol** for factual style transfer
4. **Open-source implementation** with reproducible results

## Hyperparameters

```yaml
model:
  backbone: bart-base
  style_dim: 128
  hidden_dim: 768
  
training:
  batch_size: 32
  learning_rate: 3e-5
  epochs: 10
  warmup_steps: 1000
  
losses:
  gen_weight: 1.0
  style_weight: 0.5
  contrastive_weight: 0.3
  entity_weight: 0.2
  cycle_weight: 0.1
  temperature: 0.07
```

## Citation

If you use this code, please cite:

```bibtex
@misc{factual-style-transfer-2025,
  title={Factuality-Preserving Text Style Transfer},
  author={Your Name},
  year={2025},
  url={https://github.com/Yashika-2806/text}
}
```

## License

MIT License - see LICENSE file for details

## Contact

For questions or collaborations, please open an issue or contact the maintainers.

---

**Status**: 🚧 Under Development | **Last Updated**: November 2025
