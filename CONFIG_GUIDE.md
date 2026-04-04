# Configuration Guide - config/config.yaml

Complete reference for all hyperparameters and settings in `config/config.yaml`.

## Overview

The unified configuration file allows you to customize **every aspect** of the ABSA training pipeline without modifying code. All parameters are documented with defaults and examples.

## Quick Reference

| Section | Purpose | Key Settings |
|---------|---------|--------------|
| **model** | Model selection and tokenizer | pretrained, max_length |
| **training** | Training hyperparameters | epochs, batch_size, learning_rate |
| **data** | Data paths and preprocessing | paths, encoding, validation |
| **output** | Where to save results | directories, file names |
| **mlflow** | Experiment tracking | experiment_name, register_model |
| **evaluation** | Metrics and labels | metrics, sentiment_labels |
| **hardware** | Device and GPU settings | device, mixed_precision |
| **advanced** | Fine-tuning and debugging | freeze_encoder, debug_mode |

---

## Detailed Configuration Sections

### 🎯 Model Configuration

```yaml
model:
  name: "deberta-absa"
  pretrained: "yangheng/deberta-v3-base-absa-v1.1"
  max_length: 512
  padding: "max_length"
  truncation: True
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | string | "deberta-absa" | Model identifier for logging |
| `pretrained` | string | "yangheng/deberta-v3-base-absa-v1.1" | HuggingFace model ID |
| `max_length` | int | 512 | Max tokens per sample (DeBERTa max is 512) |
| `padding` | string | "max_length" | Padding strategy |
| `truncation` | bool | True | Truncate sequences longer than max_length |

**Alternative Models:**
```yaml
# XLM-RoBERTa (good for Indic languages)
pretrained: "xlm-roberta-base"

# DistilBERT (lighter, faster)
pretrained: "distilbert-base-multilingual-cased"

# mBERT (104 languages)
pretrained: "bert-base-multilingual-cased"
```

---

### 🔧 Training Configuration

```yaml
training:
  epochs: 5
  batch_size: 32
  learning_rate: 2e-5
  early_stopping: True
  early_stopping_patience: 5
```

**Core Parameters:**

| Parameter | Type | Default | Notes |
|-----------|------|---------|-------|
| `epochs` | int | 5 | 3-5 for fine-tuning |
| `batch_size` | int | 32 | Reduce if GPU memory errors |
| `learning_rate` | float | 2e-5 | Standard for fine-tuning |
| `optimizer` | string | "adamw" | adamw, sgd, rmsprop |
| `weight_decay` | float | 0.01 | L2 regularization |

**Learning Rate Tuning:**
```yaml
# Training diverges (loss increases)
learning_rate: 1e-5

# Training too slow
learning_rate: 5e-5

# Unstable training
learning_rate: 1e-5
warmup_steps: 1000
```

**Early Stopping:**
```yaml
early_stopping: True
early_stopping_patience: 5        # Stop if no improvement after 5 epochs
early_stopping_metric: "f1"       # Watch F1-score
early_stopping_mode: "max"        # Higher is better
```

**Batch Size Guide:**
```yaml
# 4GB VRAM
batch_size: 8

# 8GB VRAM
batch_size: 16

# 11GB VRAM
batch_size: 32

# 24GB+ VRAM
batch_size: 64
```

---

### 📊 Data Configuration

```yaml
data:
  train_path: "data/processed/train_data.txt"
  val_path: "data/processed/val_data.txt"
  test_path: "data/processed/test_data.txt"
  separator: "$$$"
  encoding: "utf-8"
```

**Data Format:**
```
text$$$[aspect]$$$sentiment

Example:
Great food$$$[food]$$$positive
Slow service$$$[service]$$$negative
```

**Encoding:**
```yaml
# UTF-8 text (recommended)
encoding: "utf-8"
fallback_encoding: "latin-1"  # Automatic fallback

# Latin-1 text
encoding: "latin-1"
fallback_encoding: "utf-8"
```

**Data Validation:**
```yaml
skip_malformed: True          # Skip unparseable lines
min_text_length: 5            # Minimum 5 characters
max_text_length: 512          # Maximum 512 characters
```

---

### 📁 Output Configuration

```yaml
output:
  model_dir: "models"
  model_name: "asc_model"
  checkpoint_dir: "models/checkpoints"
  results_file: "models/training_results.json"
```

**Output Structure:**
```
models/
├── asc_model/
│   ├── best_model/
│   │   ├── pytorch_model.bin
│   │   └── config.json
│   └── asc_model_val.txt
├── checkpoints/
│   ├── checkpoint-500/
│   └── checkpoint-1000/
├── training_results.json
└── training_summary.json
```

---

### 📊 MLflow Configuration

```yaml
mlflow:
  enabled: True
  tracking_uri: null
  experiment_name: "ABSA-ASC-Fine-tuning"
  register_model: True
  model_registry_name: "ABSA-ASC-Deberta-Hindi"
  model_stage: "Staging"
```

**MLflow Server:**
```yaml
# Local tracking (default)
tracking_uri: null

# Remote server
tracking_uri: "http://mlflow-server:5000"

# With credentials
tracking_uri: "http://user:password@mlflow-server:5000"
```

**Model Registry:**
```yaml
register_model: True           # Auto-register in registry
model_stage: "Staging"         # Staging or Production
```

**Artifact Logging:**
```yaml
log_artifacts: True
artifact_path: "model"         # Subdirectory in run
```

---

### 📈 Evaluation Configuration

```yaml
evaluation:
  metrics:
    - accuracy
    - precision
    - recall
    - f1
  sentiment_labels:
    - positive
    - negative
    - neutral
  confidence_threshold: 0.5
```

**Metrics:**
- `accuracy`: Overall correctness
- `precision`: True positives / all positives
- `recall`: True positives / all actual positives
- `f1`: Harmonic mean of precision and recall

**Sentiment Labels:**
```yaml
sentiment_labels:
  - positive
  - negative  
  - neutral
```

---

### ⚙️ Hardware Configuration

```yaml
hardware:
  device: "cuda"
  device_id: 0
  distributed: False
  num_gpus: 1
  num_workers: 4
```

**Device Selection:**
```yaml
# GPU training (recommended)
device: "cuda"
device_id: 0              # First GPU

# CPU training
device: "cpu"
num_workers: 2            # Fewer workers for CPU

# Specific GPU
device_id: 1              # Use second GPU
```

**Mixed Precision:**
```yaml
# Full precision (standard)
mixed_precision: "no"

# FP16 (faster, less memory)
mixed_precision: "fp16"

# BF16 (better stability)
mixed_precision: "bf16"
```

---

### 🔬 Advanced Configuration

```yaml
advanced:
  freeze_encoder: False
  num_frozen_layers: 0
  augmentation: False
  debug_mode: False
  verbose: True
```

**Fine-tuning Strategies:**

```yaml
# Standard fine-tuning (all layers)
freeze_encoder: False
num_frozen_layers: 0

# Feature extraction (freeze encoder)
freeze_encoder: True
num_frozen_layers: 24         # Freeze all 24 layers

# Gradual unfreezing
num_frozen_layers: 12         # Train only top 12 layers
```

**Debugging:**
```yaml
debug_mode: True              # Extra logging
verbose: True                 # Detailed output
validate_data: True           # Check data before training
validate_config: True         # Validate settings
```

---

## Common Configuration Scenarios

### Scenario 1: Quick Test
```yaml
training:
  epochs: 1
  batch_size: 8
  learning_rate: 2e-5

advanced:
  debug_mode: True
```

Command:
```bash
python run_training.py --task asc
```

### Scenario 2: Production Quality
```yaml
training:
  epochs: 10
  batch_size: 64
  learning_rate: 1e-5
  early_stopping: True
  early_stopping_patience: 5

hardware:
  mixed_precision: "fp16"
```

### Scenario 3: Low Memory (4GB GPU)
```yaml
training:
  batch_size: 8
  gradient_accumulation_steps: 4

hardware:
  mixed_precision: "fp16"
  gradient_checkpointing: True
```

### Scenario 4: XLM-RoBERTa Model
```yaml
model:
  name: "xlm-roberta"
  pretrained: "xlm-roberta-base"
  max_length: 512

training:
  epochs: 5
  batch_size: 16
```

### Scenario 5: Custom MLflow Server
```yaml
mlflow:
  enabled: True
  tracking_uri: "http://mlflow-server:5000"
  experiment_name: "Production-ABSA"
  register_model: True
  model_stage: "Production"
```

---

## Using Configuration in Code

### Load Config in Python
```python
import yaml

with open('config/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Access settings
model_name = config['model']['pretrained']
epochs = config['training']['epochs']
batch_size = config['training']['batch_size']
```

### Override via CLI
```bash
# CLI arguments override config file
python run_training.py --task asc --epochs 10 --batch-size 16
```

### Export Configuration
```bash
# Save resolved config as JSON
python run_training.py --task asc --dry-run --export-config resolved_config.json
```

---

## Validation Rules

**Batch Size:**
- Must be positive integer
- Recommended: 8-64
- Max depends on GPU memory

**Learning Rate:**
- Recommended: 1e-6 to 1e-4
- Higher = faster training, unstable
- Lower = slower training, stable

**Epochs:**
- Recommended: 3-10 for fine-tuning
- 1 for testing
- 20+ for from-scratch training

**Max Length:**
- DeBERTa: max 512
- Others: check model card
- Must match training and inference

---

## Tips & Best Practices

### ✅ DO:
- Start with default config
- Adjust one parameter at a time
- Use validation set metrics
- Enable early stopping
- Log everything with MLflow

### ❌ DON'T:
- Set learning rate > 1e-4
- Set batch size > GPU memory allows
- Change max_length during inference
- Skip validation set
- Use mismatched configs for inference

---

## Troubleshooting

### "CUDA out of memory"
```yaml
training:
  batch_size: 8              # Reduce from 32
  gradient_accumulation_steps: 4

hardware:
  mixed_precision: "fp16"
  gradient_checkpointing: True
```

### "Training diverges (loss → infinity)"
```yaml
training:
  learning_rate: 1e-5        # Reduce learning rate
  max_grad_norm: 0.5         # Gradient clipping
  warmup_steps: 1000         # More warmup
```

### "Training too slow"
```yaml
training:
  learning_rate: 5e-5        # Increase LR
  
hardware:
  mixed_precision: "fp16"    # Use FP16
  num_workers: 8             # More workers
```

### "Model not improving"
```yaml
training:
  early_stopping: False      # Train longer
  epochs: 10                 # More epochs

data:
  skip_malformed: False      # Check data quality
```

---

## File Size & Performance

**Config File Size**: ~5KB (minimal overhead)

**Memory Usage**:
- Config loading: < 1MB
- No impact on training performance
- All settings pre-processed during init

**Loading Time**: < 100ms (negligible)

---

## Version Compatibility

- **YAML Format**: Standard YAML 1.1
- **Python**: 3.8+
- **PyYAML**: 5.x+

---

## Next Steps

1. **Review** default settings in `config/config.yaml`
2. **Customize** for your needs
3. **Validate** with `python run_training.py --task asc --dry-run`
4. **Train** with `python run_training.py --task asc`
5. **Monitor** with `mlflow ui`

---

## Need More Help?

- See main docs: `TRAINING_GUIDE.md`
- Check examples: `IMPLEMENTATION_SUMMARY.md`
- CLI help: `python run_training.py --help`

---

*Last Updated: April 5, 2026*
