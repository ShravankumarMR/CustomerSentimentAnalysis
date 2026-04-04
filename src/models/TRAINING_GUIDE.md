# ABSA Training Module Documentation

## Overview

The `train_model.py` module provides an end-to-end pipeline for fine-tuning Aspect-Based Sentiment Analysis (ABSA) models using PyABSA framework. It's designed specifically for Hindi/Hinglish restaurant review data with comprehensive MLflow experiment tracking.

## Features

✅ **Multilingual ABSA Support**
- Pre-trained model: `yangheng/deberta-v3-base-absa-v1.1` (DeBERTa-v3-base)
- Optimized for Hindi/Hinglish text
- Supports custom Indic-friendly models

✅ **Two ABSA Tasks**
- **ASC (Aspect-Sentiment Classification)**: Classify sentiment for given aspects
- **ATE (Aspect Term Extraction)**: Extract aspect terms from reviews

✅ **MLflow Integration**
- Track all hyperparameters
- Log training metrics (accuracy, F1, precision, recall)
- Save model artifacts
- Register models in MLflow Registry
- Automatic experiment organization

✅ **Data Format**
```
text$$$[aspect]$$$sentiment_label
Example: "Great food$$$[food]$$$positive"
```

✅ **Robust Error Handling**
- UTF-8 and Latin-1 encoding support
- Data validation and malformed line skipping
- Graceful exception handling

## Installation Requirements

All dependencies are in `requirements.txt`. Ensure you have:

```bash
pip install -r requirements.txt
```

Key packages:
- `pyabsa>=1.9.0` - ABSA framework
- `transformers>=4.30.0` - HuggingFace models
- `mlflow>=2.2.0` - Experiment tracking
- `torch>=2.0.0` - Deep learning
- `pandas>=1.5.0` - Data handling

## Data Format

The training data must be in the following format (one sample per line):

```
text$$$[aspect]$$$sentiment
```

### Examples:
```
एक romantic evening क लए शनदर ह$$$[ambiance]$$$positive
हमर waiters friendly थ$$$[staff]$$$positive
बहत fast service$$$[service]$$$neutral
```

**Available in project:**
- `data/processed/train_data.txt` - Training set (~70% of data)
- `data/processed/val_data.txt` - Validation set (~15% of data)
- `data/processed/test_data.txt` - Test set (~15% of data)

## Configuration

The training is configured in `config/model.yaml`:

```yaml
model:
  name: "deberta-absa"
  pretrained: "yangheng/deberta-v3-base-absa-v1.1"
  max_length: 512

training:
  epochs: 5              # 3-5 recommended
  batch_size: 32         # Adjust based on GPU memory
  learning_rate: 2e-5    # Standard for fine-tuning
  warmup_steps: 500      # For stable training
```

### Recommended Settings

| Parameter | Recommended | Notes |
|-----------|-------------|-------|
| `epochs` | 3-5 | Typical for fine-tuning |
| `batch_size` | 16/32 | 32 for 11GB+ VRAM, 16 for 4-8GB |
| `learning_rate` | 2e-5 | Standard for transformer fine-tuning |
| `max_length` | 512 | DeBERTa max sequence length |

## Usage

### Basic Training

```bash
# Run from project root
python src/models/train.py
```

Or directly:

```bash
python src/models/train_model.py
```

### Advanced Usage with Custom Config

```python
from src.models.train_model import ABSATrainer

# Initialize trainer
trainer = ABSATrainer(config_path="path/to/config.yaml")

# Train ASC (Aspect-Sentiment Classification)
metrics = trainer.train_asc_model(
    train_path="data/processed/train_data.txt",
    val_path="data/processed/val_data.txt",
    test_path="data/processed/test_data.txt",
    output_dir="models/asc_model",
    mlflow_experiment="My-ABSA-Experiment"
)

# Or train ATE (Aspect Term Extraction)
metrics = trainer.train_ate_model(
    train_path="data/processed/train_data.txt",
    val_path="data/processed/val_data.txt",
    test_path="data/processed/test_data.txt"
)
```

## Training Pipeline Output

### 1. **Model Artifacts**
```
models/
├── asc_model/
│   ├── best_model/
│   │   ├── pytorch_model.bin
│   │   ├── config.json
│   │   └── other_model_files
│   ├── training_config.json
│   └── asc_model_val.txt
└── training_summary.json
```

### 2. **MLflow Artifacts**
- **Parameters**: Model name, epochs, batch size, learning rate, etc.
- **Metrics**: Accuracy, F1, precision, recall
- **Model**: Complete trained model in MLflow format
- **Config**: Training configuration JSON

### 3. **Registered Models**
In MLflow Registry:
- **Model Name**: `ABSA-ASC-Deberta-Hindi`
- **Version**: Auto-incremented
- **Stage**: Staging (ready for promotion to Production)

## Monitoring with MLflow

### View experiments:
```bash
mlflow ui
# Open http://localhost:5000
```

### Key Pages in MLflow UI:
1. **Experiments**: `ABSA-ASC-Fine-tuning` - Main training runs
2. **Runs**: Individual training runs with metrics and params
3. **Models**: Registered models in the registry
4. **Model Versions**: Version history with staging info

### Access Model in Registry:
```python
import mlflow.pyfunc

# Load from registry
model = mlflow.pyfunc.load_model("models:/ABSA-ASC-Deberta-Hindi/Staging")

# Make predictions
predictions = model.predict(text_data)
```

## Troubleshooting

### Issue: CUDA Out of Memory
**Solution**: Reduce `batch_size` in config
```yaml
training:
  batch_size: 16  # Instead of 32
```

### Issue: Data file not found
**Solution**: Ensure data files exist at specified paths
```bash
ls data/processed/train_data.txt
ls data/processed/val_data.txt
ls data/processed/test_data.txt
```

### Issue: MLflow not tracking
**Solution**: Ensure MLflow is properly initialized
```bash
# Check MLflow server
mlflow ui &
# Then run training
python src/models/train.py
```

### Issue: Model not registering in MLflow
**Solution**: Check if model name already exists (will show warning but continue)
- Check MLflow UI for existing models
- Versions are auto-incremented

### Issue: Encoding errors
**Solution**: Module handles UTF-8 and Latin-1 automatically
- Logs will show "UTF-8 decoding failed, trying latin-1"
- Malformed lines are skipped with warnings

## Performance Expectations

### Training Time (on typical GPU):
- **V100/A100**: ~2-3 minutes per epoch
- **RTX 3090/4090**: ~3-5 minutes per epoch  
- **CPU Only**: ~20-30 minutes per epoch (not recommended)

### Expected Metrics:
- **Accuracy**: 60-75% (depends on data quality)
- **F1-score**: 0.55-0.70
- **Precision**: 0.60-0.75
- **Recall**: 0.55-0.70

## Model Details

### Model: DeBERTa-v3-base-ABSA
- **Architecture**: DeBERTa-v3-base with ABSA heads
- **Parameters**: ~135M
- **Languages**: Multilingual (Hindi support)
- **Input**: Max 512 tokens
- **Output**: Sentiment classification per aspect

## Next Steps

### 1. Inference
Use the trained model for predictions:
```python
from pyabsa import AspectSentimentClassification

model = AspectSentimentClassification(
    "pretrained_model_path",
    task_name="asc"
)
result = model.predict(
    "Great food",
    aspects="[food]"
)
```

### 2. Model Deployment
Deploy via FastAPI:
```bash
python src/api/main.py
```

### 3. Further Fine-tuning
- Use `production_finetuning_data.txt` for continued training
- Lower learning rate (1e-5) for production data
- Monitor metrics closely

### 4. A/B Testing
- Compare `ABSA-ASC-Deberta-Hindi` with other models
- Use MLflow to track comparison metrics
- Promote best model to Production stage

## Advanced Configuration

### Custom Model
```yaml
model:
  pretrained: "xlm-roberta-base"  # Or other multilingual model
```

### Different Learning Schedule
```yaml
training:
  learning_rate: 1e-5          # Lower for more stable training
  warmup_steps: 1000           # More warming
  epochs: 10                   # Longer training
```

### Enable Gradient Accumulation
```python
# In trainer class, modify training_args:
training_args = {
    "gradient_accumulation_steps": 4,  # Simulate batch size 128
    # ... other args
}
```

## Support & Issues

- **PyABSA Issues**: See [PyABSA GitHub](https://github.com/yangheng/PyABSA)
- **MLflow Documentation**: [MLflow Docs](https://mlflow.org/docs/latest/)
- **DeBERTa Model**: [HuggingFace Model Card](https://huggingface.co/yangheng/deberta-v3-base-absa-v1.1)

## License & Attribution

This training module uses:
- **PyABSA**: Open source ABSA framework
- **DeBERTa**: Microsoft Research model
- **MLflow**: Open source MLOps platform

All distributed under respective open-source licenses.
