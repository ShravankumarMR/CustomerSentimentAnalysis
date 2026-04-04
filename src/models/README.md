# ABSA Models Module

Complete Aspect-Based Sentiment Analysis (ABSA) training and inference pipeline for Hindi/Hinglish restaurant reviews.

## 📚 Module Contents

### Training Scripts

| File | Purpose | Usage |
|------|---------|-------|
| [`train_model.py`](train_model.py) | Core training logic with PyABSA | `from src.models.train_model import ABSATrainer` |
| [`train.py`](train.py) | Entry point for training | `python src/models/train.py` |
| [`predict.py`](predict.py) | Inference/predictions on new data | `python src/models/predict.py --model models/asc_model/best_model --text "Great food"` |
| [`TRAINING_GUIDE.md`](TRAINING_GUIDE.md) | Detailed training documentation | Read for comprehensive guide |

### From Project Root

| File | Purpose |
|------|---------|
| [`run_training.py`](../../run_training.py) | CLI interface for training with options |
| [`TRAINING_QUICKSTART.md`](../../TRAINING_QUICKSTART.md) | Quick start guide (5 min setup) |

## 🚀 Quick Start

### 1️⃣ Train Model (Simplest Way)
```bash
# From project root
python run_training.py --task asc

# Or from models directory
python train.py
```

### 2️⃣ View Results
```bash
# Start MLflow UI
mlflow ui
# Open http://localhost:5000
```

### 3️⃣ Make Predictions
```bash
python src/models/predict.py \
    --model models/asc_model/best_model \
    --text "Great food and friendly staff" \
    --aspects food service
```

## 📖 Documentation

### For Training
- **Quick Start**: See [`TRAINING_QUICKSTART.md`](../../TRAINING_QUICKSTART.md) in project root
- **Detailed Guide**: See [`TRAINING_GUIDE.md`](TRAINING_GUIDE.md) in this directory
- **CLI Options**: Run `python run_training.py --help`

### For Inference
- **Prediction Guide**: See next section
- **CLI Options**: Run `python src/models/predict.py --help`

## 🎯 Training Workflow

### Step 1: Verify Setup
```bash
# Check data exists
ls -la data/processed/train_data.txt
ls -la data/processed/val_data.txt
ls -la data/processed/test_data.txt

# Check format
head -1 data/processed/train_data.txt
```

Expected format: `text$$$[aspect]$$$sentiment`

### Step 2: Configure (Optional)
Edit `config/model.yaml`:
```yaml
training:
  epochs: 5              # Increase for better accuracy
  batch_size: 32         # Decrease if GPU memory issues
  learning_rate: 2e-5    # Good for fine-tuning
```

### Step 3: Train
```bash
# Simple
python run_training.py --task asc

# With custom params
python run_training.py --task asc --epochs 5 --batch-size 32

# Both ASC and ATE
python run_training.py --task both

# Dry run (show config only)
python run_training.py --task asc --dry-run
```

### Step 4: Monitor
```bash
# In separate terminal
mlflow ui
# View at http://localhost:5000
```

### Step 5: Review Results
```bash
# Check metrics
cat models/training_summary.json

# Check model exists
ls -la models/asc_model/best_model/
```

### Step 6: Make Predictions
```bash
python src/models/predict.py \
    --model models/asc_model/best_model \
    --task asc \
    --interactive
```

## 💡 Common Use Cases

### Use Case 1: Train with Defaults
```bash
python run_training.py --task asc
```
- Trains for 5 epochs
- Batch size 32
- Registers in MLflow
- Fast and simple

### Use Case 2: Quick Test
```bash
python run_training.py --task asc --epochs 1
```
- Single epoch for testing
- Verify everything works
- Quick turnaround

### Use Case 3: Production Training
```bash
python run_training.py \
    --task asc \
    --epochs 5 \
    --batch-size 64 \
    --learning-rate 1e-5
```
- More epochs for accuracy
- Larger batch size
- Lower learning rate for stability

### Use Case 4: Low Resource Training
```bash
python run_training.py \
    --task asc \
    --batch-size 8 \
    --device cpu
```
- Smaller batches
- CPU training only
- Takes longer but uses less memory

### Use Case 5: Batch Predictions
```bash
# Create input file with reviews (one per line)
echo "Great food" > reviews.txt
echo "Slow service" >> reviews.txt

# Run predictions on batch
python src/models/predict.py \
    --model models/asc_model/best_model \
    --task asc \
    --input-file reviews.txt \
    --output-file results.json

# View results
cat results.json
```

### Use Case 6: Interactive Testing
```bash
python src/models/predict.py \
    --model models/asc_model/best_model \
    --task asc \
    --interactive

# Then in interactive mode:
# > Great food | food
# > Slow service | service
# > quit
```

## 🔧 Advanced Configuration

### Custom Model
```python
from src.models.train_model import ABSATrainer

trainer = ABSATrainer()
# Override configuration
trainer.config['model']['pretrained'] = 'xlm-roberta-base'

metrics = trainer.train_asc_model(...)
```

### Different Language/Domain
```bash
# Update config for different language/domain
python run_training.py \
    --task asc \
    --pretrained xlm-roberta-base \
    --train-data custom_data/train.txt \
    --val-data custom_data/val.txt \
    --test-data custom_data/test.txt
```

### MLflow Tracking Server
```bash
# Use remote MLflow server
python run_training.py \
    --task asc \
    --mlflow-uri http://mlflow-server:5000

# Or custom username/password
export MLFLOW_TRACKING_USERNAME=user
export MLFLOW_TRACKING_PASSWORD=pass
python run_training.py --task asc
```

## 📊 What Gets Trained

### Model: DeBERTa-v3-base-ABSA
- **Architecture**: DeBERTa-v3-base with ABSA-specific heads
- **Parameters**: ~135M
- **Languages**: Multilingual (including Hindi)
- **Max Length**: 512 tokens
- **Pretrained On**: General corpus + ABSA examples

### Tasks

#### ASC (Aspect-Sentiment Classification)
- **Input**: Review text + aspect term
- **Output**: Sentiment label (positive/negative/neutral)
- **Uses**: Classify sentiments for specific aspects
- **Example**:
  ```
  Input:  "Great food and slow service" + [food]
  Output: positive
  ```

#### ATE (Aspect Term Extraction)  
- **Input**: Review text
- **Output**: Aspect terms and positions
- **Uses**: Find what aspects are mentioned
- **Example**:
  ```
  Input:  "Great food and slow service"
  Output: [food, service]
  ```

## 📈 MLflow Integration

### What Gets Tracked

**Hyperparameters:**
```
- model_name: deberta-absa
- pretrained_model: yangheng/deberta-v3-base-absa-v1.1
- epochs: 5
- batch_size: 32
- learning_rate: 2e-5
- max_length: 512
```

**Metrics:**
```
- accuracy: 0.72
- f1_score: 0.68
- precision: 0.70
- recall: 0.67
```

**Artifacts:**
```
- pytorch_model.bin (trained weights)
- config.json (model configuration)
- training_config.json (hyperparameters)
```

**Registry:**
```
- Model Name: ABSA-ASC-Deberta-Hindi
- Version: Auto-incremented (1, 2, 3, ...)
- Stage: Staging → Production
```

### Access Experiments
```bash
# View in UI
mlflow ui
# http://localhost:5000

# Or programmatically
import mlflow

# Get experiment
exp = mlflow.get_experiment_by_name("ABSA-ASC-Fine-tuning")

# Get best run
runs = mlflow.search_runs(experiment_ids=[exp.experiment_id])
best_run = runs.sort_values('metrics.f1').iloc[-1]
print(f"Best F1: {best_run['metrics.f1']}")
print(f"Best Accuracy: {best_run['metrics.accuracy']}")
```

## 🔄 Complete Workflow Example

```bash
# 1. Start MLflow (in background)
mlflow ui &

# 2. Check data
head data/processed/train_data.txt

# 3. Preview config
python run_training.py --task asc --dry-run

# 4. Train model
python run_training.py --task asc --epochs 5

# 5. Check results
cat models/training_summary.json

# 6. Make a prediction
python src/models/predict.py \
    --model models/asc_model/best_model \
    --task asc \
    --text "Great food, but expensive" \
    --aspects food price

# 7. View in MLflow UI
# Open http://localhost:5000
# Go to Experiments → ABSA-ASC-Fine-tuning
```

## 📁 Output Structure

After training:
```
models/
├── asc_model/
│   ├── best_model/
│   │   ├── pytorch_model.bin           # Model weights
│   │   ├── config.json                 # Model config
│   │   ├── special_tokens_map.json     # Tokenizer
│   │   └── ...
│   └── asc_model_val.txt               # Formatted data
├── training_summary.json               # Final metrics
└── training_results.json               # Detailed results

mlruns/
├── 0/                                  # Experiment ID
│   └── <run-id>/                       # Run directory
│       ├── params/                     # Hyperparameters
│       ├── metrics/                    # Metrics
│       └── artifacts/                  # Model artifacts
```

## ⚠️ Troubleshooting

### Training Issues

**CUDA Out of Memory**
```bash
python run_training.py --task asc --batch-size 8
```

**Data Not Found**
```bash
# Verify files exist
python -c "
import os
files = ['data/processed/train_data.txt', 'data/processed/val_data.txt', 'data/processed/test_data.txt']
for f in files:
    print(f'{f}: {os.path.exists(f)}')"
```

**Import Errors**
```bash
pip install --upgrade -r requirements.txt
```

**MLflow Issues**
```bash
# Start MLflow explicitly
mlflow server &

# Check it's running
curl http://localhost:5000
```

### Inference Issues

**Model Not Loading**
```bash
# Check model path exists
ls -la models/asc_model/best_model/

# Check it's a valid PyABSA model
python -c "
from pyabsa import AspectSentimentClassification
model = AspectSentimentClassification('models/asc_model/best_model', task_name='asc')
print('Model loaded successfully')"
```

**Bad Predictions**
```bash
# Verify model was trained properly
cat models/training_summary.json

# Check input format
# For ASC: text should match training format
# For ATE: text can be any review
```

## 🎓 Learning Resources

- **PyABSA**: [GitHub](https://github.com/yangheng/PyABSA) | [Paper](https://arxiv.org/abs/2010.06705)
- **MLflow**: [Documentation](https://mlflow.org/docs/latest/)
- **DeBERTa**: [Paper](https://arxiv.org/abs/2006.03654)
- **ABSA Survey**: [Link](https://arxiv.org/abs/2010.06705)

## 📞 Support

**For Questions:**
- Check documentation in this file and linked guides
- Review [`TRAINING_GUIDE.md`](TRAINING_GUIDE.md) for detailed info
- Check logs in `models/training_results.json`

**For Issues:**
- Create detailed error reports
- Include output of `python run_training.py --help`
- Share relevant config and data samples

## 📝 License

Uses open-source components:
- PyABSA: License as per original project
- MLflow: Apache 2.0
- DeBERTa: MIT
- Transformers: Apache 2.0

---

**Ready to train? Start with:**
```bash
python run_training.py --task asc
```

**Questions? Check:**
- [`TRAINING_QUICKSTART.md`](../../TRAINING_QUICKSTART.md) - 5 min setup
- [`TRAINING_GUIDE.md`](TRAINING_GUIDE.md) - Complete reference
- `python run_training.py --help` - CLI options
- `python src/models/predict.py --help` - Inference options
