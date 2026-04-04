# ABSA Training Quick Start Guide

## 📋 Overview

This guide walks you through training an Aspect-Based Sentiment Analysis (ABSA) model on your Hindi/Hinglish restaurant review data using PyABSA with MLflow experiment tracking.

**What you'll get:**
- Fine-tuned DeBERTa model for Hindi/Hinglish ABSA
- Tracked experiments in MLflow
- Registered model ready for deployment
- Complete metrics and artifacts

## 🚀 Quick Start (5 minutes)

### 1. Verify Environment
```bash
# Check Python version (3.8+)
python --version

# Activate virtual environment if using one
# source venv/bin/activate  # on Linux/Mac
# venv\Scripts\activate     # on Windows

# Install/update dependencies
pip install -r requirements.txt
```

### 2. Verify Data Exists
```bash
# Check data files
ls -la data/processed/train_data.txt
ls -la data/processed/val_data.txt
ls -la data/processed/test_data.txt

# Quick sample
head -3 data/processed/train_data.txt
```

Expected format:
```
text$$$[aspect]$$$sentiment
एक romantic evening क लए शनदर ह$$$[ambiance]$$$positive
```

### 3. Start Training

**Option A: Simple (default config)**
```bash
python run_training.py --task asc
```

**Option B: Custom parameters**
```bash
python run_training.py --task asc --epochs 5 --batch-size 32 --learning-rate 2e-5
```

**Option C: Both ASC and ATE**
```bash
python run_training.py --task both
```

### 4. Monitor Progress
```bash
# In a new terminal, start MLflow UI
mlflow ui

# Open browser to http://localhost:5000
```

### 5. Verify Results
```bash
# Check trained model
ls -la models/

# View training summary
cat models/training_summary.json
```

## 📊 Training Details

### What Gets Trained

**Aspect-Sentiment Classification (ASC)**
- Input: Review text + aspect (e.g., "Great food" + [food])
- Output: Sentiment label (positive, negative, neutral)
- Use case: Sentiment for specific aspects

**Aspect-Term Extraction (ATE)** (optional)
- Input: Review text
- Output: Aspect term locations (e.g., "food" in "Great food")
- Use case: Finding aspects in text

### Key Parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Model | DeBERTa-v3-base-ABSA | Multilingual, optimized for ABSA |
| Epochs | 3-5 | Adjust for overfitting/underfitting |
| Batch Size | 32 | Reduce to 16 if GPU memory issues |
| Learning Rate | 2e-5 | Standard for transformer fine-tuning |
| Max Tokens | 512 | DeBERTa maximum length |
| Language | Hindi/Hinglish | Multilingual model support |

### Expected Results

After training you'll see:
```
- Accuracy: 65-75%
- F1-Score: 0.60-0.70
- Precision: 0.65-0.75
- Recall: 0.60-0.70
```

Actual values depend on data quality and cleanliness.

## 📁 Output Structure

After training, your `models/` directory will contain:

```
models/
├── asc_model/
│   ├── best_model/
│   │   ├── pytorch_model.bin        # Model weights
│   │   ├── config.json              # Model config
│   │   └── special_tokens_map.json  # Tokenizer setup
│   └── asc_model_val.txt            # Validation data
├── training_summary.json             # Final metrics
└── training_results.json             # Complete results
```

## 🔄 MLflow Tracking

### Via CLI
```bash
# Start MLflow server
mlflow server --host 0.0.0.0 --port 5000 &

# Or use default local tracking
mlflow ui
```

### What Gets Logged

**Parameters:**
- Model name and pretrained weights
- Epochs, batch size, learning rate
- Max sequence length

**Metrics:**
- Accuracy, F1, precision, recall
- Loss values per epoch

**Artifacts:**
- Trained model (PyTorch format)
- Training configuration (JSON)
- Model config (JSON)

**Registry:**
- Model registered as: `ABSA-ASC-Deberta-Hindi`
- Version tracking
- Staging/Production promotion

## 🎯 Different Training Scenarios

### Scenario 1: Default Training
```bash
python run_training.py --task asc
```
- Uses config from `config/model.yaml`
- 5 epochs, batch size 32
- Registers model in MLflow

### Scenario 2: Quick Testing (1 epoch)
```bash
python run_training.py --task asc --epochs 1 --dry-run
python run_training.py --task asc --epochs 1
```
- Only trains 1 epoch
- Good for debugging

### Scenario 3: Custom Config
```bash
# Edit config/model.yaml first, then:
python run_training.py --task asc --config config/model.yaml
```

### Scenario 4: Show Config Without Training
```bash
python run_training.py --task asc --dry-run
```
- Shows what would be trained
- No actual training runs
- Useful for verification

### Scenario 5: Export Config
```bash
python run_training.py --task asc --export-config my_config.json
```
- Exports resolved configuration
- Useful for reproducibility

### Scenario 6: Skip MLflow Registration
```bash
python run_training.py --task asc --no-register
```
- Trains and saves model
- Skips MLflow Registry registration

## 🐛 Troubleshooting

### Issue: CUDA Out of Memory
```bash
# Solution: Reduce batch size
python run_training.py --task asc --batch-size 16
```

### Issue: Data file not found
```bash
# Check file exists
test -f data/processed/train_data.txt && echo "Found" || echo "Missing"

# Check format (should have $$$)
head -1 data/processed/train_data.txt | grep '\$\$\$'
```

### Issue: MLflow not tracking
```bash
# Start MLflow server
mlflow server &

# Check it's running
curl http://localhost:5000

# Then run training
python run_training.py --task asc
```

### Issue: Import errors
```bash
# Reinstall dependencies
pip install --upgrade -r requirements.txt

# Or specific package
pip install --upgrade pyabsa transformers mlflow
```

### Issue: CUDA not found
```bash
# Use CPU only (slower)
python run_training.py --task asc --batch-size 8

# Or install CUDA-compatible PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## 📈 Next Steps After Training

### 1. View Results in MLflow
```bash
mlflow ui
# Open http://localhost:5000 → Experiments → ABSA-ASC-Fine-tuning
```

### 2. Make Predictions
```python
from pyabsa import AspectSentimentClassification

model = AspectSentimentClassification("models/asc_model/best_model", task_name="asc")
result = model.predict("Great food", "[food]")
print(result)  # Should show sentiment prediction
```

### 3. Promote to Production
In MLflow UI:
- Go to Models → ABSA-ASC-Deberta-Hindi
- Find your version
- Click "Change Stage" → Production

### 4. Deploy API
```bash
python src/api/main.py
# API available at http://localhost:8000
```

### 5. Further Fine-tuning
```bash
# On new data (with lower learning rate)
python run_training.py --task asc --learning-rate 1e-5 --epochs 2
```

## 📝 Configuration Examples

### For Low-Resource Environment
```yaml
# config/model.yaml
training:
  epochs: 3
  batch_size: 8
  learning_rate: 2e-5
```

### For High Accuracy (expensive)
```yaml
training:
  epochs: 10
  batch_size: 64
  learning_rate: 5e-5
```

### For Multi-GPU Training
```bash
# PyABSA will auto-detect and use
python run_training.py --task asc
```

## ✅ Validation Checklist

Before running training, verify:
- [ ] Python 3.8+ installed
- [ ] All requirements installed: `pip install -r requirements.txt`
- [ ] Data files exist in `data/processed/`
- [ ] Data format correct: `text$$$[aspect]$$$sentiment`
- [ ] Adequate GPU memory (8GB+ recommended) or set smaller batch size
- [ ] MLflow configured (optional but recommended)
- [ ] Config file readable: `config/model.yaml`

## 📞 Getting Help

**For PyABSA Issues:**
- GitHub: https://github.com/yangheng/PyABSA
- Documentation: https://github.com/yangheng/PyABSA/wiki

**For MLflow Issues:**
- Documentation: https://mlflow.org/docs/latest/
- Community: https://discuss.mlflow.org/

**For Project Issues:**
- Check `logs/` directory for training logs
- Review `models/training_results.json` for error details

## 🎉 Success Indicators

After successful training, you should see:
1. ✅ Models saved in `models/` directory
2. ✅ Training summary in `models/training_summary.json`
3. ✅ MLflow experiment with logged metrics
4. ✅ Model registered in MLflow Registry
5. ✅ No errors in training logs

If all checks pass, your model is ready for deployment!

---

**Quick Command Reference:**
```bash
# Train with defaults
python run_training.py --task asc

# Train with custom params
python run_training.py --task asc --epochs 5 --batch-size 16

# Show configuration only
python run_training.py --task asc --dry-run

# Train both ASC and ATE
python run_training.py --task both

# View MLflow UI
mlflow ui
```

For more options, see:
```bash
python run_training.py --help
```

---

**Happy Training! 🚀**
