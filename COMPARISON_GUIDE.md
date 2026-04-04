# Model Comparison & MLflow Guide

Complete guide for comparing ABSA models with different hyperparameters and using MLflow for experiment tracking.

## Overview

The model comparison pipeline allows you to:
- ✅ Train multiple models with different hyperparameters
- ✅ Compare results side-by-side in MLflow UI
- ✅ Automatically identify the best model
- ✅ Promote best model to Production
- ✅ Generate detailed comparison reports

---

## Quick Start (2 Minutes)

### Option 1: Windows PowerShell (Easiest)
```powershell
# Default: compare 3 learning rates (1e-5, 2e-5, 5e-5)
.\compare.ps1

# Custom learning rates
.\compare.ps1 -LearningRates 1e-5,3e-5,1e-4

# Just start MLflow UI (no training)
.\compare.ps1 -NoTrain

# Help
.\compare.ps1 -Help
```

### Option 2: Python (All Platforms)
```bash
# Default comparison
python start_mlflow_and_compare.py

# Custom learning rates
python start_mlflow_and_compare.py --learning-rates 1e-5 3e-5 1e-4

# Just start MLflow UI
python start_mlflow_and_compare.py --no-train

# Show help
python start_mlflow_and_compare.py --help
```

### Option 3: Manual Control
```bash
# Terminal 1: Start MLflow UI
mlflow ui

# Terminal 2: Run comparison
python compare_models.py

# Terminal 3 (browser)
# Open http://localhost:5000
```

---

## What Happens

### During Training
```
Run 1: Learning Rate = 1e-5
  Training... 100% [████████████████] 3 epochs
  
Run 2: Learning Rate = 2e-5
  Training... 100% [████████████████] 3 epochs
  
Run 3: Learning Rate = 5e-5
  Training... 100% [████████████████] 3 epochs
```

### During Comparison
```
MODEL COMPARISON RESULTS
┌─────────────────────────────────────────────────────────────────┐
│ Run Name      │ LR     │ F1    │ Accuracy │ Precision │ Recall  │
├───────────────┼────────┼───────┼──────────┼───────────┼─────────┤
│ lr-2e-5-run-1 │ 0.0002 │ 0.720 │ 0.7400   │ 0.7100    │ 0.7300  │
│ lr-5e-5-run-1 │ 0.0005 │ 0.705 │ 0.7200   │ 0.6900    │ 0.7200  │
│ lr-1e-5-run-1 │ 0.0001 │ 0.698 │ 0.7100   │ 0.6800    │ 0.7100  │
└─────────────────────────────────────────────────────────────────┘

BEST MODEL IDENTIFIED
  Run ID: <run-id>
  F1 Score: 0.720
  Learning Rate: 2e-5
  Status: Promoted to Production ✓
```

---

## Files Used

| File | Purpose |
|------|---------|
| `compare_models.py` | Core comparison logic |
| `start_mlflow_and_compare.py` | Start MLflow + run comparison (Python) |
| `compare.ps1` | One-command comparison (PowerShell for Windows) |

---

## MLflow UI Features

Access at: **http://localhost:5000**

### 1. Experiments Tab
View all training runs organized by experiment:
- See all runs with their parameters
- View metric values
- Search and filter runs

### 2. Runs Tab
Compare multiple runs side-by-side:
- Click on run names to compare
- View parameter values
- See metric charts
- Download artifacts

### 3. Models Tab
Manage model lifecycle:
- View registered models
- See version history
- Check current stage (Staging/Production)
- Download model artifacts

### 4. Charts
- Plot metrics over time
- Scatter plots of parameters vs. metrics
- Compare runs visually

---

## Command Options

### compare_models.py

```bash
python compare_models.py [OPTIONS]

Options:
  --learning-rates LR [LR ...]  Learning rates to compare
                                (default: 1e-5 2e-5 5e-5)
  --epochs N                    Number of epochs
                                (default: 3)
  --batch-size N                Batch size
                                (default: 16)
  --output FILE                 Output report file
                                (default: models/comparison_report.json)
```

**Examples:**
```bash
# Compare 4 learning rates
python compare_models.py --learning-rates 1e-5 2e-5 5e-5 1e-4

# Longer training
python compare_models.py --epochs 5 --batch-size 32

# Quick test
python compare_models.py --epochs 1 --batch-size 8

# Custom output
python compare_models.py --output results/my_comparison.json
```

### start_mlflow_and_compare.py

```bash
python start_mlflow_and_compare.py [OPTIONS]

Options:
  --learning-rates LR [LR ...]  Learning rates
                                (default: 1e-5 2e-5 5e-5)
  --epochs N                    Epochs per run
                                (default: 3)
  --batch-size N                Batch size
                                (default: 16)
  --host HOST                   MLflow server host
                                (default: 127.0.0.1)
  --port PORT                   MLflow server port
                                (default: 5000)
  --no-train                    Start MLflow only
  --no-browser                  Don't open browser
```

**Examples:**
```bash
# Default
python start_mlflow_and_compare.py

# Custom port
python start_mlflow_and_compare.py --port 8000

# Just MLflow (no training)
python start_mlflow_and_compare.py --no-train

# Different learning rates
python start_mlflow_and_compare.py --learning-rates 1e-5 3e-5 1e-4
```

### compare.ps1 (Windows PowerShell)

```powershell
.\compare.ps1 [OPTIONS]

Options:
  -LearningRates <rates>    Comma-separated learning rates
                            (default: 1e-5,2e-5,5e-5)
  -Epochs <n>               Number of epochs
                            (default: 3)
  -BatchSize <n>            Batch size
                            (default: 16)
  -Port <n>                 MLflow port
                            (default: 5000)
  -NoTrain                  Start MLflow only
  -NoBrowser                Don't open browser
  -Help                     Show help
```

**Examples:**
```powershell
# Default
.\compare.ps1

# Custom learning rates
.\compare.ps1 -LearningRates 1e-5,3e-5,1e-4

# Extended training
.\compare.ps1 -Epochs 10 -BatchSize 64

# Just start MLflow
.\compare.ps1 -NoTrain
```

---

## Common Scenarios

### Scenario 1: Quick Learning Rate Comparison
```bash
# Find best learning rate quickly
python start_mlflow_and_compare.py \
    --learning-rates 1e-5 2e-5 5e-5 \
    --epochs 2 \
    --batch-size 8
```

Time: ~10-15 minutes (GPU)

### Scenario 2: Detailed Hyperparameter Study
```bash
# Extended training for production
python start_mlflow_and_compare.py \
    --learning-rates 1e-5 2e-5 5e-5 \
    --epochs 10 \
    --batch-size 32
```

Time: ~1-2 hours (GPU)

### Scenario 3: Compare Different Models
Edit `config/config.yaml`:
```yaml
# Run 1
model:
  pretrained: "yangheng/deberta-v3-base-absa-v1.1"

# Run 2
model:
  pretrained: "xlm-roberta-base"
```

Then manually run training for each.

### Scenario 4: Low Memory GPU
```bash
python start_mlflow_and_compare.py \
    --learning-rates 1e-5 2e-5 \
    --epochs 3 \
    --batch-size 8
```

### Scenario 5: CPU Only Training
```bash
# Edit config/config.yaml
hardware:
  device: "cpu"
  num_workers: 2

# Then run
python start_mlflow_and_compare.py --epochs 1
```

---

## Understanding Results

### Key Metrics

**F1-Score** (Primary metric)
- Best overall metric for imbalanced classification
- Range: 0-1 (higher is better)
- Combines precision and recall

**Accuracy**
- Overall correctness
- Can be misleading with imbalanced data
- Range: 0-1

**Precision**
- True positives / all predicted positives
- "When model predicts positive, how often correct?"
- High precision = fewer false positives

**Recall**
- True positives / all actual positives
- "Of all actual positives, how many found?"
- High recall = fewer false negatives

### Example Interpretation
```
Model A: F1=0.72, Accuracy=0.74, Precision=0.71, Recall=0.73
Model B: F1=0.68, Accuracy=0.70, Precision=0.69, Recall=0.67

→ Model A is better
  - Higher F1 (primary metric)
  - Better balanced precision/recall
  - Good overall accuracy
```

---

## Output Files

After comparison, check:

1. **models/comparison_report.json**
   - Detailed metrics for all runs
   - Best model identified
   - Timestamp and configuration

2. **MLflow Database** (mlruns/)
   - All run artifacts
   - Models and checkpoints
   - Configuration snapshots

3. **Console Output**
   - Comparison table
   - Best model info
   - Promotion status

---

## Troubleshooting

### Issue: "Port 5000 already in use"
```bash
# Use different port
python start_mlflow_and_compare.py --port 8000
```

### Issue: "CUDA out of memory"
```bash
# Reduce batch size
python compare_models.py --batch-size 8 --epochs 2
```

### Issue: "Training takes too long"
```bash
# Reduce epochs
python start_mlflow_and_compare.py --epochs 1

# Or test with quick run
python compare_models.py --epochs 1 --batch-size 4
```

### Issue: "Browser won't open"
```bash
# Disable auto-browser
python start_mlflow_and_compare.py --no-browser

# Then manually open:
# http://127.0.0.1:5000
```

### Issue: "Comparison failed"
```bash
# Check errors
python compare_models.py --epochs 1 --batch-size 8

# View logs
cat logs/training_log.txt
```

---

## Best Practices

### ✅ DO:
1. **Start with quick test**
   ```bash
   python start_mlflow_and_compare.py --epochs 1
   ```

2. **Use wide range of learning rates**
   ```bash
   python start_mlflow_and_compare.py \
       --learning-rates 1e-5 2e-5 5e-5 1e-4
   ```

3. **Check MLflow UI regularly**
   - Monitor training progress
   - See metric trends
   - Catch issues early

4. **Save comparison reports**
   ```bash
   # Results saved automatically to:
   models/comparison_report.json
   ```

5. **Document best model**
   - Note run ID
   - Record parameters
   - Save metrics

### ❌ DON'T:
1. Don't ignore MLflow UI
2. Don't skip early validation runs
3. Don't use extremely large learning rates (>1e-3)
4. Don't compare runs on different data
5. Don't forget to promote best model to Production

---

## Next Steps

### After Comparison:
1. ✓ Review comparison report
2. ✓ Check MLflow metrics
3. ✓ Promote best model to Production
4. ✓ Deploy for inference
5. ✓ Monitor in production

### To Use Best Model:
```python
import mlflow.pyfunc

# Load from registry
model = mlflow.pyfunc.load_model("models:/ABSA-ASC-Deberta-Hindi/Production")

# Make predictions
predictions = model.predict(data)
```

### To Deploy API:
```bash
python src/api/main.py
```

---

## Learning Resources

- **MLflow Documentation**: https://mlflow.org/docs/latest/
- **PyABSA GitHub**: https://github.com/yangheng/PyABSA
- **Hyperparameter Tuning**: https://scikit-optimize.github.io/

---

## FAQ

**Q: How many runs should I compare?**
A: Start with 3 learning rates. If you have time, try 4-5.

**Q: How long does comparison take?**
A: 
- Quick test (1 epoch): 5-10 min
- Standard (3 epochs): 20-40 min
- Thorough (5 epochs): 1-2 hours

**Q: Can I cancel a run?**
A: Yes, just press Ctrl+C. MLflow will still track it as "FAILED".

**Q: Where are models saved?**
A: 
- Best model: `models/`
- All artifacts: `mlruns/`
- Registry: MLflow database

**Q: How do I export the best model?**
A:
```python
import mlflow
model_uri = "models:/ABSA-ASC-Deberta-Hindi/Production"
mlflow.artifacts.download_artifacts(model_uri)
```

**Q: Can I use different models in comparison?**
A: Yes, edit `config/config.yaml` and run training separately.

---

## Commands Quick Reference

```bash
# Windows PowerShell - Default comparison
.\compare.ps1

# Windows PowerShell - Custom rates
.\compare.ps1 -LearningRates 1e-5,3e-5,1e-4

# Python - Default
python start_mlflow_and_compare.py

# Python - MLflow only
python start_mlflow_and_compare.py --no-train

# Direct comparison
python compare_models.py --learning-rates 1e-5 2e-5 5e-5

# Manual MLflow only
mlflow ui

# View results
cat models/comparison_report.json
```

---

*Last Updated: April 5, 2026*
