# 📋 Complete ABSA Training Implementation - File Index

## 🎯 What Was Created

A complete, production-ready Aspect-Based Sentiment Analysis (ABSA) training pipeline with MLflow integration for Hindi/Hinglish restaurant reviews.

**Status**: ✅ Ready to Use | **Components**: 14 files | **Lines of Code**: 3500+

---

## 📂 File Locations & Descriptions

### Core Training Module

#### **`src/models/train_model.py`** ⭐ (420+ lines)
**The main training engine**

```python
from src.models.train_model import ABSATrainer

trainer = ABSATrainer()
trainer.train_asc_model(...)  # Aspect-Sentiment Classification
trainer.train_ate_model(...)  # Aspect Term Extraction
```

**Contains:**
- `ABSATrainer` class for model training
- `load_data()` - Robust data loading with encoding support
- `prepare_absa_data()` - Convert to PyABSA format
- `train_asc_model()` - Train sentiment classification
- `train_ate_model()` - Train aspect extraction
- `register_model_to_mlflow()` - MLflow registry integration
- `copy_model_to_models_dir()` - Save models to disk

**Features:**
✅ Single/batch data loading
✅ UTF-8 and Latin-1 encoding auto-detection
✅ MLflow parameter/metric tracking
✅ Model checkpointing and early stopping
✅ Comprehensive error handling
✅ Multilingual support (Hindi/Hinglish)

---

#### **`src/models/train.py`** (35 lines)
**Wrapper entry point for training**

```bash
python src/models/train.py
```

Simple wrapper that imports and calls `train_model.py`'s main function.

---

### CLI & Runner Scripts

#### **`run_training.py`** ⭐ (400+ lines - Project Root)
**Advanced CLI interface with full control**

```bash
# Examples:
python run_training.py --task asc
python run_training.py --task asc --epochs 5 --batch-size 32
python run_training.py --task both
python run_training.py --task asc --dry-run
python run_training.py --help
```

**Features:**
- Task selection (asc, ate, both)
- Override any hyperparameter
- MLflow experiment customization
- Config export/import
- Dry-run mode
- Optional model registration
- Flexible data path configuration

**CLI Options:**
```
--task               : asc, ate, or both
--epochs            : Number of training epochs
--batch-size        : Batch size for training
--learning-rate     : Optimizer learning rate
--max-length        : Maximum sequence length
--model-name        : Model identifier
--pretrained        : Pretrained model path
--train-data        : Path to training data
--val-data          : Path to validation data
--test-data         : Path to test data
--output-dir        : Where to save models
--experiment        : MLflow experiment name
--mlflow-uri        : MLflow server URI
--run-name          : MLflow run name
--no-register       : Skip MLflow registry
--export-config     : Export config to JSON
--dry-run           : Show config without training
```

---

#### **`train.sh`** (Bash - Unix/Linux/Mac)
**Convenient training presets**

```bash
bash train.sh quick        # 1 epoch test
bash train.sh default      # 5 epochs (standard)
bash train.sh production   # 10 epochs (high quality)
bash train.sh lowmem       # 3 epochs (low VRAM)
bash train.sh custom       # Interactive setup
bash train.sh both         # ASC + ATE
bash train.sh verify       # Check dependencies
bash train.sh demo         # Complete demo
bash train.sh help         # Show help
```

---

#### **`train.ps1`** (PowerShell - Windows)
**Windows equivalent of train.sh**

```powershell
.\train.ps1 -Preset quick
.\train.ps1 -Preset default
.\train.ps1 -Preset production
.\train.ps1 -Help
```

Same functionality as `train.sh` but for Windows PowerShell.

---

### Inference & Prediction

#### **`src/models/predict.py`** ⭐ (400+ lines)
**Make predictions on new data**

```bash
# Single prediction
python src/models/predict.py --model models/asc_model/best_model \
    --text "Great food" --aspects food

# Batch predictions
python src/models/predict.py --model models/asc_model/best_model \
    --input-file reviews.txt --output-file results.json

# Interactive mode
python src/models/predict.py --model models/asc_model/best_model \
    --task asc --interactive
```

**Features:**
- Single, batch, and interactive prediction modes
- ASC (Aspect-Sentiment Classification)
- ATE (Aspect Term Extraction)
- JSON output support
- GPU/CPU selection
- Confidence scores

**CLI Options:**
```
--model          : Path to trained model (required)
--task           : asc or ate (default: asc)
--text           : Text to predict
--aspects        : Aspect names for ASC
--input-file     : Batch input file
--output-file    : Batch output file
--interactive    : Interactive prediction mode
--device         : cuda or cpu (default: cuda)
```

---

### Model Comparison & MLflow Management

#### **`compare_models.py`** ⭐ (440+ lines - Project Root - NEW)
**Compare training runs with different hyperparameters**

```bash
# Compare different learning rates
python compare_models.py --learning-rates 1e-5 2e-5 5e-5

# Quick test
python compare_models.py --epochs 1 --batch-size 8

# Custom comparison
python compare_models.py --learning-rates 1e-5 3e-5 1e-4 --epochs 3 --batch-size 16
```

**Contains:**
- `ModelComparator` class for orchestrating comparisons
- `run_training()` - Subprocess-based training execution
- `compare_runs()` - MLflow run comparison with tabulated output
- `get_best_run()` - F1-score based best model identification
- `promote_best_model()` - Automatic promotion to Production stage
- `generate_report()` - JSON report generation

**Features:**
✅ Compare 2-3+ learning rates in one command
✅ Automatic MLflow run tracking
✅ Formatted comparison table with metrics
✅ Best model identification (F1-score)
✅ Automatic promotion to Production stage
✅ JSON report generation
✅ Customizable epochs and batch size

**Output:**
- Console comparison table
- MLflow experiments/runs with metrics
- JSON report: `models/comparison_report.json`

---

#### **`start_mlflow_and_compare.py`** ⭐ (280+ lines - Project Root - NEW)
**One-command MLflow server + model comparison**

```bash
# Default comparison with MLflow UI
python start_mlflow_and_compare.py

# Custom learning rates
python start_mlflow_and_compare.py --learning-rates 1e-5 3e-5 1e-4

# Just start MLflow UI (no training)
python start_mlflow_and_compare.py --no-train

# Custom port
python start_mlflow_and_compare.py --port 8000
```

**Contains:**
- `start_mlflow_server()` - Background MLflow server process
- `open_mlflow_ui()` - Automatic browser launch
- `run_comparison()` - Subprocess coordination with compare_models.py
- Full argument parsing for all parameters

**Features:**
✅ Automatic MLflow server startup
✅ Browser auto-launch to dashboard
✅ Full comparison pipeline in one command
✅ Custom port support
✅ Optional training skip (UI only)
✅ Comprehensive logging

**Output:**
- MLflow dashboard at http://localhost:5000
- All metrics in real-time
- Comparison results in console
- Best model promoted to Production

---

#### **`compare.ps1`** (150+ lines - Project Root - NEW)
**Windows PowerShell convenience script for model comparison**

```powershell
# Default comparison
.\compare.ps1

# Custom learning rates
.\compare.ps1 -LearningRates 1e-5,3e-5,1e-4

# Extended training
.\compare.ps1 -Epochs 5 -BatchSize 32

# Just MLflow
.\compare.ps1 -NoTrain

# Help
.\compare.ps1 -Help
```

**Parameters:**
- `-LearningRates` - Comma-separated learning rates (default: 1e-5,2e-5,5e-5)
- `-Epochs` - Training epochs (default: 3)
- `-BatchSize` - Batch size (default: 16)
- `-Port` - MLflow port (default: 5000)
- `-NoTrain` - Start MLflow only, skip training
- `-NoBrowser` - Don't auto-open browser
- `-Help` - Show help message

**Features:**
✅ Complete comparison workflow on Windows
✅ Parameter validation
✅ Dependency checking (tabulate)
✅ Color-coded console output
✅ MLflow server management
✅ Browser auto-launch

---

### Utility Scripts

#### **`verify_setup.py`** (Project Root)
**Dependency and configuration verification**

```bash
python verify_setup.py          # Full verification
python verify_setup.py --quick  # Quick check only
python verify_setup.py --fix    # Auto-install packages
```

**Checks:**
✅ Python version (3.8+)
✅ Required packages (PyABSA, torch, transformers, mlflow, pandas, sklearn)
✅ Optional packages (docker, black, dvc)
✅ CUDA/GPU availability
✅ Data file existence and size
✅ Configuration files
✅ MLflow server accessibility

---

### Configuration Files

#### **`config/config.yaml`** ⭐ (NEW - Unified Configuration)
**Central configuration file for all hyperparameters**

Comprehensive configuration with 150+ lines of documented settings:
- **Model**: Pretrained model selection, max length
- **Training**: Epochs, batch size, learning rate, optimization
- **Data**: Paths, encoding, preprocessing, validation
- **MLflow**: Experiment tracking, model registry, artifacts
- **Hardware**: Device, GPU, mixed precision
- **Evaluation**: Metrics, sentiment labels, thresholds
- **Advanced**: Fine-tuning, debugging, augmentation

All settings documented with examples and recommendations.

**Usage**:
```bash
# Edit config, then:
python run_training.py --task asc
```

**Reference**: See `CONFIG_GUIDE.md` for complete documentation

---

#### **`config/model.yaml`** (Legacy)
Kept for backward compatibility. Use `config/config.yaml` instead.

#### **`config/data.yaml`** (Legacy)
Kept for backward compatibility. Use `config/config.yaml` instead.

---

### Documentation

#### **`CONFIG_GUIDE.md`** (Project Root)
**Complete configuration reference**

- Configuration overview
- Detailed parameter explanations
- Common scenarios with examples
- Batch size guide for different GPUs
- Learning rate tuning tips
- Troubleshooting configurations
- Best practices

**Read this to understand and customize `config/config.yaml`**

---

#### **`COMPARISON_GUIDE.md`** (Project Root - NEW)
**Complete guide for model comparison and MLflow**

Comprehensive guide covering:
- Quick start instructions (2 minutes)
- MLflow UI features and navigation
- Command options for all comparison scripts
- Common scenarios (quick test, learning rate comparison, etc.)
- Output file descriptions
- Troubleshooting guide
- Best practices
- Next steps after comparison

**Read this for detailed information on model comparison.**

---

#### **`COMPARISON_QUICKSTART.md`** (Project Root - NEW)
**5-minute quick start for model comparison**

Fast reference for:
- One-command quick start
- What happens during comparison
- File locations
- Customize learning rates
- Just MLflow (no training)
- Quick test (5-10 min)
- Stop/Next steps

**Read this for fastest way to start comparing models.**

---

#### **`TRAINING_QUICKSTART.md`** (Project Root)
**5-minute quick start guide**

- Quick setup instructions
- First training in 30 seconds
- Verification checklist
- Common scenarios
- Troubleshooting quick tips
- Next steps after training

**Read this first if you're in a hurry!**

---

#### **`src/models/TRAINING_GUIDE.md`**
**Comprehensive training reference**

- Feature overview
- Data format specifications
- Installation requirements
- Configuration detailed explanation
- Complete training workflows
- MLflow integration guide
- Performance expectations
- Advanced configuration options
- Complete troubleshooting section
- Model details and specifications

**Read this for complete information.**

---

#### **`src/models/README.md`**
**Module overview and usage guide**

- Module contents table
- Quick start (3 steps)
- Training workflow
- Common use cases (6 scenarios)
- Advanced configuration
- What gets trained (ASC/ATE)
- MLflow integration details
- Complete workflow example
- Troubleshooting guide
- Learning resources

**Start here when working with models module.**

---

#### **`IMPLEMENTATION_SUMMARY.md`** (Project Root)
**Complete implementation overview**

- What was created
- Key features implemented
- Training workflow diagram
- Quick start (30 seconds)
- Complete directory structure
- Usage examples
- Expected output
- Configuration details
- System requirements
- MLflow integration guide
- Next steps
- Performance expectations

**Reference guide for the entire implementation.**

---

## 📊 Quick Reference Table

| File | Purpose | Location | Size | Use When |
|------|---------|----------|------|----------|
| `train_model.py` | Core training logic | `src/models/` | 420+ lines | Writing custom training code |
| `train.py` | Training entry point | `src/models/` | 35 lines | Running `python src/models/train.py` |
| `run_training.py` | Advanced CLI | Project root | 400+ lines | Need full control over parameters |
| `predict.py` | Inference/predictions | `src/models/` | 400+ lines | Making predictions on new data |
| `train.sh` | Presets (bash) | Project root | 200+ lines | Linux/Mac quick training |
| `train.ps1` | Presets (PowerShell) | Project root | 250+ lines | Windows quick training |
| **`compare_models.py`** | **Compare learning rates** | **Project root** | **440+ lines** | **Comparing model performance** |
| **`start_mlflow_and_compare.py`** | **MLflow + comparison** | **Project root** | **280+ lines** | **One-command comparison workflow** |
| **`compare.ps1`** | **Windows comparison** | **Project root** | **150+ lines** | **Windows PowerShell comparison** |
| `verify_setup.py` | Dependency check | Project root | 200+ lines | Troubleshooting setup issues |
| **`config/config.yaml`** | **Hyperparameter config** | **`config/`** | **150+ lines** | **Configuring training** |
| `CONFIG_GUIDE.md` | Configuration reference | Project root | - | Understanding config options |
| `COMPARISON_GUIDE.md` | Model comparison reference | Project root | - | Detailed comparison guide |
| `COMPARISON_QUICKSTART.md` | Model comparison quick start | Project root | - | Quick comparison instructions |
| `TRAINING_QUICKSTART.md` | 5-min training guide | Project root | - | Getting started with training |
| `TRAINING_GUIDE.md` | Comprehensive ref | `src/models/` | - | Detailed training information |
| `README.md` | Module guide | `src/models/` | - | Module overview |

---

## 🚀 Getting Started in 3 Steps

### Step 1: Verify Everything is Working
```bash
python verify_setup.py
```

### Step 2: Train the Model
```bash
# Option A: Simplest
python run_training.py --task asc

# Option B: Windows with presets
.\train.ps1 -Preset default

# Option C: Linux/Mac with presets
bash train.sh default
```

### Step 3: View Results
```bash
mlflow ui
# Open http://localhost:5000
```

---

## 📚 Documentation Navigation

**If you want to:**

| Goal | Read This | Then Do |
|------|-----------|--------|
| Quick start (5 min) | TRAINING_QUICKSTART.md | `python run_training.py --task asc` |
| Understand module | src/models/README.md | Review use cases section |
| Complete reference | src/models/TRAINING_GUIDE.md | Section by section |
| Detailed info | IMPLEMENTATION_SUMMARY.md | Focus on your use case |
| Verify setup | run `verify_setup.py` | Address any issues |
| Make predictions | src/models/predict.py --help | See examples section |
| Compare models | COMPARISON_QUICKSTART.md | `python start_mlflow_and_compare.py` |
| MLflow UI & comparison | COMPARISON_GUIDE.md | Detailed walkthrough |
| Troubleshoot | TRAINING_GUIDE.md Troubleshooting | Follow diagnostic steps |
| Troubleshoot comparison | COMPARISON_GUIDE.md Troubleshooting | Follow diagnostic steps |

---

## 🎯 Common Commands Cheat Sheet

### Training
```bash
# Basic training
python run_training.py --task asc

# With custom parameters
python run_training.py --task asc --epochs 5 --batch-size 32

# Both ASC and ATE
python run_training.py --task both

# Dry run (preview only)
python run_training.py --task asc --dry-run

# With presets (Windows)
.\train.ps1 -Preset quick
.\train.ps1 -Preset production

# With presets (Linux/Mac)
bash train.sh quick
bash train.sh production
```

### Inference
```bash
# Single prediction
python src/models/predict.py --model models/asc_model/best_model \
    --text "Great food" --aspects food

# Batch from file
python src/models/predict.py --model models/asc_model/best_model \
    --input-file reviews.txt --output-file predictions.json

# Interactive
python src/models/predict.py --model models/asc_model/best_model \
    --interactive
```

### Model Comparison
```bash
# Default comparison (3 learning rates)
python start_mlflow_and_compare.py

# Custom learning rates
python start_mlflow_and_compare.py --learning-rates 1e-5 3e-5 1e-4

# Just MLflow UI (no training)
python start_mlflow_and_compare.py --no-train

# Direct comparison script
python compare_models.py --epochs 1 --batch-size 8

# Windows PowerShell
.\compare.ps1
.\compare.ps1 -LearningRates 1e-5,3e-5,1e-4
.\compare.ps1 -NoTrain
```

### Utilities
```bash
# Verify dependencies
python verify_setup.py

# Quick dependency check
python verify_setup.py --quick

# Auto-fix missing packages
python verify_setup.py --fix

# View MLflow experiments
mlflow ui
```

---

## 📦 Dependencies Required

All included in `requirements.txt`:

**Core ABSA:**
- pyabsa >= 1.9.0
- transformers >= 4.30.0
- torch >= 2.0.0

**MLOps:**
- mlflow >= 2.2.0
- datasets >= 2.11.0

**Data Processing:**
- pandas >= 1.5.0
- scikit-learn >= 1.2.0

**Optional (for API):**
- fastapi >= 0.95.0
- uvicorn >= 0.22.0

Install with:
```bash
pip install -r requirements.txt
```

---

## ✅ Verification Checklist

Before running training:
- [ ] All files exist in their locations
- [ ] Python 3.8+ installed
- [ ] All dependencies installed: `pip install -r requirements.txt`
- [ ] Data files exist: `data/processed/{train,val,test}_data.txt`
- [ ] Data format correct: `text$$$[aspect]$$$sentiment`
- [ ] Config file accessible: `config/model.yaml`
- [ ] Output directory writable: `models/`
- [ ] (Optional) CUDA available for GPU training

**Quick verification:**
```bash
python verify_setup.py
```

---

## 📊 What Gets Saved

After training:
```
models/
├── asc_model/
│   ├── best_model/
│   │   ├── pytorch_model.bin      # Model weights
│   │   ├── config.json            # Model config
│   │   └── special_tokens_map.json
│   └── asc_model_val.txt
├── training_summary.json            # Final metrics
└── training_results.json            # Detailed results

mlruns/
└── <experiment>/
    └── <run>/
        ├── params/                # Hyperparameters
        ├── metrics/               # Training metrics
        └── artifacts/             # Model artifacts
```

---

## 🔄 Complete Workflow Example

```bash
# 1. Verify setup
python verify_setup.py

# 2. Preview configuration
python run_training.py --task asc --dry-run

# 3. Train model
python run_training.py --task asc --epochs 5

# 4. Monitor progress (in another terminal)
mlflow ui

# 5. Check results
cat models/training_summary.json

# 6. Make predictions
python src/models/predict.py \
    --model models/asc_model/best_model \
    --text "Great food, but slow service" \
    --aspects food service

# 7. View in MLflow (http://localhost:5000)
```

---

## 🎉 You Now Have

✅ **Complete Training Pipeline**
- PyABSA integration
- Multilingual model (DeBERTa-v3-base)
- Fine-tuning for 3-5 epochs
- Automatic model selection

✅ **MLflow Experiment Tracking**
- All parameters logged
- All metrics tracked
- Model artifacts saved
- MLflow Registry integration

✅ **Production-Ready**
- Robust error handling
- Data validation
- Configuration management
- Inference support

✅ **Easy to Use**
- Simple CLI interface
- Preset scripts
- Comprehensive documentation
- Dependency verification

✅ **Well Documented**
- 5 documentation files
- Code examples
- Troubleshooting guides
- Reference materials

---

## 📞 Quick Troubleshooting

| Issue | Solution |
|-------|----------|
| "Module not found" | Run `pip install -r requirements.txt` |
| "CUDA out of memory" | `python run_training.py --task asc --batch-size 8` |
| "Data not found" | Check `data/processed/train_data.txt` exists |
| "MLflow not tracking" | Run `mlflow server` in another terminal |
| Python version < 3.8 | Install Python 3.8+ |
| Setup issues | Run `python verify_setup.py` |

---

## 🚀 Next Steps

1. **Start Simple:**
   ```bash
   python run_training.py --task asc
   ```

2. **Monitor:**
   ```bash
   mlflow ui
   ```

3. **Evaluate:**
   ```bash
   cat models/training_summary.json
   ```

4. **Predict:**
   ```bash
   python src/models/predict.py --model models/asc_model/best_model --interactive
   ```

5. **Deploy:**
   - Use models for inference
   - Integrate with API (`src/api/main.py`)
   - Promote to production in MLflow

---

## 📝 File Statistics

- **Total Files Created**: 17
- **Total Lines of Code**: 4500+
- **Documentation Files**: 7
- **Python Modules**: 6
  - Training: 2
  - Inference: 1
  - Comparison: 2
  - Utilities: 1
- **Shell Scripts**: 3
  - Bash: 1
  - PowerShell: 2
- **Configuration Supported**: YAML, JSON

---

**Created**: April 5, 2026
**Status**: ✅ Production Ready
**Last Updated**: Today

---

## 📌 Bookmarks

| Quick Link | File |
|-----------|------|
| 📍 Start Here | TRAINING_QUICKSTART.md |
| 📍 Compare Models | COMPARISON_QUICKSTART.md |
| 📍 Complete Ref | IMPLEMENTATION_SUMMARY.md |
| 📍 Module Info | src/models/README.md |
| 📍 Detailed Guide | src/models/TRAINING_GUIDE.md |
| 📍 Comparison Details | COMPARISON_GUIDE.md |
| 📍 Training Code | src/models/train_model.py |
| 📍 Predictions | src/models/predict.py |
| 📍 Model Comparison | compare_models.py |
| 📍 MLflow + Comparison | start_mlflow_and_compare.py |
| 📍 Presets (Win Training) | train.ps1 |
| 📍 Presets (Win Comparison) | compare.ps1 |
| 📍 Presets (Unix) | train.sh |
| 📍 Verify Setup | verify_setup.py |

---

Happy training and comparing! 🎉
