# Model Comparison - Quick Start

**TL;DR: Run 2-3 commands to compare models and view results in MLflow UI**

## One-Command Start

### Windows (PowerShell)
```powershell
cd d:\Mtech\NLP\Project\CustomerSentimentAnalysis
.\compare.ps1
```

### Mac/Linux
```bash
cd d:/Mtech/NLP/Project/CustomerSentimentAnalysis
python start_mlflow_and_compare.py
```

---

## What Happens In 20-40 Minutes

1. ✅ **MLflow Server** starts
2. ✅ **Browser opens** to http://localhost:5000
3. ✅ **Training runs** for 3 different learning rates:
   - 1e-5 (conservative)
   - 2e-5 (moderate)
   - 5e-5 (aggressive)
4. ✅ **Comparison table** shows which learning rate is best
5. ✅ **Best model** automatically promoted to Production

---

## Check Results

### In Browser (Live Dashboard)
Go to: **http://localhost:5000**

- **Experiments** → See all training runs
- **Runs** → Compare runs side-by-side
- **Models** → Check registered models
- **Charts** → Plot metrics

### In Terminal
```bash
# View detailed report
cat models/comparison_report.json
```

---

## Customize Learning Rates

### PowerShell
```powershell
.\compare.ps1 -LearningRates 1e-5,3e-5,1e-4
```

### Python
```bash
python start_mlflow_and_compare.py --learning-rates 1e-5 3e-5 1e-4
```

---

## Just Start MLflow (No Training)

### PowerShell
```powershell
.\compare.ps1 -NoTrain
```

### Python
```bash
python start_mlflow_and_compare.py --no-train
```

Then access: **http://localhost:5000**

---

## Quick Test (5-10 min)

### PowerShell
```powershell
.\compare.ps1 -Epochs 1 -BatchSize 8
```

### Python
```bash
python start_mlflow_and_compare.py --epochs 1 --batch-size 8
```

---

## Stop Everything

Press **Ctrl+C** in the terminal

MLflow data is saved automatically.

---

## Next: Deploy Best Model

Once you confirm best model in MLflow UI:

```bash
# Start API server
python src/api/main.py

# Then in another terminal, test it:
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Restaurant was amazing"}'
```

---

## Need Help?

See detailed guide:
```bash
cat COMPARISON_GUIDE.md
```

Or check MLflow documentation:
```bash
mlflow --version
mlflow ui --help
```

---

**That's it! Run one command and watch MLflow work.**
