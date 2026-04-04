# ABSA Training Helper Script (PowerShell for Windows)
# 
# Usage:
#   .\train.ps1 -Preset quick          # Quick test (1 epoch)
#   .\train.ps1 -Preset default        # Default settings
#   .\train.ps1 -Preset production     # Production quality
#   .\train.ps1 -Preset lowmem         # Low memory usage
#   .\train.ps1 -Help                  # Show help

param(
    [Parameter(Position = 0)]
    [ValidateSet('quick', 'default', 'production', 'lowmem', 'custom', 'both', 'verify', 'demo', 'help')]
    [string]$Preset = 'help',
    
    [switch]$Help
)

function Show-Help {
    Write-Host "ABSA Training Helper (PowerShell)" -ForegroundColor Blue
    Write-Host ""
    Write-Host "Convenient presets for training ABSA models with different configurations." -ForegroundColor Gray
    Write-Host ""
    Write-Host "Usage:" -ForegroundColor Green
    Write-Host "    .\train.ps1 -Preset [PRESET]" -ForegroundColor Cyan
    Write-Host "    .\train.ps1 quick           # Quick test"
    Write-Host "    .\train.ps1 default         # Standard training"
    Write-Host ""
    Write-Host "Available Presets:" -ForegroundColor Green
    Write-Host ""
    Write-Host "  quick           - Quick test run (1 epoch)" -ForegroundColor Cyan
    Write-Host "                    Batch: 8 | Learning Rate: 2e-5"
    Write-Host "                    Use for: Testing pipeline"
    Write-Host ""
    Write-Host "  default         - Standard training (5 epochs)" -ForegroundColor Cyan
    Write-Host "                    Batch: 32 | Learning Rate: 2e-5"
    Write-Host "                    Use for: Normal training"
    Write-Host ""
    Write-Host "  production      - High quality training (10 epochs)" -ForegroundColor Cyan
    Write-Host "                    Batch: 64 | Learning Rate: 1e-5"
    Write-Host "                    Use for: Production deployment"
    Write-Host ""
    Write-Host "  lowmem          - Low memory usage (3 epochs)" -ForegroundColor Cyan
    Write-Host "                    Batch: 8 | Learning Rate: 2e-5"
    Write-Host "                    Use for: Low VRAM/CPU training"
    Write-Host ""
    Write-Host "  custom          - Custom parameters (interactive)" -ForegroundColor Cyan
    Write-Host "                    Use for: Fine-grained control"
    Write-Host ""
    Write-Host "  both            - Train both ASC and ATE (5 epochs)" -ForegroundColor Cyan
    Write-Host "                    Batch: 32"
    Write-Host "                    Use for: Complete ABSA pipeline"
    Write-Host ""
    Write-Host "  verify          - Verify setup and dependencies" -ForegroundColor Cyan
    Write-Host "                    Use for: Troubleshooting"
    Write-Host ""
    Write-Host "  demo            - Run complete demo (verify + dry-run + quick)" -ForegroundColor Cyan
    Write-Host "                    Use for: Getting started"
    Write-Host ""
    Write-Host "Examples:" -ForegroundColor Green
    Write-Host "  .\train.ps1 quick" -ForegroundColor Cyan
    Write-Host "  .\train.ps1 production" -ForegroundColor Cyan
    Write-Host "  .\train.ps1 demo" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Tips:" -ForegroundColor Green
    Write-Host "  • Start with 'quick' to test your setup" -ForegroundColor Gray
    Write-Host "  • Use 'default' for normal training" -ForegroundColor Gray
    Write-Host "  • Use 'production' for best accuracy" -ForegroundColor Gray
    Write-Host "  • Use 'lowmem' if you get GPU memory errors" -ForegroundColor Gray
    Write-Host "  • Monitor with 'mlflow ui' in another terminal" -ForegroundColor Gray
    Write-Host ""
}

function Quick-Train {
    Write-Host "Running QUICK TRAIN (1 epoch test)" -ForegroundColor Blue
    Write-Host ""
    & python run_training.py `
        --task asc `
        --epochs 1 `
        --batch-size 8 `
        --learning-rate 2e-5
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host ""
        Write-Host "✓ Quick training completed!" -ForegroundColor Green
    }
}

function Default-Train {
    Write-Host "Running DEFAULT TRAIN (5 epochs)" -ForegroundColor Blue
    Write-Host ""
    & python run_training.py `
        --task asc `
        --epochs 5 `
        --batch-size 32 `
        --learning-rate 2e-5
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host ""
        Write-Host "✓ Default training completed!" -ForegroundColor Green
    }
}

function Production-Train {
    Write-Host "Running PRODUCTION TRAIN (10 epochs, high quality)" -ForegroundColor Blue
    Write-Host "Note: This may take 30+ minutes" -ForegroundColor Yellow
    Write-Host ""
    & python run_training.py `
        --task asc `
        --epochs 10 `
        --batch-size 64 `
        --learning-rate 1e-5
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host ""
        Write-Host "✓ Production training completed!" -ForegroundColor Green
    }
}

function LowMem-Train {
    Write-Host "Running LOW MEMORY TRAIN (3 epochs, batch 8)" -ForegroundColor Blue
    Write-Host ""
    & python run_training.py `
        --task asc `
        --epochs 3 `
        --batch-size 8 `
        --learning-rate 2e-5
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host ""
        Write-Host "✓ Low memory training completed!" -ForegroundColor Green
    }
}

function Custom-Train {
    Write-Host "Custom Training Configuration" -ForegroundColor Blue
    Write-Host ""
    
    $task = Read-Host "Task (asc/ate/both) [asc]"
    if ([string]::IsNullOrWhiteSpace($task)) { $task = "asc" }
    
    $epochs = Read-Host "Epochs [5]"
    if ([string]::IsNullOrWhiteSpace($epochs)) { $epochs = "5" }
    
    $batchSize = Read-Host "Batch size [32]"
    if ([string]::IsNullOrWhiteSpace($batchSize)) { $batchSize = "32" }
    
    $learningRate = Read-Host "Learning rate [2e-5]"
    if ([string]::IsNullOrWhiteSpace($learningRate)) { $learningRate = "2e-5" }
    
    Write-Host ""
    Write-Host "Running with:" -ForegroundColor Blue
    Write-Host "  Task: $task"
    Write-Host "  Epochs: $epochs"
    Write-Host "  Batch Size: $batchSize"
    Write-Host "  Learning Rate: $learningRate"
    Write-Host ""
    
    & python run_training.py `
        --task $task `
        --epochs $epochs `
        --batch-size $batchSize `
        --learning-rate $learningRate
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host ""
        Write-Host "✓ Custom training completed!" -ForegroundColor Green
    }
}

function Both-Train {
    Write-Host "Running BOTH ASC and ATE (5 epochs each)" -ForegroundColor Blue
    Write-Host ""
    & python run_training.py `
        --task both `
        --epochs 5 `
        --batch-size 32 `
        --learning-rate 2e-5
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host ""
        Write-Host "✓ Both models trained successfully!" -ForegroundColor Green
    }
}

function Verify-Setup {
    Write-Host "Verifying Setup and Dependencies" -ForegroundColor Blue
    Write-Host ""
    & python verify_setup.py
}

function Run-Demo {
    Write-Host "Running Complete Demo" -ForegroundColor Blue
    Write-Host ""
    
    Write-Host "Step 1: Verifying setup..." -ForegroundColor Yellow
    & python verify_setup.py
    
    Write-Host ""
    Write-Host "Step 2: Previewing configuration (dry-run)..." -ForegroundColor Yellow
    & python run_training.py --task asc --dry-run
    
    Write-Host ""
    Write-Host "Step 3: Running quick training (1 epoch)..." -ForegroundColor Yellow
    & python run_training.py `
        --task asc `
        --epochs 1 `
        --batch-size 8
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host ""
        Write-Host "✓ Demo completed!" -ForegroundColor Green
        Write-Host ""
        Write-Host "Next steps:" -ForegroundColor Blue
        Write-Host "  1. View results: cat models/training_summary.json" -ForegroundColor Cyan
        Write-Host "  2. Monitor: mlflow ui (http://localhost:5000)" -ForegroundColor Cyan
        Write-Host "  3. Predict: python src/models/predict.py --model models/asc_model/best_model --interactive" -ForegroundColor Cyan
    }
}

# Main execution
if ($Help -or $Preset -eq 'help') {
    Show-Help
    exit 0
}

# Check if Python is available
if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Host "Error: Python not found. Please install Python 3.8+" -ForegroundColor Red
    exit 1
}

# Execute based on preset
switch ($Preset) {
    'quick' { Quick-Train }
    'default' { Default-Train }
    'production' { Production-Train }
    'lowmem' { LowMem-Train }
    'custom' { Custom-Train }
    'both' { Both-Train }
    'verify' { Verify-Setup }
    'demo' { Run-Demo }
    default {
        Write-Host "Unknown preset: $Preset" -ForegroundColor Yellow
        Write-Host ""
        Show-Help
        exit 1
    }
}

exit $LASTEXITCODE
