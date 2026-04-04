# MLflow Comparison Script (PowerShell for Windows)
# 
# Start MLflow UI and run model comparison with different learning rates
# 
# Usage:
#   .\compare.ps1                                    # Default comparison
#   .\compare.ps1 -LearningRates 1e-5,2e-5,5e-5    # Custom learning rates
#   .\compare.ps1 -Epochs 5 -BatchSize 32           # Custom training params
#   .\compare.ps1 -NoTrain                          # Only start MLflow UI

param(
    [Parameter(ValueFromRemainingArguments=$true)]
    [string[]]$LearningRates = @('1e-5', '2e-5', '5e-5'),
    
    [int]$Epochs = 3,
    
    [int]$BatchSize = 16,
    
    [int]$Port = 5000,
    
    [switch]$NoTrain,
    
    [switch]$NoBrowser,
    
    [switch]$Help
)

if ($Help) {
    Write-Host "MLflow Model Comparison Script (PowerShell)" -ForegroundColor Blue
    Write-Host ""
    Write-Host "Usage:" -ForegroundColor Green
    Write-Host "  .\compare.ps1 [OPTIONS]"
    Write-Host ""
    Write-Host "Options:" -ForegroundColor Green
    Write-Host "  -LearningRates <rates>    Learning rates to compare (comma-separated)"
    Write-Host "  -Epochs <n>               Number of epochs (default: 3)"
    Write-Host "  -BatchSize <n>            Batch size (default: 16)"
    Write-Host "  -Port <n>                 MLflow port (default: 5000)"
    Write-Host "  -NoTrain                  Only start MLflow, skip training"
    Write-Host "  -NoBrowser                Don't open browser automatically"
    Write-Host "  -Help                     Show this help message"
    Write-Host ""
    Write-Host "Examples:" -ForegroundColor Green
    Write-Host "  .\compare.ps1                                    # Default"
    Write-Host "  .\compare.ps1 -Epochs 5                          # 5 epochs"
    Write-Host "  .\compare.ps1 -LearningRates 1e-5,3e-5,1e-4     # Custom rates"
    Write-Host "  .\compare.ps1 -NoTrain                           # MLflow only"
    Write-Host ""
    exit 0
}

# Convert string learning rates to array if needed
if ($LearningRates.Count -eq 1 -and $LearningRates[0] -like "*,*") {
    $LearningRates = $LearningRates[0] -split ','
}

Write-Host "=" * 80 -ForegroundColor Cyan
Write-Host "ABSA Model Comparison with MLflow" -ForegroundColor Cyan
Write-Host "=" * 80
Write-Host ""
Write-Host "Configuration:" -ForegroundColor Green
Write-Host "  Learning Rates: $($LearningRates -join ', ')"
Write-Host "  Epochs: $Epochs"
Write-Host "  Batch Size: $BatchSize"
Write-Host "  MLflow Port: $Port"
Write-Host ""

# Install tabulate if needed
Write-Host "Checking dependencies..." -ForegroundColor Yellow
try {
    python -c "import tabulate" 2>$null
    Write-Host "✓ All dependencies found" -ForegroundColor Green
} catch {
    Write-Host "Installing tabulate..." -ForegroundColor Yellow
    pip install tabulate | Out-Null
    Write-Host "✓ Dependencies installed" -ForegroundColor Green
}

# Start MLflow server
Write-Host ""
Write-Host "Starting MLflow server..." -ForegroundColor Cyan
$mlflowPath = "mlruns"
if (-not (Test-Path $mlflowPath)) {
    New-Item -ItemType Directory -Path $mlflowPath | Out-Null
}

# Start in background
$mlflowProcess = Start-Process -FilePath python -ArgumentList @(
    "-m", "mlflow", "server",
    "--host", "127.0.0.1",
    "--port", $Port,
    "--backend-store-uri", "sqlite:///mlflow.db",
    "--default-artifact-root", "mlruns"
) -PassThru -NoNewWindow

Write-Host "✓ MLflow server started (PID: $($mlflowProcess.Id))" -ForegroundColor Green

# Give server time to start
Start-Sleep -Seconds 3

# Open browser if requested
if (-not $NoBrowser) {
    Write-Host "Opening MLflow UI in browser..." -ForegroundColor Cyan
    $mlflowUrl = "http://127.0.0.1:$Port"
    Start-Process $mlflowUrl
    Write-Host "✓ MLflow UI opened: $mlflowUrl" -ForegroundColor Green
}

# Run comparison if requested
if (-not $NoTrain) {
    Write-Host ""
    Write-Host "=" * 80 -ForegroundColor Cyan
    Write-Host "Starting Model Comparison" -ForegroundColor Cyan
    Write-Host "=" * 80
    Write-Host ""
    
    # Build command
    $compareCmd = @(
        "compare_models.py",
        "--epochs", $Epochs,
        "--batch-size", $BatchSize,
        "--learning-rates"
    ) + $LearningRates
    
    # Run comparison
    & python @compareCmd
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host ""
        Write-Host "✓ Comparison completed successfully!" -ForegroundColor Green
    } else {
        Write-Host ""
        Write-Host "✗ Comparison failed!" -ForegroundColor Red
        Stop-Process -Id $mlflowProcess.Id
        exit 1
    }
}

# Show final status
Write-Host ""
Write-Host "=" * 80 -ForegroundColor Cyan
Write-Host "MLflow Dashboard Ready" -ForegroundColor Cyan
Write-Host "=" * 80
Write-Host ""
Write-Host "URL: http://127.0.0.1:$Port" -ForegroundColor Green
Write-Host ""
Write-Host "Available Sections:" -ForegroundColor Yellow
Write-Host "  1. Experiments - View all training runs"
Write-Host "  2. Models - Check registered models"
Write-Host "  3. Runs - Compare metrics and parameters"
Write-Host "  4. Artifacts - Download trained models"
Write-Host ""
Write-Host "To stop server: Press Ctrl+C"
Write-Host "=" * 80
Write-Host ""

# Keep server running
$mlflowProcess.WaitForExit()
