#!/usr/bin/env python
"""
Dependency Verification Script for ABSA Training Pipeline.

Checks that all required packages are installed and working correctly.
Provides diagnostic information for troubleshooting.

Usage:
    python verify_setup.py              # Full verification
    python verify_setup.py --quick      # Quick check only
    python verify_setup.py --fix        # Try to auto-fix issues
"""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

try:
    import importlib
except ImportError:
    print("❌ CRITICAL: importlib not available")
    sys.exit(1)


class DependencyChecker:
    """Check and verify project dependencies."""

    # Core packages required for training
    REQUIRED_PACKAGES = {
        # ABSA & NLP
        "pyabsa": "PyABSA for ABSA training",
        "transformers": "HuggingFace transformers",
        "datasets": "HuggingFace datasets",
        "torch": "PyTorch deep learning",

        # Data & ML
        "pandas": "Data processing",
        "sklearn": "Scikit-learn utilities",

        # MLOps & Tracking
        "mlflow": "MLflow experiment tracking",

        # API
        "fastapi": "FastAPI web framework (optional)",
        "uvicorn": "ASGI server (optional)",

        # Testing
        "pytest": "Testing framework (optional)",
    }

    # Optional packages
    OPTIONAL_PACKAGES = [
        "docker",
        "black",
        "dvc",
    ]

    def __init__(self):
        """Initialize checker."""
        self.issues = []
        self.warnings = []
        self.info = []

    def check_python_version(self) -> bool:
        """Check Python version is 3.8+."""
        version = sys.version_info
        if version.major == 3 and version.minor >= 8:
            self.info.append(f"✓ Python {version.major}.{version.minor}.{version.micro}")
            return True
        else:
            self.issues.append(
                f"Python 3.8+ required (found {version.major}.{version.minor})"
            )
            return False

    def check_package(self, package_name: str, import_name: str = None) -> bool:
        """
        Check if a package is installed.

        Args:
            package_name: Name as shown in pip
            import_name: Name to import (if different from package_name)

        Returns:
            True if installed, False otherwise
        """
        import_name = import_name or package_name
        try:
            mod = importlib.import_module(import_name)
            version = getattr(mod, "__version__", "unknown")
            return True, version
        except ImportError:
            return False, None

    def check_required_packages(self) -> Dict[str, Tuple[bool, str]]:
        """Check all required packages."""
        results = {}

        for package, description in self.REQUIRED_PACKAGES.items():
            # Map import names for packages with different names
            import_map = {
                "sklearn": "sklearn",
                "pyabsa": "pyabsa",
            }
            import_name = import_map.get(package, package)

            installed, version = self.check_package(package, import_name)
            results[package] = (installed, version, description)

            if installed:
                self.info.append(f"✓ {package} ({version})")
            else:
                self.issues.append(f"Missing: {package} - {description}")

        return results

    def check_optional_packages(self) -> Dict[str, Tuple[bool, str]]:
        """Check optional packages."""
        results = {}

        for package in self.OPTIONAL_PACKAGES:
            installed, version = self.check_package(package)
            results[package] = (installed, version)

            if installed:
                self.info.append(f"✓ {package} ({version})")
            else:
                self.warnings.append(f"Optional: {package} not installed")

        return results

    def check_torch_cuda(self) -> bool:
        """Check if PyTorch has CUDA support."""
        try:
            import torch

            if torch.cuda.is_available():
                self.info.append(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
                self.info.append(f"  CUDA Version: {torch.version.cuda}")
                return True
            else:
                self.warnings.append("CUDA not available (will use CPU - slower)")
                return False
        except Exception as e:
            self.warnings.append(f"Could not check CUDA: {e}")
            return False

    def check_data_files(self) -> bool:
        """Check if training data files exist."""
        data_files = [
            Path("data/processed/train_data.txt"),
            Path("data/processed/val_data.txt"),
            Path("data/processed/test_data.txt"),
        ]

        all_exist = True
        for file in data_files:
            if file.exists():
                size_mb = file.stat().st_size / (1024 * 1024)
                self.info.append(f"✓ {file} ({size_mb:.2f} MB)")
            else:
                self.issues.append(f"Missing: {file}")
                all_exist = False

        return all_exist

    def check_config_files(self) -> bool:
        """Check if configuration files exist."""
        config_files = [
            Path("config/model.yaml"),
            Path("config/data.yaml"),
        ]

        all_exist = True
        for file in config_files:
            if file.exists():
                self.info.append(f"✓ {file}")
            else:
                self.warnings.append(f"Missing: {file} (will use defaults)")
                all_exist = False

        return all_exist

    def check_mlflow_server(self) -> bool:
        """Check if MLflow server is accessible."""
        try:
            import mlflow

            # Try to get default tracking URI
            uri = mlflow.get_tracking_uri()
            self.info.append(f"✓ MLflow tracking URI: {uri}")
            return True
        except Exception as e:
            self.warnings.append(f"MLflow issue: {e}")
            return False

    def run_full_check(self) -> bool:
        """Run complete verification."""
        print("\n" + "=" * 60)
        print("ABSA Training Pipeline - Dependency Verification")
        print("=" * 60 + "\n")

        print("🔍 Checking Python Version...")
        python_ok = self.check_python_version()

        print("\n🔍 Checking Required Packages...")
        required = self.check_required_packages()

        print("\n🔍 Checking Optional Packages...")
        self.check_optional_packages()

        print("\n🔍 Checking CUDA/GPU Support...")
        self.check_torch_cuda()

        print("\n🔍 Checking Data Files...")
        data_ok = self.check_data_files()

        print("\n🔍 Checking Configuration Files...")
        self.check_config_files()

        print("\n🔍 Checking MLflow...")
        mlflow_ok = self.check_mlflow_server()

        return self.report()

    def run_quick_check(self) -> bool:
        """Quick check of critical components only."""
        print("\n" + "=" * 60)
        print("Quick Verification (Critical Components Only)")
        print("=" * 60 + "\n")

        print("🔍 Python Version...")
        python_ok = self.check_python_version()

        print("\n🔍 Critical Packages...")
        critical = {
            "pyabsa": "PyABSA for ABSA training",
            "torch": "PyTorch",
            "transformers": "HuggingFace transformers",
            "mlflow": "MLflow",
        }

        for package in critical:
            installed, version = self.check_package(package)
            if installed:
                self.info.append(f"✓ {package} ({version})")
            else:
                self.issues.append(f"Missing: {package}")

        return self.report()

    def report(self) -> bool:
        """Print report and return success status."""
        success = len(self.issues) == 0

        print("\n" + "=" * 60)
        print("REPORT")
        print("=" * 60 + "\n")

        if self.info:
            print("✅ WORKING:")
            for item in self.info:
                print(f"  {item}")

        if self.warnings:
            print("\n⚠️  WARNINGS:")
            for item in self.warnings:
                print(f"  {item}")

        if self.issues:
            print("\n❌ ISSUES (MUST FIX):")
            for item in self.issues:
                print(f"  {item}")

        print("\n" + "=" * 60)

        if success:
            print("✅ All checks passed! Ready to train.")
        else:
            print("❌ Some issues found. See above for details.")
            print("\n💡 To fix issues, run:")
            print("   pip install -r requirements.txt")

        print("=" * 60 + "\n")

        return success

    def auto_fix(self) -> bool:
        """Try to automatically fix issues."""
        print("\n🔧 Attempting automatic fixes...\n")

        if self.issues:
            print("Installing missing packages from requirements.txt...")
            try:
                result = subprocess.run(
                    [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"],
                    capture_output=True,
                    text=True,
                    timeout=300,
                )

                if result.returncode == 0:
                    print("✓ Installation successful")
                    self.issues.clear()
                    # Recheck
                    self.check_required_packages()
                    return True
                else:
                    print("✗ Installation failed")
                    print(result.stderr)
                    return False
            except Exception as e:
                print(f"✗ Error during installation: {e}")
                return False

        return True

    def suggest_next_steps(self):
        """Suggest next steps."""
        print("\n📋 SUGGESTED NEXT STEPS:\n")

        if any("torch" in issue.lower() for issue in self.issues):
            print("1. Install PyTorch with CUDA support:")
            print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118\n")

        if any("pyabsa" in issue.lower() for issue in self.issues):
            print("2. Install PyABSA:")
            print("   pip install pyabsa\n")

        print("3. Run training:")
        print("   python run_training.py --task asc --dry-run")
        print("   python run_training.py --task asc\n")

        print("4. View results:")
        print("   mlflow ui")
        print("   # Open http://localhost:5000\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Verify ABSA training pipeline dependencies"
    )

    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick check (critical packages only)",
    )

    parser.add_argument(
        "--fix",
        action="store_true",
        help="Try to auto-fix missing packages",
    )

    args = parser.parse_args()

    checker = DependencyChecker()

    if args.fix:
        checker.auto_fix()
        success = checker.run_full_check()
    elif args.quick:
        success = checker.run_quick_check()
    else:
        success = checker.run_full_check()

    if success:
        checker.suggest_next_steps()

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
