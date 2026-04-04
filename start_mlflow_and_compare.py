#!/usr/bin/env python
"""
Start MLflow UI and Run Model Comparison.

This script:
1. Starts MLflow tracking server
2. Opens MLflow UI in browser
3. Runs model comparison with different learning rates
4. Automatically shows results in MLflow

Usage:
    python start_mlflow_and_compare.py
    python start_mlflow_and_compare.py --learning-rates 1e-5 2e-5
    python start_mlflow_and_compare.py --epochs 5
"""

import argparse
import logging
import subprocess
import sys
import time
import webbrowser
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def start_mlflow_server(host: str = "127.0.0.1", port: int = 5000) -> subprocess.Popen:
    """
    Start MLflow tracking server.

    Args:
        host: Server host
        port: Server port

    Returns:
        Popen object for the server process
    """
    logger.info(f"Starting MLflow server on {host}:{port}")

    try:
        # Start MLflow server process
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "mlflow",
                "server",
                "--host",
                host,
                "--port",
                str(port),
                "--backend-store-uri",
                "sqlite:///mlflow.db",
                "--default-artifact-root",
                "mlruns",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        # Wait for server to start
        logger.info("Waiting for MLflow server to start...")
        time.sleep(3)

        logger.info(f"✓ MLflow server started on http://{host}:{port}")
        return process

    except Exception as e:
        logger.error(f"Failed to start MLflow server: {e}")
        return None


def open_mlflow_ui(host: str = "127.0.0.1", port: int = 5000) -> None:
    """
    Open MLflow UI in default browser.

    Args:
        host: Server host
        port: Server port
    """
    url = f"http://{host}:{port}"
    logger.info(f"Opening MLflow UI in browser: {url}")

    try:
        webbrowser.open(url)
        logger.info("✓ MLflow UI opened in browser")
    except Exception as e:
        logger.warning(f"Could not open browser: {e}")
        logger.info(f"Open manually: {url}")


def run_comparison(args: argparse.Namespace) -> None:
    """
    Run model comparison script.

    Args:
        args: Command-line arguments
    """
    logger.info("\nStarting model comparison...")
    logger.info(f"Learning Rates: {args.learning_rates}")
    logger.info(f"Epochs: {args.epochs}, Batch Size: {args.batch_size}\n")

    try:
        # Build command
        cmd = [
            sys.executable,
            "compare_models.py",
            "--epochs",
            str(args.epochs),
            "--batch-size",
            str(args.batch_size),
            "--learning-rates",
        ] + [str(lr) for lr in args.learning_rates]

        # Run comparison
        result = subprocess.run(cmd, timeout=7200)  # 2 hour timeout

        if result.returncode == 0:
            logger.info("\n✓ Comparison completed successfully!")
        else:
            logger.error(f"\n✗ Comparison failed with code {result.returncode}")
            sys.exit(1)

    except subprocess.TimeoutExpired:
        logger.error("Comparison timed out")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error running comparison: {e}")
        sys.exit(1)


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Start MLflow UI and run model comparison",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default: Start MLflow and compare 3 learning rates (1e-5, 2e-5, 5e-5)
  python start_mlflow_and_compare.py

  # Compare different learning rates
  python start_mlflow_and_compare.py --learning-rates 1e-5 3e-5 1e-4

  # Longer training (5 epochs)
  python start_mlflow_and_compare.py --epochs 5

  # Quick test (1 epoch)
  python start_mlflow_and_compare.py --epochs 1 --batch-size 8

  # Just start MLflow (no training)
  python start_mlflow_and_compare.py --no-train
        """,
    )

    parser.add_argument(
        "--learning-rates",
        type=float,
        nargs="+",
        default=[1e-5, 2e-5, 5e-5],
        help="Learning rates to compare (default: 1e-5 2e-5 5e-5)",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of epochs per training run (default: 3)",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size per training run (default: 16)",
    )

    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="MLflow server host (default: 127.0.0.1)",
    )

    parser.add_argument(
        "--port",
        type=int,
        default=5000,
        help="MLflow server port (default: 5000)",
    )

    parser.add_argument(
        "--no-train",
        action="store_true",
        help="Start MLflow UI only, don't run training",
    )

    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Don't open browser automatically",
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_arguments()

    logger.info("=" * 80)
    logger.info("ABSA MODEL COMPARISON WITH MLflow")
    logger.info("=" * 80)

    # Start MLflow server
    mlflow_process = start_mlflow_server(args.host, args.port)

    if not mlflow_process:
        logger.error("Failed to start MLflow server")
        sys.exit(1)

    try:
        # Open UI in browser
        if not args.no_browser:
            time.sleep(1)
            open_mlflow_ui(args.host, args.port)

        # Run comparison if requested
        if not args.no_train:
            logger.info("\n" + "=" * 80)
            logger.info("STARTING MODEL COMPARISON")
            logger.info("=" * 80)

            time.sleep(1)
            run_comparison(args)

        # Show final instructions
        logger.info("\n" + "=" * 80)
        logger.info("MLflow Dashboard")
        logger.info("=" * 80)
        logger.info(f"URL: http://{args.host}:{args.port}")
        logger.info("")
        logger.info("Features:")
        logger.info("  • View all training runs in Experiments")
        logger.info("  • Compare metrics side-by-side")
        logger.info("  • Check model registry (Models)")
        logger.info("  • View artifacts and parameters")
        logger.info("")
        logger.info("Press Ctrl+C to stop MLflow server")
        logger.info("=" * 80 + "\n")

        # Keep server running
        mlflow_process.wait()

    except KeyboardInterrupt:
        logger.info("\nShutting down MLflow server...")
        mlflow_process.terminate()
        mlflow_process.wait()
        logger.info("✓ MLflow server stopped")
        sys.exit(0)

    except Exception as e:
        logger.error(f"Error: {e}")
        mlflow_process.terminate()
        sys.exit(1)


if __name__ == "__main__":
    main()
