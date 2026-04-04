#!/usr/bin/env python
"""
ABSA Model Comparison Script with Multiple Learning Rates.

This script trains the ABSA model with different learning rates,
compares the runs in MLflow, and promotes the best model to Production.

Usage:
    python compare_models.py
    python compare_models.py --learning-rates 1e-5 2e-5 5e-5
    python compare_models.py --epochs 3 --batch-size 16
"""

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import mlflow
import yaml
from tabulate import tabulate

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class ModelComparator:
    """Compare ABSA models trained with different hyperparameters."""

    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize comparator."""
        self.config_path = config_path
        self.runs = []
        self.best_run = None
        self.comparison_results = {}

    def run_training(
        self,
        learning_rate: float,
        epochs: int,
        batch_size: int,
        run_name: str,
    ) -> str:
        """
        Run training with specific hyperparameters.

        Args:
            learning_rate: Learning rate for training
            epochs: Number of epochs
            batch_size: Batch size
            run_name: Name for this run

        Returns:
            MLflow run ID
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"Training Run: {run_name}")
        logger.info(f"Parameters: lr={learning_rate}, epochs={epochs}, batch={batch_size}")
        logger.info(f"{'='*80}\n")

        try:
            # Run training with specified parameters
            cmd = [
                sys.executable,
                "run_training.py",
                "--task",
                "asc",
                "--epochs",
                str(epochs),
                "--batch-size",
                str(batch_size),
                "--learning-rate",
                str(learning_rate),
            ]

            # Run the training script
            result = subprocess.run(cmd, capture_output=False, text=True, timeout=3600)

            if result.returncode != 0:
                logger.error(f"Training failed with return code {result.returncode}")
                return None

            # Get the latest run ID
            client = mlflow.tracking.MlflowClient()
            experiments = client.search_experiments()
            experiment_id = None

            for exp in experiments:
                if "ABSA-ASC" in exp.name:
                    experiment_id = exp.experiment_id
                    break

            if not experiment_id:
                logger.error("Could not find ABSA experiment")
                return None

            # Get the latest run
            runs = client.search_runs(experiment_ids=[experiment_id])
            if runs:
                run_id = runs[0].info.run_id
                logger.info(f"Training completed. Run ID: {run_id}")
                return run_id

        except subprocess.TimeoutExpired:
            logger.error("Training timed out")
            return None
        except Exception as e:
            logger.error(f"Training error: {e}")
            return None

    def compare_runs(self, run_ids: List[str]) -> Dict:
        """
        Compare multiple MLflow runs.

        Args:
            run_ids: List of MLflow run IDs to compare

        Returns:
            Dictionary with comparison results
        """
        logger.info("\n" + "=" * 80)
        logger.info("Comparing Runs")
        logger.info("=" * 80 + "\n")

        client = mlflow.tracking.MlflowClient()
        comparison_data = []

        metrics_to_compare = ["f1", "accuracy", "precision", "recall"]

        for run_id in run_ids:
            try:
                run = client.get_run(run_id)
                run_name = run.data.tags.get("mlflow.runName", run_id[:8])

                row = {
                    "Run Name": run_name,
                    "Run ID": run_id[:8],
                }

                # Get parameters
                params = run.data.params
                row["Learning Rate"] = params.get("learning_rate", "N/A")
                row["Epochs"] = params.get("epochs", "N/A")
                row["Batch Size"] = params.get("batch_size", "N/A")

                # Get metrics
                metrics = run.data.metrics
                for metric in metrics_to_compare:
                    # Find metric with test prefix
                    metric_key = None
                    for key in metrics.keys():
                        if metric in key.lower():
                            metric_key = key
                            break

                    if metric_key:
                        row[f"Test {metric.upper()}"] = f"{metrics[metric_key]:.4f}"
                    else:
                        row[f"Test {metric.upper()}"] = "N/A"

                # Get status
                row["Status"] = run.info.status

                comparison_data.append(row)

            except Exception as e:
                logger.error(f"Error comparing run {run_id}: {e}")
                continue

        # Sort by F1 score (descending)
        comparison_data.sort(
            key=lambda x: float(x.get("Test F1", "0").replace("N/A", "0")),
            reverse=True,
        )

        # Print comparison table
        print("\n" + "=" * 80)
        print("MODEL COMPARISON RESULTS")
        print("=" * 80)
        print(tabulate(comparison_data, headers="keys", tablefmt="grid"))
        print("=" * 80 + "\n")

        return comparison_data

    def get_best_run(self, run_ids: List[str]) -> Tuple[Optional[str], Dict]:
        """
        Identify the best run based on F1 score.

        Args:
            run_ids: List of MLflow run IDs

        Returns:
            Tuple of (best_run_id, best_metrics)
        """
        client = mlflow.tracking.MlflowClient()
        best_f1 = -1
        best_run_id = None
        best_metrics = {}

        for run_id in run_ids:
            try:
                run = client.get_run(run_id)
                metrics = run.data.metrics

                # Find F1 score
                f1_score = None
                for key, value in metrics.items():
                    if "f1" in key.lower():
                        f1_score = value
                        break

                if f1_score is not None and f1_score > best_f1:
                    best_f1 = f1_score
                    best_run_id = run_id
                    best_metrics = {k: v for k, v in metrics.items()}

            except Exception as e:
                logger.error(f"Error evaluating run {run_id}: {e}")
                continue

        return best_run_id, best_metrics

    def promote_best_model(
        self, best_run_id: str, model_name: str = "ABSA-ASC-Deberta-Hindi"
    ) -> None:
        """
        Promote the best model to Production stage in MLflow Registry.

        Args:
            best_run_id: ID of the best run
            model_name: Name of the model in registry
        """
        if not best_run_id:
            logger.error("No best run ID provided")
            return

        try:
            logger.info(f"\nPromoting model from run {best_run_id} to Production")

            client = mlflow.tracking.MlflowClient()

            # Get all versions of this model
            try:
                versions = client.search_model_versions(
                    filter_string=f"name='{model_name}'"
                )

                if versions:
                    # Find version from best run
                    for version in versions:
                        if best_run_id in version.source:
                            # Transition to Production
                            client.transition_model_version_stage(
                                name=model_name,
                                version=version.version,
                                stage="Production",
                            )
                            logger.info(
                                f"✓ Model {model_name} v{version.version} promoted to Production"
                            )
                            return

                logger.warning(
                    "Could not find model version for best run in registry"
                )

            except mlflow.exceptions.MlflowException as e:
                if "does not exist" in str(e):
                    logger.warning(f"Model {model_name} not found in registry")
                else:
                    raise

        except Exception as e:
            logger.error(f"Error promoting model: {e}")

    def generate_report(
        self,
        run_ids: List[str],
        best_run_id: str,
        output_file: str = "models/comparison_report.json",
    ) -> None:
        """
        Generate comparison report.

        Args:
            run_ids: List of run IDs
            best_run_id: Best run ID
            output_file: Where to save report
        """
        client = mlflow.tracking.MlflowClient()

        report = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_runs": len(run_ids),
            "best_run_id": best_run_id,
            "runs": [],
        }

        for run_id in run_ids:
            try:
                run = client.get_run(run_id)
                run_info = {
                    "run_id": run_id,
                    "name": run.data.tags.get("mlflow.runName", "Unknown"),
                    "parameters": run.data.params,
                    "metrics": run.data.metrics,
                    "is_best": run_id == best_run_id,
                }
                report["runs"].append(run_info)
            except Exception as e:
                logger.error(f"Error generating report for {run_id}: {e}")

        # Save report
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(report, f, indent=2)

        logger.info(f"Comparison report saved to {output_file}")

    def run_comparison(
        self,
        learning_rates: List[float],
        epochs: int = 3,
        batch_size: int = 16,
    ) -> None:
        """
        Run full comparison pipeline.

        Args:
            learning_rates: List of learning rates to try
            epochs: Number of epochs for each run
            batch_size: Batch size for each run
        """
        logger.info("=" * 80)
        logger.info("ABSA MODEL COMPARISON - DIFFERENT LEARNING RATES")
        logger.info("=" * 80)
        logger.info(f"Learning Rates: {learning_rates}")
        logger.info(f"Epochs: {epochs}, Batch Size: {batch_size}\n")

        run_ids = []

        # Train models with different learning rates
        for i, lr in enumerate(learning_rates, 1):
            run_name = f"lr-{lr}-run-{i}"
            logger.info(f"\nTraining {i}/{len(learning_rates)}: {run_name}")

            run_id = self.run_training(
                learning_rate=lr,
                epochs=epochs,
                batch_size=batch_size,
                run_name=run_name,
            )

            if run_id:
                run_ids.append(run_id)
                logger.info(f"✓ Completed: {run_name} (ID: {run_id[:8]})")
            else:
                logger.error(f"✗ Failed: {run_name}")

        # Compare runs
        if len(run_ids) > 0:
            logger.info(f"\nSuccessfully completed {len(run_ids)} runs")

            # Compare
            comparison = self.compare_runs(run_ids)

            # Find best
            best_run_id, best_metrics = self.get_best_run(run_ids)

            if best_run_id:
                logger.info("\n" + "=" * 80)
                logger.info("BEST MODEL IDENTIFIED")
                logger.info("=" * 80)
                logger.info(f"Run ID: {best_run_id}")
                logger.info(f"Metrics:")
                for metric, value in best_metrics.items():
                    if isinstance(value, float):
                        logger.info(f"  {metric}: {value:.4f}")

                # Promote to production
                self.promote_best_model(
                    best_run_id, model_name="ABSA-ASC-Deberta-Hindi"
                )

                # Generate report
                self.generate_report(run_ids, best_run_id)

                logger.info("\n✓ Comparison complete!")
                logger.info("View results in MLflow: mlflow ui")

            else:
                logger.error("Could not identify best model")

        else:
            logger.error("No successful runs to compare")


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Compare ABSA models with different learning rates",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare default learning rates (1e-5, 2e-5, 5e-5)
  python compare_models.py

  # Compare custom learning rates
  python compare_models.py --learning-rates 1e-5 3e-5 1e-4

  # Longer training
  python compare_models.py --epochs 5 --batch-size 32

  # Quick test
  python compare_models.py --epochs 1 --batch-size 8
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
        help="Number of epochs (default: 3)",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size (default: 16)",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="models/comparison_report.json",
        help="Output file for comparison report",
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_arguments()

    # Initialize comparator
    comparator = ModelComparator()

    try:
        # Run comparison
        comparator.run_comparison(
            learning_rates=args.learning_rates,
            epochs=args.epochs,
            batch_size=args.batch_size,
        )

    except KeyboardInterrupt:
        logger.info("\nComparison interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Comparison failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
