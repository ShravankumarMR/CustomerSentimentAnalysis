#!/usr/bin/env python
"""
ABSA Training Runner with CLI Options.

This script provides a command-line interface for training ABSA models with different configurations.

Usage:
    python run_training.py --task asc --epochs 5 --batch-size 32
    python run_training.py --task ate --model-name deberta --experiment "My-Experiment"
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import mlflow
import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.models.train_model import ABSATrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train ABSA models with PyABSA and MLflow tracking",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train ASC model with default config
  python run_training.py --task asc

  # Train with custom epochs and batch size
  python run_training.py --task asc --epochs 5 --batch-size 16

  # Train ATE model with custom experiment
  python run_training.py --task ate --experiment "My-ATE-Experiment"

  # Use custom config file
  python run_training.py --config my_config.yaml --task asc

  # Track in custom MLflow server
  python run_training.py --mlflow-uri http://localhost:5000
        """,
    )

    # Task selection
    parser.add_argument(
        "--task",
        type=str,
        choices=["asc", "ate", "both"],
        default="asc",
        help="Training task: asc=Aspect-Sentiment Classification, ate=Aspect-Term Extraction, both=train both",
    )

    # Configuration
    parser.add_argument(
        "--config",
        type=str,
        default="config/model.yaml",
        help="Path to configuration YAML file (default: config/model.yaml)",
    )

    # Training parameters
    parser.add_argument(
        "--epochs",
        type=int,
        help="Number of training epochs (overrides config)",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        help="Batch size for training (overrides config)",
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        help="Learning rate for optimizer (overrides config)",
    )

    parser.add_argument(
        "--max-length",
        type=int,
        help="Maximum sequence length (overrides config)",
    )

    # Model selection
    parser.add_argument(
        "--model-name",
        type=str,
        default="deberta-absa",
        help="Model name (default: deberta-absa)",
    )

    parser.add_argument(
        "--pretrained",
        type=str,
        default="yangheng/deberta-v3-base-absa-v1.1",
        help="Pretrained model (default: yangheng/deberta-v3-base-absa-v1.1)",
    )

    # Data paths
    parser.add_argument(
        "--train-data",
        type=str,
        default="data/processed/train_data.txt",
        help="Path to training data",
    )

    parser.add_argument(
        "--val-data",
        type=str,
        default="data/processed/val_data.txt",
        help="Path to validation data",
    )

    parser.add_argument(
        "--test-data",
        type=str,
        default="data/processed/test_data.txt",
        help="Path to test data",
    )

    # Output configuration
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models",
        help="Output directory for models (default: models)",
    )

    # MLflow configuration
    parser.add_argument(
        "--experiment",
        type=str,
        help="MLflow experiment name (default: ABSA-ASC-Fine-tuning or ABSA-ATE-Fine-tuning)",
    )

    parser.add_argument(
        "--mlflow-uri",
        type=str,
        help="MLflow tracking server URI (e.g., http://localhost:5000)",
    )

    parser.add_argument(
        "--run-name",
        type=str,
        help="MLflow run name (default: auto-generated)",
    )

    # Flags
    parser.add_argument(
        "--no-register",
        action="store_true",
        help="Skip model registration in MLflow Registry",
    )

    parser.add_argument(
        "--export-config",
        type=str,
        help="Export resolved configuration to JSON file",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show configuration without training",
    )

    return parser.parse_args()


def merge_config(config_path: str, args: argparse.Namespace) -> dict:
    """
    Merge CLI arguments with configuration file.
    CLI arguments override config file values.

    Args:
        config_path: Path to configuration file
        args: Parsed arguments

    Returns:
        Merged configuration dictionary
    """
    # Load config file
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Override with CLI arguments
    if args.epochs:
        config["training"]["epochs"] = args.epochs

    if args.batch_size:
        config["training"]["batch_size"] = args.batch_size

    if args.learning_rate:
        config["training"]["learning_rate"] = args.learning_rate

    if args.max_length:
        config["model"]["max_length"] = args.max_length

    config["model"]["name"] = args.model_name
    config["model"]["pretrained"] = args.pretrained

    return config


def setup_mlflow(args: argparse.Namespace):
    """Configure MLflow tracking."""
    if args.mlflow_uri:
        mlflow.set_tracking_uri(args.mlflow_uri)
        logger.info(f"MLflow tracking URI set to: {args.mlflow_uri}")


def train_asc(trainer: ABSATrainer, args: argparse.Namespace) -> dict:
    """Train Aspect-Sentiment Classification model."""
    logger.info("=" * 80)
    logger.info("Training Aspect-Sentiment Classification Model (ASC)")
    logger.info("=" * 80)

    asc_output_dir = os.path.join(args.output_dir, "asc_model")
    experiment_name = args.experiment or "ABSA-ASC-Fine-tuning"

    metrics = trainer.train_asc_model(
        train_path=args.train_data,
        val_path=args.val_data,
        test_path=args.test_data,
        output_dir=asc_output_dir,
        mlflow_experiment=experiment_name,
    )

    # Copy to models directory
    asc_final_path = trainer.copy_model_to_models_dir(
        trainer.best_model_path, args.output_dir
    )

    # Register in MLflow
    if not args.no_register:
        try:
            with mlflow.start_run(run_name="asc-model-registration"):
                mlflow.log_params({"model_type": "ASC", "task": "asc"})
                mlflow.log_metrics(metrics)

                logged_model = "runs:/{}/model".format(mlflow.active_run().info.run_id)
                trainer.register_model_to_mlflow(
                    model_uri=logged_model,
                    model_name="ABSA-ASC-Deberta-Hindi",
                    stage="Staging",
                )
        except Exception as e:
            logger.warning(f"Model registration skipped: {e}")

    logger.info(f"\n✓ ASC training completed")
    logger.info(f"  - Model path: {asc_final_path}")
    logger.info(f"  - Metrics: {metrics}")

    return {"task": "asc", "metrics": metrics, "model_path": asc_final_path}


def train_ate(trainer: ABSATrainer, args: argparse.Namespace) -> dict:
    """Train Aspect Term Extraction model."""
    logger.info("=" * 80)
    logger.info("Training Aspect Term Extraction Model (ATE)")
    logger.info("=" * 80)

    ate_output_dir = os.path.join(args.output_dir, "ate_model")
    experiment_name = args.experiment or "ABSA-ATE-Fine-tuning"

    metrics = trainer.train_ate_model(
        train_path=args.train_data,
        val_path=args.val_data,
        test_path=args.test_data,
        output_dir=ate_output_dir,
        mlflow_experiment=experiment_name,
    )

    # Copy to models directory
    ate_final_path = trainer.copy_model_to_models_dir(
        trainer.best_model_path, args.output_dir
    )

    # Register in MLflow
    if not args.no_register:
        try:
            with mlflow.start_run(run_name="ate-model-registration"):
                mlflow.log_params({"model_type": "ATE", "task": "ate"})
                mlflow.log_metrics(metrics)

                logged_model = "runs:/{}/model".format(mlflow.active_run().info.run_id)
                trainer.register_model_to_mlflow(
                    model_uri=logged_model,
                    model_name="ABSA-ATE-Deberta-Hindi",
                    stage="Staging",
                )
        except Exception as e:
            logger.warning(f"ATE model registration skipped: {e}")

    logger.info(f"\n✓ ATE training completed")
    logger.info(f"  - Model path: {ate_final_path}")
    logger.info(f"  - Metrics: {metrics}")

    return {"task": "ate", "metrics": metrics, "model_path": ate_final_path}


def show_config(config: dict, args: argparse.Namespace):
    """Display the configuration that will be used."""
    logger.info("=" * 80)
    logger.info("Training Configuration")
    logger.info("=" * 80)
    logger.info("\nModel Configuration:")
    logger.info(f"  Name: {config['model']['name']}")
    logger.info(f"  Pretrained: {config['model']['pretrained']}")
    logger.info(f"  Max Length: {config['model']['max_length']}")

    logger.info("\nTraining Configuration:")
    logger.info(f"  Epochs: {config['training']['epochs']}")
    logger.info(f"  Batch Size: {config['training']['batch_size']}")
    logger.info(f"  Learning Rate: {config['training']['learning_rate']}")

    logger.info("\nData Paths:")
    logger.info(f"  Train: {args.train_data}")
    logger.info(f"  Validation: {args.val_data}")
    logger.info(f"  Test: {args.test_data}")

    logger.info("\nMLflow Configuration:")
    logger.info(f"  Experiment: {args.experiment or 'Default for task'}")
    logger.info(f"  Register Model: {not args.no_register}")

    logger.info("\nTask:")
    logger.info(f"  Task: {args.task}")
    logger.info("=" * 80 + "\n")


def main():
    """Main entry point."""
    args = parse_arguments()

    # Setup MLflow
    setup_mlflow(args)

    # Load and merge configuration
    if not os.path.exists(args.config):
        logger.error(f"Config file not found: {args.config}")
        sys.exit(1)

    config = merge_config(args.config, args)

    # Show configuration
    show_config(config, args)

    # Export config if requested
    if args.export_config:
        with open(args.export_config, "w") as f:
            json.dump(config, f, indent=2)
        logger.info(f"Configuration exported to {args.export_config}")

    # Dry run - exit after showing config
    if args.dry_run:
        logger.info("Dry run mode - exiting without training")
        return

    # Verify data files exist
    for path in [args.train_data, args.val_data, args.test_data]:
        if not os.path.exists(path):
            logger.error(f"Data file not found: {path}")
            sys.exit(1)

    # Initialize trainer
    trainer = ABSATrainer(config_path=args.config)

    results = []

    try:
        if args.task in ["asc", "both"]:
            result = train_asc(trainer, args)
            results.append(result)

        if args.task in ["ate", "both"]:
            result = train_ate(trainer, args)
            results.append(result)

        # Save training summary
        summary = {
            "tasks_trained": args.task,
            "results": results,
            "config": config,
        }

        summary_path = os.path.join(args.output_dir, "training_results.json")
        os.makedirs(args.output_dir, exist_ok=True)
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        logger.info(f"\n{'=' * 80}")
        logger.info("All training tasks completed successfully!")
        logger.info(f"Results saved to: {summary_path}")
        logger.info(f"{'=' * 80}")

    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
