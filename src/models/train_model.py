"""
ABSA Training Module using PyABSA for Aspect-Based Sentiment Analysis.

This module fine-tunes a multilingual ABSA model (DeBERTa-v3-base or similar)
on Hindi/Hinglish restaurant review data with MLflow experiment tracking.

The data format is: text$$$[aspect]$$$sentiment
Example: "Great food$$$[food]$$$positive"

Author: MLOps Team
"""

import json
import logging
import os
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import mlflow
import mlflow.pytorch
import pandas as pd
import torch
import yaml
from pyabsa import AspectExtraction, AspectSentimentClassification
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import set_seed

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {DEVICE}")


class ABSATrainer:
    """Trainer class for Aspect-Based Sentiment Analysis model fine-tuning."""

    def __init__(self, config_path: str = "config/model.yaml"):
        """
        Initialize the ABSA trainer.

        Args:
            config_path: Path to model configuration YAML file.
        """
        self.config = self._load_config(config_path)
        self.device = DEVICE
        self.model = None
        self.best_model_path = None
        self.best_metrics = {}

    @staticmethod
    def _load_config(config_path: str) -> Dict:
        """Load YAML configuration file."""
        try:
            with open(config_path, "r") as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.warning(f"Config file {config_path} not found. Using defaults.")
            return {
                "model": {
                    "name": "deberta-absa",
                    "pretrained": "yangheng/deberta-v3-base-absa-v1.1",
                    "max_length": 512,
                },
                "training": {
                    "epochs": 5,
                    "batch_size": 32,
                    "learning_rate": 2e-5,
                    "warmup_steps": 500,
                },
                "evaluation": {
                    "metrics": ["accuracy", "precision", "recall", "f1-score"]
                },
            }

    def load_data(
        self, data_path: str, split: str = "train"
    ) -> Tuple[List[str], List[str], List[str]]:
        """
        Load and parse ABSA data from file.

        Format: text$$$[aspect]$$$sentiment

        Args:
            data_path: Path to data file
            split: Data split name (train/val/test)

        Returns:
            Tuple of (texts, aspects, sentiments)
        """
        logger.info(f"Loading {split} data from {data_path}")

        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found: {data_path}")

        texts, aspects, sentiments = [], [], []

        try:
            with open(data_path, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        parts = line.split("$$$")
                        if len(parts) != 3:
                            logger.warning(
                                f"Skipping malformed line {line_num}: {line[:50]}"
                            )
                            continue

                        text, aspect, sentiment = parts
                        aspect = aspect.strip("[]").lower()
                        sentiment = sentiment.strip().lower()

                        if text and aspect and sentiment:
                            texts.append(text.strip())
                            aspects.append(aspect)
                            sentiments.append(sentiment)
                    except Exception as e:
                        logger.warning(f"Error parsing line {line_num}: {e}")
                        continue

        except UnicodeDecodeError:
            logger.warning("UTF-8 decoding failed, trying latin-1")
            with open(data_path, "r", encoding="latin-1") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        parts = line.split("$$$")
                        if len(parts) == 3:
                            texts.append(parts[0].strip())
                            aspects.append(parts[1].strip("[]").lower())
                            sentiments.append(parts[2].strip().lower())

        logger.info(f"Loaded {len(texts)} samples for {split} split")
        if len(texts) == 0:
            raise ValueError(f"No valid samples found in {data_path}")

        return texts, aspects, sentiments

    def prepare_absa_data(
        self,
        texts: List[str],
        aspects: List[str],
        sentiments: List[str],
        output_path: str,
    ) -> None:
        """
        Prepare data in PyABSA format.

        PyABSA expects: text {aspect} sentiment
        or for ATE: text {aspect_term}

        Args:
            texts: List of review texts
            aspects: List of aspect categories
            sentiments: List of sentiment labels
            output_path: Path to save formatted data
        """
        logger.info(f"Preparing ABSA format data for {output_path}")

        with open(output_path, "w", encoding="utf-8") as f:
            for text, aspect, sentiment in zip(texts, aspects, sentiments):
                # Format: text {aspect} sentiment
                formatted_line = f"{text} {{{aspect}}} {sentiment}\n"
                f.write(formatted_line)

        logger.info(f"Data prepared: {output_path}")

    def train_asc_model(
        self,
        train_path: str,
        val_path: str,
        test_path: str,
        output_dir: str = "models/absa_model",
        mlflow_experiment: str = "ABSA-Fine-tuning",
    ) -> Dict:
        """
        Train Aspect-Sentiment Classification model using PyABSA.

        Args:
            train_path: Path to training data
            val_path: Path to validation data
            test_path: Path to test data
            output_dir: Directory to save trained model
            mlflow_experiment: MLflow experiment name

        Returns:
            Dictionary of final metrics
        """
        # Set MLflow experiment
        mlflow.set_experiment(mlflow_experiment)

        with mlflow.start_run(run_name="absa-asc-training"):
            logger.info("Starting ABSA training with MLflow tracking")

            # Log hyperparameters
            params = self.config["training"]
            params.update(self.config["model"])
            mlflow.log_params({
                "model_name": self.config["model"]["name"],
                "pretrained_model": self.config["model"]["pretrained"],
                "epochs": params["epochs"],
                "batch_size": params["batch_size"],
                "learning_rate": params["learning_rate"],
                "max_length": self.config["model"]["max_length"],
            })

            try:
                # Prepare data in PyABSA format
                train_file = f"{output_dir}_train.txt"
                val_file = f"{output_dir}_val.txt"
                test_file = f"{output_dir}_test.txt"

                os.makedirs(output_dir, exist_ok=True)

                texts_train, aspects_train, sentiments_train = self.load_data(
                    train_path, "train"
                )
                texts_val, aspects_val, sentiments_val = self.load_data(
                    val_path, "val"
                )
                texts_test, aspects_test, sentiments_test = self.load_data(
                    test_path, "test"
                )

                self.prepare_absa_data(
                    texts_train, aspects_train, sentiments_train, train_file
                )
                self.prepare_absa_data(
                    texts_val, aspects_val, sentiments_val, val_file
                )
                self.prepare_absa_data(
                    texts_test, aspects_test, sentiments_test, test_file
                )

                # Configure training args
                training_args = {
                    "model_name": self.config["model"]["name"],
                    "num_epoch": self.config["training"]["epochs"],
                    "batch_size": self.config["training"]["batch_size"],
                    "learning_rate": self.config["training"]["learning_rate"],
                    "pretrained_bert": self.config["model"]["pretrained"],
                    "max_seq_len": self.config["model"]["max_length"],
                    "device": str(self.device),
                    "seed": 42,
                    "log_step": 5,
                    "repeat_dataset": 1,
                    "patience": 5,
                }

                # Initialize trainer
                aspect_extractor = AspectSentimentClassification(
                    train_file,
                    eval_dataset=val_file,
                    task_name="asc",
                    **training_args,
                )

                set_seed(42)

                logger.info("Starting fine-tuning...")
                aspect_extractor.train(
                    early_stopping_patience=5, metric_to_watch="f1"
                )

                # Evaluate on test set
                logger.info("Evaluating on test set...")
                test_results = aspect_extractor.evaluate(test_file)

                logger.info(f"Test results: {test_results}")

                # Save model
                model_save_path = os.path.join(output_dir, "best_model")
                os.makedirs(model_save_path, exist_ok=True)

                aspect_extractor.save_model(model_save_path)
                self.best_model_path = model_save_path

                logger.info(f"Model saved to {model_save_path}")

                # Log metrics
                if isinstance(test_results, dict):
                    self.best_metrics = test_results
                    for metric_name, metric_value in test_results.items():
                        if isinstance(metric_value, (int, float)):
                            mlflow.log_metric(f"test_{metric_name}", metric_value)
                            logger.info(f"test_{metric_name}: {metric_value}")

                # Log model artifacts
                mlflow.log_artifacts(model_save_path, artifact_path="model")

                # Log training config
                config_path_log = os.path.join(output_dir, "training_config.json")
                with open(config_path_log, "w") as f:
                    json.dump(training_args, f, indent=2)
                mlflow.log_artifact(config_path_log)

                logger.info(f"Training completed. Best metrics: {self.best_metrics}")

                return self.best_metrics

            except Exception as e:
                logger.error(f"Training failed: {e}")
                mlflow.log_param("training_status", "failed")
                raise

    def train_ate_model(
        self,
        train_path: str,
        val_path: str,
        test_path: str,
        output_dir: str = "models/ate_model",
        mlflow_experiment: str = "ABSA-ATE-Fine-tuning",
    ) -> Dict:
        """
        Train Aspect Term Extraction model using PyABSA.

        Args:
            train_path: Path to training data
            val_path: Path to validation data
            test_path: Path to test data
            output_dir: Directory to save trained model
            mlflow_experiment: MLflow experiment name

        Returns:
            Dictionary of final metrics
        """
        mlflow.set_experiment(mlflow_experiment)

        with mlflow.start_run(run_name="absa-ate-training"):
            logger.info("Starting Aspect Term Extraction training")

            params = self.config["training"]
            mlflow.log_params({
                "model_name": "aspect-term-extraction",
                "pretrained_model": self.config["model"]["pretrained"],
                "epochs": params["epochs"],
                "batch_size": params["batch_size"],
                "learning_rate": params["learning_rate"],
            })

            try:
                train_file = f"{output_dir}_train.txt"
                val_file = f"{output_dir}_val.txt"
                test_file = f"{output_dir}_test.txt"

                os.makedirs(output_dir, exist_ok=True)

                # Load data
                texts_train, aspects_train, _ = self.load_data(train_path, "train")
                texts_val, aspects_val, _ = self.load_data(val_path, "val")
                texts_test, aspects_test, _ = self.load_data(test_path, "test")

                # Format for ATE: text {aspect}
                with open(train_file, "w", encoding="utf-8") as f:
                    for text, aspect in zip(texts_train, aspects_train):
                        f.write(f"{text} {{{aspect}}}\n")

                with open(val_file, "w", encoding="utf-8") as f:
                    for text, aspect in zip(texts_val, aspects_val):
                        f.write(f"{text} {{{aspect}}}\n")

                with open(test_file, "w", encoding="utf-8") as f:
                    for text, aspect in zip(texts_test, aspects_test):
                        f.write(f"{text} {{{aspect}}}\n")

                training_args = {
                    "model_name": "ate-model",
                    "num_epoch": self.config["training"]["epochs"],
                    "batch_size": self.config["training"]["batch_size"],
                    "learning_rate": self.config["training"]["learning_rate"],
                    "pretrained_bert": self.config["model"]["pretrained"],
                    "max_seq_len": self.config["model"]["max_length"],
                    "device": str(self.device),
                    "seed": 42,
                }

                ate_trainer = AspectExtraction(
                    train_file, eval_dataset=val_file, task_name="ate", **training_args
                )

                set_seed(42)
                logger.info("Starting ATE fine-tuning...")
                ate_trainer.train(early_stopping_patience=5, metric_to_watch="f1")

                # Evaluate
                test_results = ate_trainer.evaluate(test_file)
                logger.info(f"ATE Test results: {test_results}")

                # Save model
                model_save_path = os.path.join(output_dir, "best_model")
                os.makedirs(model_save_path, exist_ok=True)
                ate_trainer.save_model(model_save_path)

                self.best_model_path = model_save_path

                if isinstance(test_results, dict):
                    self.best_metrics = test_results
                    for metric_name, metric_value in test_results.items():
                        if isinstance(metric_value, (int, float)):
                            mlflow.log_metric(f"ate_test_{metric_name}", metric_value)

                mlflow.log_artifacts(model_save_path, artifact_path="ate_model")

                logger.info("ATE training completed")
                return self.best_metrics

            except Exception as e:
                logger.error(f"ATE training failed: {e}")
                raise

    def register_model_to_mlflow(
        self, model_uri: str, model_name: str, stage: str = "Production"
    ) -> None:
        """
        Register trained model in MLflow Model Registry.

        Args:
            model_uri: URI of the model to register
            model_name: Name for the registered model
            stage: Stage to transition model to (Staging/Production)
        """
        try:
            logger.info(f"Registering model: {model_name}")

            result = mlflow.register_model(model_uri, model_name)
            logger.info(f"Model registered: {result.name} version {result.version}")

            # Transition to stage
            client = mlflow.tracking.MlflowClient()
            client.transition_model_version_stage(
                name=model_name, version=result.version, stage=stage
            )

            logger.info(f"Model transitioned to {stage} stage")

        except mlflow.exceptions.MlflowException as e:
            if "already exists" in str(e):
                logger.info(f"Model {model_name} already registered")
            else:
                logger.error(f"Failed to register model: {e}")
                raise

    def copy_model_to_models_dir(
        self, source_path: str, dest_dir: str = "models"
    ) -> str:
        """
        Copy best model to models directory.

        Args:
            source_path: Source path of trained model
            dest_dir: Destination models directory

        Returns:
            Path to copied model
        """
        os.makedirs(dest_dir, exist_ok=True)
        model_name = os.path.basename(source_path)
        dest_path = os.path.join(dest_dir, model_name)

        if os.path.exists(dest_path):
            shutil.rmtree(dest_path)

        shutil.copytree(source_path, dest_path)
        logger.info(f"Model copied to {dest_path}")

        return dest_path


def main():
    """Main training pipeline."""
    logger.info("=" * 80)
    logger.info("Customer Sentiment Analysis - ABSA Training Pipeline")
    logger.info("=" * 80)

    # Initialize trainer
    trainer = ABSATrainer(config_path="config/model.yaml")

    # Data paths
    train_path = "data/processed/train_data.txt"
    val_path = "data/processed/val_data.txt"
    test_path = "data/processed/test_data.txt"

    # Verify data exists
    for path in [train_path, val_path, test_path]:
        if not os.path.exists(path):
            logger.error(f"Data file not found: {path}")
            sys.exit(1)

    try:
        # Train ASC model (main task)
        logger.info("\n{'=' * 80}")
        logger.info("Training Aspect-Sentiment Classification Model")
        logger.info("=" * 80)

        asc_metrics = trainer.train_asc_model(
            train_path=train_path,
            val_path=val_path,
            test_path=test_path,
            output_dir="models/asc_model",
            mlflow_experiment="ABSA-ASC-Fine-tuning",
        )

        # Copy model to models directory
        asc_final_path = trainer.copy_model_to_models_dir(
            trainer.best_model_path, "models"
        )

        # Register in MLflow
        with mlflow.start_run(run_name="model-registration"):
            mlflow.log_params({"model_type": "ASC"})
            mlflow.log_metrics(asc_metrics)

            # Register model
            logged_model = "runs:/{}/model".format(mlflow.active_run().info.run_id)
            trainer.register_model_to_mlflow(
                model_uri=logged_model,
                model_name="ABSA-ASC-Deberta-Hindi",
                stage="Staging",
            )

        logger.info(f"\n✓ ASC Model training completed")
        logger.info(f"  - Best model path: {asc_final_path}")
        logger.info(f"  - Metrics: {asc_metrics}")

        # Save summary
        summary = {
            "model_type": "Aspect-Sentiment Classification",
            "model_name": "yangheng/deberta-v3-base-absa-v1.1",
            "language": "Hindi/Hinglish",
            "metrics": asc_metrics,
            "model_path": asc_final_path,
            "epochs": trainer.config["training"]["epochs"],
            "batch_size": trainer.config["training"]["batch_size"],
        }

        summary_path = "models/training_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        logger.info(f"\nTraining summary saved to {summary_path}")
        logger.info("=" * 80)
        logger.info("Training pipeline completed successfully!")
        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"Training pipeline failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
