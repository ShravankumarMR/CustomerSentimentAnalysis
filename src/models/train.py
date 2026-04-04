"""
Training module for sentiment analysis model.

This module provides the main entry point for training ABSA models.
Uses train_model.py for actual training logic with PyABSA and MLflow.
"""

import logging
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.train_model import ABSATrainer, main

logger = logging.getLogger(__name__)


def train_model(config_path: str = "config/model.yaml"):
    """
    Train the sentiment analysis model using PyABSA.
    
    Args:
        config_path: Path to model configuration YAML file.
    """
    logger.info("Starting ABSA model training...")
    main()
    logger.info("Model training completed")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    train_model()
