"""
Feature Engineering Module for Customer Sentiment Analysis
Handles text cleaning, PyABSA format conversion, and data splitting
"""

import json
import logging
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Configuration
tokenizer = "transformer"  # Options: transformer, nltk, basic
max_length = 512
feature_type = "aspect_sentiment"  # Options: aspect_sentiment, sentiment_only
RANDOM_SEED = 42


class TextCleaner:
    """Handles text cleaning for both Devanagari and Roman Hindi"""

    def __init__(self):
        """Initialize regex patterns for text cleaning"""
        # Devanagari script ranges
        self.devanagari_pattern = re.compile(r'[\u0900-\u097F]+')
        # Roman Hindi pattern (transliterated Hindi with special chars)
        self.roman_hindi_pattern = re.compile(r'[a-zA-Z\-ā-ū]{2,}', re.UNICODE)

    def clean_devanagari(self, text: str) -> str:
        """Clean Devanagari Hindi text"""
        if not text:
            return ""

        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        # Normalize common punctuation
        text = text.replace('।', '।')  # Devanagari danda
        text = text.replace('॥', '।')  # Double danda

        # Remove special characters except common Hindi punctuation
        text = re.sub(r'[^\u0900-\u097F\s।!?,\'-]', '', text)

        # Remove multiple spaces
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    def clean_roman_hindi(self, text: str) -> str:
        """Clean Roman/Transliterated Hindi text"""
        if not text:
            return ""

        # Convert to lowercase for consistency
        text = text.lower()

        # Remove URLs
        text = re.sub(r'http\S+|www\S+', '', text)

        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)

        # Remove mentions and hashtags
        text = re.sub(r'@\w+|#\w+', '', text)

        # Remove extra punctuation but keep meaningful ones
        text = re.sub(r'[^\w\s\.\!\?,-]', '', text)

        # Remove multiple spaces
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    def detect_script(self, text: str) -> str:
        """Detect if text is Devanagari or Roman Hindi"""
        devanagari_chars = len(re.findall(self.devanagari_pattern, text))
        total_chars = len(text)

        if total_chars == 0:
            return "empty"

        if devanagari_chars / total_chars > 0.5:
            return "devanagari"
        else:
            return "roman"

    def clean(self, text: str) -> str:
        """Main cleaning function - auto-detects script and cleans accordingly"""
        if not text or not isinstance(text, str):
            return ""

        script = self.detect_script(text)

        if script == "devanagari":
            return self.clean_devanagari(text)
        else:
            return self.clean_roman_hindi(text)


class PyABSAConverter:
    """Converts cleaned text to PyABSA format (aspect + sentiment triplets)"""

    # Common aspect terms for restaurant/product reviews
    COMMON_ASPECTS = {
        # Food aspects
        "टेस्ट": "taste",
        "स्वाद": "taste",
        "taste": "taste",
        "quality": "quality",
        "गुणवत्ता": "quality",
        "freshness": "freshness",
        "temperature": "temperature",
        # Service aspects
        "service": "service",
        "सेवा": "service",
        "staff": "staff",
        "delivery": "delivery",
        # Ambiance aspects
        "ambiance": "ambiance",
        "atmosphere": "ambiance",
        "hygiene": "cleanliness",
        # Price aspects
        "price": "price",
        "कीमत": "price",
        "cost": "price",
        # Restaurant aspects
        "restaurant": "restaurant",
        "place": "restaurant",
    }

    # Sentiment indicators
    POSITIVE_INDICATORS = {
        "अच्छा", "बढ़िया", "शानदार", "बेहतरीन", "स्वादिष्ट",
        "good", "great", "excellent", "amazing", "delicious", "awesome",
        "nice", "perfect", "lovely", "best",
    }

    NEGATIVE_INDICATORS = {
        "बुरा", "खराब", "भयानक", "असमर्थ", "दुर्गंध",
        "bad", "poor", "terrible", "awful", "horrible", "worst",
        "dirty", "rude", "cold",
    }

    def extract_aspects(self, text: str) -> List[str]:
        """Extract aspect terms from text"""
        aspects = []
        text_lower = text.lower()

        for aspect_term, normalized in self.COMMON_ASPECTS.items():
            if aspect_term.lower() in text_lower:
                if normalized not in aspects:
                    aspects.append(normalized)

        # Default aspect if none found
        if not aspects:
            aspects = ["general"]

        return aspects

    def infer_sentiment(self, text: str) -> str:
        """Infer sentiment from text using keyword matching"""
        text_lower = text.lower()

        # Count positive and negative indicators
        pos_count = sum(1 for word in self.POSITIVE_INDICATORS if word in text_lower)
        neg_count = sum(1 for word in self.NEGATIVE_INDICATORS if word in text_lower)

        if pos_count > neg_count:
            return "positive"
        elif neg_count > pos_count:
            return "negative"
        else:
            return "neutral"

    def convert_to_pyabsa_format(self, text: str) -> List[str]:
        """
        Convert text to PyABSA format: text$$$aspect$$$sentiment
        Returns list of triplets (one per aspect found)
        """
        if not text or len(text.strip()) == 0:
            return []

        aspects = self.extract_aspects(text)
        sentiment = self.infer_sentiment(text)

        # Create PyABSA format triplets
        triplets = []
        for aspect in aspects:
            pyabsa_line = f"{text}$$$[{aspect}]$$${sentiment}"
            triplets.append(pyabsa_line)

        return triplets


class DataSplitter:
    """Handles train/val/test splitting"""

    def __init__(self, train_ratio: float = 0.8, val_ratio: float = 0.1, test_ratio: float = 0.1):
        """Initialize split ratios"""
        total = train_ratio + val_ratio + test_ratio
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"Split ratios must sum to 1.0, got {total}")

        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio

    def split_data(
        self, data: List[str], random_state: int = RANDOM_SEED
    ) -> Tuple[List[str], List[str], List[str]]:
        """
        Split data into train, val, test sets with specified ratios
        Returns: (train_data, val_data, test_data)
        """
        # First split: train vs (val + test)
        train_data, temp_data = train_test_split(
            data,
            train_size=self.train_ratio,
            random_state=random_state,
        )

        # Second split: val vs test from remaining data
        val_size = self.val_ratio / (self.val_ratio + self.test_ratio)
        val_data, test_data = train_test_split(
            temp_data,
            train_size=val_size,
            random_state=random_state,
        )

        return train_data, val_data, test_data


class FeatureBuilder:
    """Main feature building pipeline"""

    def __init__(
        self,
        input_path: str = "data/processed",
        output_path: str = "data/processed",
        test_size: float = 0.1,
        val_size: float = 0.1,
    ):
        """Initialize feature builder"""
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
        self.test_size = test_size
        self.val_size = val_size
        self.train_size = 1.0 - test_size - val_size

        self.cleaner = TextCleaner()
        self.converter = PyABSAConverter()
        self.splitter = DataSplitter(
            train_ratio=self.train_size,
            val_ratio=self.val_size,
            test_ratio=self.test_size,
        )

        # Create output directory
        self.output_path.mkdir(parents=True, exist_ok=True)

    def load_processed_data(self) -> pd.DataFrame:
        """Load processed data from previous stage"""
        processed_file = self.input_path / "processed_data.csv"

        if not processed_file.exists():
            logger.warning(f"Processed data not found at {processed_file}")
            logger.info("Creating sample data for demonstration")
            return self._create_sample_data()

        logger.info(f"Loading processed data from {processed_file}")
        return pd.read_csv(processed_file)

    def _create_sample_data(self) -> pd.DataFrame:
        """Create sample data for testing (if real data not available)"""
        sample_texts = [
            "खाना बहुत स्वादिष्ट था और सेवा अच्छी थी।",
            "The food was excellent but service was slow.",
            "बहुत महंगा और खराब quality।",
            "Great ambiance but cold food.",
            "स्वाद शानदार है, फिर आऊंगा।",
            "Tasty food, reasonable price, good staff.",
        ]

        data = pd.DataFrame(
            {
                "text": sample_texts,
                "id": range(len(sample_texts)),
            }
        )
        return data

    def build_features(self) -> Dict[str, List[str]]:
        """Build features: clean, convert to PyABSA format, split data"""
        # Load data
        df = self.load_processed_data()
        logger.info(f"Loaded {len(df)} records")

        # Clean text
        logger.info("Cleaning text...")
        df["cleaned_text"] = df["text"].apply(self.cleaner.clean)

        # Remove empty texts after cleaning
        df = df[df["cleaned_text"].str.len() > 0]
        logger.info(f"After cleaning: {len(df)} records")

        # Convert to PyABSA format
        logger.info("Converting to PyABSA format...")
        pyabsa_data = []
        for cleaned_text in df["cleaned_text"]:
            triplets = self.converter.convert_to_pyabsa_format(cleaned_text)
            pyabsa_data.extend(triplets)

        logger.info(f"Created {len(pyabsa_data)} aspect-sentiment triplets")

        # Split data
        logger.info(f"Splitting data: {self.train_size*100:.0f}% train, {self.val_size*100:.0f}% val, {self.test_size*100:.0f}% test")
        train_data, val_data, test_data = self.splitter.split_data(pyabsa_data)

        logger.info(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

        return {
            "train": train_data,
            "val": val_data,
            "test": test_data,
        }

    def save_features(self, feature_dict: Dict[str, List[str]]) -> None:
        """Save features to files in PyABSA format"""
        # Save individual splits
        for split_name, data in feature_dict.items():
            output_file = self.output_path / f"{split_name}_data.txt"

            with open(output_file, "w", encoding="utf-8") as f:
                for line in data:
                    f.write(line + "\n")

            logger.info(f"Saved {split_name} data to {output_file} ({len(data)} lines)")

        # Save metadata
        metadata = {
            "features_type": feature_type,
            "tokenizer": tokenizer,
            "max_length": max_length,
            "splits": {
                "train": len(feature_dict["train"]),
                "val": len(feature_dict["val"]),
                "test": len(feature_dict["test"]),
            },
            "total_samples": sum(len(v) for v in feature_dict.values()),
        }

        metadata_file = self.output_path / "features_metadata.json"
        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved metadata to {metadata_file}")

    def run(self) -> None:
        """Execute full feature building pipeline"""
        logger.info("Starting feature building pipeline...")

        try:
            # Build features
            feature_dict = self.build_features()

            # Save features
            self.save_features(feature_dict)

            logger.info("✓ Feature building completed successfully")

        except Exception as e:
            logger.error(f"✗ Feature building failed: {str(e)}")
            raise


def main():
    """Main entry point"""
    # Create builder instance
    builder = FeatureBuilder(
        input_path="data/processed",
        output_path="data/processed",
        test_size=test_size,
        val_size=val_size,
    )

    # Run pipeline
    builder.run()

    logger.info("\n" + "="*60)
    logger.info("Feature Engineering Summary:")
    logger.info("="*60)
    logger.info(f"Input: data/processed/processed_data.csv")
    logger.info(f"Output: data/processed/{{train,val,test}}_data.txt (PyABSA format)")
    logger.info(f"Split Ratio: 80% train, 10% val, 10% test")
    logger.info(f"Text Cleaning: Devanagari + Roman Hindi support")
    logger.info(f"Format: text$$$[aspect]$$$sentiment")
    logger.info("="*60)


if __name__ == "__main__":
    # Variables for DVC parametrization (can be edited from dvc.yaml)
    test_size = 0.1
    val_size = 0.1

    main()

