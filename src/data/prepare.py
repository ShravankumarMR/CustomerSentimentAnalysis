"""
Data Preparation Module for Customer Sentiment Analysis
Loads raw data and prepares it for feature engineering
"""

import logging
from pathlib import Path

import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class DataPreparator:
    """Handles data loading and basic preparation"""

    def __init__(self, input_path: str = "data/raw", output_path: str = "data/processed"):
        """Initialize data preparator"""
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)

        # Create output directory
        self.output_path.mkdir(parents=True, exist_ok=True)

    def load_raw_data(self) -> pd.DataFrame:
        """Load raw CSV file"""
        csv_file = self.input_path / "restaurant.csv"

        if not csv_file.exists():
            logger.warning(f"Raw data file not found: {csv_file}")
            logger.info("Creating sample data for demonstration")
            return self._create_sample_data()

        logger.info(f"Loading raw data from {csv_file}")
        try:
            df = pd.read_csv(csv_file)
            logger.info(f"Loaded {len(df)} records from {csv_file}")
            return df
        except Exception as e:
            logger.error(f"Error loading CSV: {str(e)}")
            raise

    def _create_sample_data(self) -> pd.DataFrame:
        """Create sample data if real data not available"""
        sample_data = {
            "text": [
                "खाना बहुत स्वादिष्ट था और सेवा अच्छी थी।",
                "The food was excellent but service was slow.",
                "बहुत महंगा और खराब quality।",
                "Great ambiance but cold food.",
                "स्वाद शानदार है, फिर आऊंगा।",
                "Tasty food, reasonable price, good staff.",
                "Restaurant bahut sundar hai aur khaana delicious hai.",
                "बहुत अच्छा restaurant है, सब कुछ परफेक्ट है।",
            ]
        }

        df = pd.DataFrame(sample_data)
        df["id"] = range(len(df))
        logger.info(f"Created sample data with {len(df)} records")
        return df

    def prepare(self) -> pd.DataFrame:
        """Main preparation pipeline"""
        logger.info("Starting data preparation pipeline...")

        # Load raw data
        df = self.load_raw_data()

        # Basic data quality checks
        logger.info("Performing data quality checks...")

        # Map 'sentence' to 'text' if needed
        if "sentence" in df.columns and "text" not in df.columns:
            df = df.rename(columns={"sentence": "text"})
            logger.info("Mapped 'sentence' column to 'text'")

        # Check for missing values in text column
        if "text" in df.columns:
            missing_count = df["text"].isna().sum()
            if missing_count > 0:
                logger.warning(f"Found {missing_count} missing text values")
                df = df.dropna(subset=["text"])

            # Remove empty strings
            df = df[df["text"].str.strip().str.len() > 0]

        logger.info(f"After data quality checks: {len(df)} records")

        # Ensure id column exists
        if "id" not in df.columns:
            df["id"] = range(len(df))

        # Keep text and id columns
        if "text" in df.columns:
            df = df[["id", "text"]]
            # Add aspect and sentiment info if available
            if "aspect_terms" in df.columns or "at_polarity" in df.columns:
                logger.info("Aspect and polarity information detected in raw data")
        else:
            raise ValueError("'text' or 'sentence' column not found in data")

        logger.info(f"✓ Data preparation completed: {len(df)} records ready")

        return df

    def save_prepared_data(self, df: pd.DataFrame) -> None:
        """Save prepared data to CSV"""
        output_file = self.output_path / "processed_data.csv"

        try:
            df.to_csv(output_file, index=False, encoding="utf-8")
            logger.info(f"Saved prepared data to {output_file}")
        except Exception as e:
            logger.error(f"Error saving data: {str(e)}")
            raise

    def run(self) -> None:
        """Execute full data preparation pipeline"""
        try:
            df = self.prepare()
            self.save_prepared_data(df)
            logger.info("="*60)
            logger.info("Data Preparation Summary:")
            logger.info("="*60)
            logger.info(f"Input: data/raw/restaurant.csv")
            logger.info(f"Output: data/processed/processed_data.csv")
            logger.info(f"Total records: {len(df)}")
            logger.info("="*60)
        except Exception as e:
            logger.error(f"Data preparation failed: {str(e)}")
            raise


def main():
    """Main entry point"""
    preparator = DataPreparator(
        input_path="data/raw",
        output_path="data/processed",
    )
    preparator.run()


if __name__ == "__main__":
    main()

