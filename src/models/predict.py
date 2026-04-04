"""
Inference Script for Trained ABSA Models.

Use trained ABSA models to make predictions on new text samples.
Supports both ASC (Aspect-Sentiment Classification) and ATE (Aspect Term Extraction).

Usage:
    # Single prediction
    python src/models/predict.py --model models/asc_model/best_model --text "Great food" --aspect food

    # Batch predictions from file
    python src/models/predict.py --model models/asc_model/best_model --input-file predictions.txt

    # Interactive mode
    python src/models/predict.py --model models/asc_model/best_model --interactive
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    from pyabsa import AspectExtraction, AspectSentimentClassification
except ImportError:
    print("PyABSA not installed. Install with: pip install pyabsa")
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class ABSAPredictor:
    """Make predictions with trained ABSA models."""

    def __init__(self, model_path: str, task_name: str = "asc", device: str = "cuda"):
        """
        Initialize predictor with trained model.

        Args:
            model_path: Path to trained model
            task_name: "asc" or "ate"
            device: "cuda" or "cpu"
        """
        self.model_path = model_path
        self.task_name = task_name
        self.device = device

        logger.info(f"Loading model from {model_path}")
        logger.info(f"Task: {task_name}, Device: {device}")

        try:
            if task_name == "asc":
                self.model = AspectSentimentClassification(
                    model_path, task_name="asc", device=device
                )
            elif task_name == "ate":
                self.model = AspectExtraction(
                    model_path, task_name="ate", device=device
                )
            else:
                raise ValueError(f"Unknown task: {task_name}")

            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def predict_asc(self, text: str, aspects: Optional[List[str]] = None) -> Dict:
        """
        Predict aspects sentiment (ASC).

        Args:
            text: Review text
            aspects: Optional list of aspects (e.g., ["food", "service", "ambiance"])
                    If None, will extract all aspects

        Returns:
            Prediction results with sentiments
        """
        try:
            # Format: text {aspect} label
            if aspects:
                # Predict for specific aspects
                predictions = {}
                for aspect in aspects:
                    result = self.model.predict(text, [aspect])
                    predictions[aspect] = result
                return predictions
            else:
                # Try to predict for common aspects
                common_aspects = [
                    "food",
                    "service",
                    "ambiance",
                    "price",
                    "staff",
                    "cleanliness",
                ]
                predictions = {}
                for aspect in common_aspects:
                    try:
                        result = self.model.predict(text, [aspect])
                        predictions[aspect] = result
                    except:
                        pass

                return predictions if predictions else {"error": "No aspects detected"}
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return {"error": str(e)}

    def predict_ate(self, text: str) -> Dict:
        """
        Predict aspects (ATE - Aspect Term Extraction).

        Args:
            text: Review text

        Returns:
            Detected aspects with positions
        """
        try:
            result = self.model.predict(text)
            return result
        except Exception as e:
            logger.error(f"Aspect extraction failed: {e}")
            return {"error": str(e)}

    def predict_batch(
        self, texts: List[str], aspects: Optional[List[str]] = None
    ) -> List[Dict]:
        """
        Make predictions on batch of texts.

        Args:
            texts: List of review texts
            aspects: Optional list of aspects (for ASC only)

        Returns:
            List of predictions
        """
        results = []
        for i, text in enumerate(texts, 1):
            logger.info(f"Processing {i}/{len(texts)}")
            if self.task_name == "asc":
                result = self.predict_asc(text, aspects)
            else:
                result = self.predict_ate(text)
            results.append({"text": text, "prediction": result})

        return results


def format_output(result: Dict, task_name: str = "asc") -> str:
    """Format prediction result for display."""
    if "error" in result:
        return f"❌ Error: {result['error']}"

    output_lines = []

    if task_name == "asc":
        output_lines.append("📊 Aspect Sentiment Predictions:")
        for aspect, prediction in result.items():
            if isinstance(prediction, dict):
                sentiment = prediction.get("sentiment", "unknown")
                score = prediction.get("confidence", prediction.get("score", "N/A"))
                output_lines.append(f"  • {aspect}: {sentiment} (confidence: {score})")
            else:
                output_lines.append(f"  • {aspect}: {prediction}")
    else:
        output_lines.append("🏷️  Detected Aspects:")
        for key, value in result.items():
            output_lines.append(f"  • {key}: {value}")

    return "\n".join(output_lines)


def interactive_mode(predictor: ABSAPredictor):
    """Interactive prediction mode."""
    logger.info("Entering interactive mode. Type 'quit' to exit.")
    logger.info("=" * 60)

    if predictor.task_name == "asc":
        logger.info("Enter text and aspects separated by '|'")
        logger.info("Example: Great food | food, service")
    else:
        logger.info("Enter review text to extract aspects")

    while True:
        try:
            user_input = input("\n> ").strip()

            if user_input.lower() in ["quit", "exit", "q"]:
                logger.info("Exiting interactive mode")
                break

            if not user_input:
                continue

            if predictor.task_name == "asc":
                parts = user_input.split("|")
                text = parts[0].strip()
                aspects = (
                    [a.strip() for a in parts[1].split(",")]
                    if len(parts) > 1
                    else None
                )
                result = predictor.predict_asc(text, aspects)
            else:
                result = predictor.predict_ate(user_input)

            print("\n" + format_output(result, predictor.task_name))

        except KeyboardInterrupt:
            logger.info("\nExiting interactive mode")
            break
        except Exception as e:
            logger.error(f"Error: {e}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Make predictions with trained ABSA models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:

  # ASC: Predict sentiment for aspects
  python src/models/predict.py \\
    --model models/asc_model/best_model \\
    --task asc \\
    --text "Great food but slow service" \\
    --aspects food service

  # ATE: Extract aspects from text
  python src/models/predict.py \\
    --model models/ate_model/best_model \\
    --task ate \\
    --text "The food was great, service was slow"

  # Batch predictions from file
  python src/models/predict.py \\
    --model models/asc_model/best_model \\
    --task asc \\
    --input-file reviews.txt \\
    --output-file predictions.json

  # Interactive mode
  python src/models/predict.py \\
    --model models/asc_model/best_model \\
    --task asc \\
    --interactive

  # Use CPU instead of GPU
  python src/models/predict.py \\
    --model models/asc_model/best_model \\
    --task asc \\
    --text "Great food" \\
    --device cpu
        """,
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to trained model directory",
    )

    parser.add_argument(
        "--task",
        type=str,
        choices=["asc", "ate"],
        default="asc",
        help="Task type: asc=Aspect-Sentiment Classification (default), ate=Aspect-Term Extraction",
    )

    parser.add_argument(
        "--text",
        type=str,
        help="Single text for prediction",
    )

    parser.add_argument(
        "--aspects",
        type=str,
        nargs="+",
        help="Aspect names to classify (for ASC task only)",
    )

    parser.add_argument(
        "--input-file",
        type=str,
        help="Input file with texts (one per line)",
    )

    parser.add_argument(
        "--output-file",
        type=str,
        help="Output file for batch predictions (JSON format)",
    )

    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Interactive mode for manual predictions",
    )

    parser.add_argument(
        "--device",
        type=str,
        choices=["cuda", "cpu"],
        default="cuda",
        help="Device for inference (default: cuda)",
    )

    args = parser.parse_args()

    # Initialize predictor
    try:
        predictor = ABSAPredictor(
            model_path=args.model, task_name=args.task, device=args.device
        )
    except Exception as e:
        logger.error(f"Failed to initialize predictor: {e}")
        sys.exit(1)

    # Single prediction
    if args.text:
        if args.task == "asc":
            result = predictor.predict_asc(args.text, args.aspects)
        else:
            result = predictor.predict_ate(args.text)

        print(f"\n📝 Text: {args.text}")
        print(format_output(result, args.task))

    # Batch prediction from file
    elif args.input_file:
        if not Path(args.input_file).exists():
            logger.error(f"Input file not found: {args.input_file}")
            sys.exit(1)

        logger.info(f"Reading texts from {args.input_file}")
        with open(args.input_file, "r", encoding="utf-8") as f:
            texts = [line.strip() for line in f if line.strip()]

        logger.info(f"Making predictions on {len(texts)} samples")
        results = predictor.predict_batch(texts, args.aspects)

        if args.output_file:
            with open(args.output_file, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            logger.info(f"Predictions saved to {args.output_file}")
        else:
            for result in results:
                print(f"\n📝 Text: {result['text']}")
                print(format_output(result["prediction"], args.task))

    # Interactive mode
    elif args.interactive:
        interactive_mode(predictor)

    else:
        # Show help if no action specified
        parser.print_help()
        logger.error("Please provide --text, --input-file, or --interactive")
        sys.exit(1)


if __name__ == "__main__":
    main()
