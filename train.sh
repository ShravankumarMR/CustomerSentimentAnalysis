#!/bin/bash
# ABSA Training Helper Script - Common Presets
# 
# Usage:
#   bash train.sh quick          # Quick test (1 epoch)
#   bash train.sh default        # Default settings
#   bash train.sh production     # Production quality
#   bash train.sh lowmem         # Low memory usage
#   bash train.sh help          # Show help

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

show_help() {
    cat << EOF
${BLUE}ABSA Training Helper${NC}

Convenient presets for training ABSA models with different configurations.

${GREEN}Usage:${NC}
    bash train.sh [PRESET]

${GREEN}Available Presets:${NC}

    quick           - Quick test run (1 epoch)
                      Batch: 8 | Learning Rate: 2e-5
                      Use for: Testing pipeline

    default         - Standard training (5 epochs)
                      Batch: 32 | Learning Rate: 2e-5
                      Use for: Normal training

    production      - High quality training (10 epochs)
                      Batch: 64 | Learning Rate: 1e-5
                      Use for: Production deployment

    lowmem          - Low memory usage (3 epochs)
                      Batch: 8 | Learning Rate: 2e-5
                      Use for: Low VRAM/CPU training

    custom          - Custom parameters (prompts for input)
                      Use for: Fine-grained control

    both            - Train both ASC and ATE (5 epochs each)
                      Batch: 32
                      Use for: Complete ABSA pipeline

    verify          - Verify setup and dependencies
                      Use for: Troubleshooting

    demo            - Run complete demo (quickstart)
                      Includes: verify → dry-run → quick train

    help            - Show this help message

${GREEN}Examples:${NC}

    # Quick test
    bash train.sh quick

    # Production training
    bash train.sh production

    # Complete demo
    bash train.sh demo

    # Custom parameters
    bash train.sh custom

${GREEN}Additional Options:${NC}

    After running with preset, models will be saved to 'models/' directory
    Training runs tracked in MLflow (view with: mlflow ui)

${GREEN}Tips:${NC}

    • Start with 'quick' to test your setup
    • Use 'default' for normal training
    • Use 'production' for best accuracy
    • Use 'lowmem' if you get GPU memory errors
    • Monitor with 'mlflow ui' in another terminal

EOF
}

quick_train() {
    echo -e "${BLUE}Running QUICK TRAIN (1 epoch test)${NC}\n"
    python run_training.py \
        --task asc \
        --epochs 1 \
        --batch-size 8 \
        --learning-rate 2e-5
    echo -e "\n${GREEN}✓ Quick training completed!${NC}"
}

default_train() {
    echo -e "${BLUE}Running DEFAULT TRAIN (5 epochs)${NC}\n"
    python run_training.py \
        --task asc \
        --epochs 5 \
        --batch-size 32 \
        --learning-rate 2e-5
    echo -e "\n${GREEN}✓ Default training completed!${NC}"
}

production_train() {
    echo -e "${BLUE}Running PRODUCTION TRAIN (10 epochs, high quality)${NC}\n"
    echo -e "${YELLOW}Note: This may take 30+ minutes${NC}\n"
    python run_training.py \
        --task asc \
        --epochs 10 \
        --batch-size 64 \
        --learning-rate 1e-5
    echo -e "\n${GREEN}✓ Production training completed!${NC}"
}

lowmem_train() {
    echo -e "${BLUE}Running LOW MEMORY TRAIN (3 epochs, batch 8)${NC}\n"
    python run_training.py \
        --task asc \
        --epochs 3 \
        --batch-size 8 \
        --learning-rate 2e-5
    echo -e "\n${GREEN}✓ Low memory training completed!${NC}"
}

custom_train() {
    echo -e "${BLUE}Custom Training Configuration${NC}\n"
    
    read -p "Task (asc/ate/both) [asc]: " task
    task=${task:-asc}
    
    read -p "Epochs [5]: " epochs
    epochs=${epochs:-5}
    
    read -p "Batch size [32]: " batch_size
    batch_size=${batch_size:-32}
    
    read -p "Learning rate [2e-5]: " learning_rate
    learning_rate=${learning_rate:-2e-5}
    
    echo -e "\n${BLUE}Running with:${NC}"
    echo "  Task: $task"
    echo "  Epochs: $epochs"
    echo "  Batch Size: $batch_size"
    echo "  Learning Rate: $learning_rate"
    echo ""
    
    python run_training.py \
        --task "$task" \
        --epochs "$epochs" \
        --batch-size "$batch_size" \
        --learning-rate "$learning_rate"
    
    echo -e "\n${GREEN}✓ Custom training completed!${NC}"
}

both_train() {
    echo -e "${BLUE}Running BOTH ASC and ATE (5 epochs each)${NC}\n"
    python run_training.py \
        --task both \
        --epochs 5 \
        --batch-size 32 \
        --learning-rate 2e-5
    echo -e "\n${GREEN}✓ Both models trained successfully!${NC}"
}

verify_setup() {
    echo -e "${BLUE}Verifying Setup and Dependencies${NC}\n"
    python verify_setup.py
}

demo() {
    echo -e "${BLUE}Running Complete Demo${NC}\n"
    
    echo -e "${YELLOW}Step 1: Verifying setup...${NC}"
    python verify_setup.py
    
    echo -e "\n${YELLOW}Step 2: Previewing configuration (dry-run)...${NC}"
    python run_training.py --task asc --dry-run
    
    echo -e "\n${YELLOW}Step 3: Running quick training (1 epoch)...${NC}"
    python run_training.py \
        --task asc \
        --epochs 1 \
        --batch-size 8
    
    echo -e "\n${GREEN}✓ Demo completed!${NC}"
    echo -e "${BLUE}Next steps:${NC}"
    echo "  1. View results: cat models/training_summary.json"
    echo "  2. Monitor: mlflow ui (http://localhost:5000)"
    echo "  3. Make predictions: python src/models/predict.py --model models/asc_model/best_model --interactive"
}

main() {
    local preset="${1:-help}"
    
    # Check if Python is available
    if ! command -v python &> /dev/null; then
        echo -e "${RED}Error: Python not found. Please install Python 3.8+${NC}"
        exit 1
    fi
    
    case "$preset" in
        quick)
            quick_train
            ;;
        default)
            default_train
            ;;
        production)
            production_train
            ;;
        lowmem)
            lowmem_train
            ;;
        custom)
            custom_train
            ;;
        both)
            both_train
            ;;
        verify)
            verify_setup
            ;;
        demo)
            demo
            ;;
        help|--help|-h)
            show_help
            ;;
        *)
            echo -e "${YELLOW}Unknown preset: $preset${NC}\n"
            show_help
            exit 1
            ;;
    esac
}

# Run main function
main "$@"
