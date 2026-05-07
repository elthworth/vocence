#!/bin/bash
# Quick Start Script for Fine-Tuning Qwen3-TTS Vocence Model

set -e  # Exit on error

echo "================================================================================"
echo "QWEN3-TTS VOCENCE MODEL FINE-TUNING - QUICK START"
echo "================================================================================"
echo ""

# Check if venv exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found. Please create one first:"
    echo "   python -m venv venv"
    exit 1
fi

# Activate venv
echo "✓ Activating virtual environment..."
source venv/bin/activate

# Check required packages
echo "✓ Checking dependencies..."
pip install -q --upgrade pip

REQUIRED_PACKAGES="torch transformers datasets peft accelerate torchaudio soundfile librosa"
echo "  Installing required packages: $REQUIRED_PACKAGES"
pip install -q $REQUIRED_PACKAGES

echo ""
echo "================================================================================"
echo "STEP 1: DOWNLOAD MODELS"
echo "================================================================================"
echo ""

if [ ! -d "downloaded_models/top_model" ]; then
    echo "Downloading models (this will take ~10-15 minutes)..."
    python download_and_inspect_models.py
else
    echo "✓ Models already downloaded, skipping..."
fi

echo ""
echo "================================================================================"
echo "STEP 2: ANALYZE DATASETS"
echo "================================================================================"
echo ""

echo "Running dataset analysis..."
python dataset_analysis.py | head -100
echo ""
echo "See full analysis in dataset_analysis.py output above"
echo "Recommendation: Start with Expresso dataset (VERY HIGH priority)"

echo ""
echo "================================================================================"
echo "STEP 3: TEST BASELINE MODEL"
echo "================================================================================"
echo ""

if [ ! -d "test_outputs_baseline" ]; then
    echo "Testing baseline model performance..."
    read -p "Run baseline test? This will generate 8 audio samples. (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        python test_model_inference.py \
            --model_path downloaded_models/v8_model \
            --output_dir test_outputs_baseline \
            --num_tests 8
        echo ""
        echo "✓ Baseline test complete. Audio files saved to test_outputs_baseline/"
        echo "  Listen to these files to establish baseline quality"
    fi
else
    echo "✓ Baseline test already performed, skipping..."
fi

echo ""
echo "================================================================================"
echo "STEP 4: PREPARE FOR FINE-TUNING"
echo "================================================================================"
echo ""

echo "Before fine-tuning, you need to:"
echo ""
echo "1. Review the fine-tuning guide:"
echo "   cat FINETUNING_GUIDE.md"
echo ""
echo "2. Implement the training loop in finetune_lora.py"
echo "   - The script is a template that needs Qwen3-TTS specific code"
echo "   - See 'Implementation Notes' section in FINETUNING_GUIDE.md"
echo ""
echo "3. Decide on training strategy:"
echo "   a) Phase 1 only: Expresso (recommended to start)"
echo "   b) Phase 1-2-3: Expresso → LibriTTS-R → EmoV-DB"
echo "   c) Mixed dataset: All three combined"
echo ""

read -p "Have you reviewed the guide and are ready to proceed? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo "No problem! Next steps:"
    echo "  1. Read FINETUNING_GUIDE.md thoroughly"
    echo "  2. Implement training loop in finetune_lora.py"
    echo "  3. Run this script again"
    exit 0
fi

echo ""
echo "================================================================================"
echo "STEP 5: RUN FINE-TUNING (PHASE 1)"
echo "================================================================================"
echo ""

echo "Fine-tuning configuration:"
echo "  Base model: downloaded_models/v8_model (magma90909/vocence_miner_v8)"
echo "  Dataset: Expresso"
echo "  Output: finetuned_models/phase1_expresso"
echo "  Epochs: 3"
echo "  Batch size: 4"
echo "  Learning rate: 1e-4"
echo "  LoRA rank: 8"
echo ""

read -p "Start fine-tuning? This will take 12-18 hours on A100. (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo "Starting fine-tuning..."
    python finetune_lora.py \
        --base_model_path downloaded_models/v8_model \
        --dataset expresso \
        --output_dir finetuned_models/phase1_expresso \
        --num_epochs 3 \
        --batch_size 4 \
        --learning_rate 1e-4 \
        --lora_r 8 \
        --lora_alpha 16 \
        --lora_dropout 0.05 \
        --gradient_accumulation_steps 2 \
        --save_steps 500
    
    echo ""
    echo "✓ Fine-tuning complete!"
else
    echo ""
    echo "Fine-tuning skipped. You can run it manually later with:"
    echo "  python finetune_lora.py --base_model_path downloaded_models/top_model --dataset expresso ..."
    exit 0
fi

echo ""
echo "================================================================================"
echo "STEP 6: MERGE LORA WEIGHTS"
echo "================================================================================"
echo ""

if [ -d "finetuned_models/phase1_expresso/final_model" ]; then
    echo "Merging LoRA adapter weights into base model..."
    python merge_lora.py \
        --base_model downloaded_models/v8_model \
        --adapter finetuned_models/phase1_expresso/final_model \
        --output finetuned_models/phase1_expresso/merged_model
    
    echo ""
    echo "✓ Weights merged successfully!"
else
    echo "❌ No trained model found. Fine-tuning may have failed."
    exit 1
fi

echo ""
echo "================================================================================"
echo "STEP 7: TEST FINE-TUNED MODEL"
echo "================================================================================"
echo ""

echo "Testing fine-tuned model..."
python test_model_inference.py \
    --model_path finetuned_models/phase1_expresso/merged_model \
    --output_dir test_outputs_finetuned \
    --num_tests 8

echo ""
echo "✓ Fine-tuned model tested!"
echo ""
echo "Compare audio files:"
echo "  Baseline:   test_outputs_baseline/"
echo "  Fine-tuned: test_outputs_finetuned/"

echo ""
echo "================================================================================"
echo "COMPLETE!"
echo "================================================================================"
echo ""
echo "Summary:"
echo "  ✓ Models downloaded"
echo "  ✓ Datasets analyzed"
echo "  ✓ Baseline tested"
echo "  ✓ Model fine-tuned"
echo "  ✓ Weights merged"
echo "  ✓ Fine-tuned model tested"
echo ""
echo "Next steps:"
echo "  1. Listen to test_outputs_baseline/ vs test_outputs_finetuned/"
echo "  2. If improvements are good, deploy to HuggingFace and Vocence"
echo "  3. If not satisfied, try Phase 2 or adjust hyperparameters"
echo ""
echo "Deployment:"
echo "  1. Upload to HuggingFace:"
echo "     python -m huggingface_hub upload finetuned_models/phase1_expresso/merged_model your-username/vocence-improved"
echo "  2. Update your miner repository to use the new model"
echo ""
echo "For more details, see:"
echo "  - FINETUNING_GUIDE.md (comprehensive guide)"
echo "  - dataset_analysis.py (dataset comparison)"
echo ""

deactivate
