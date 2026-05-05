#!/bin/bash
#
# Vocence Hybrid Miner - Quick Start Setup Script
# 
# This script automates the local setup process for testing your hybrid miner
# before deploying to Chutes and Bittensor Subnet 78.
#
# Usage: bash quickstart.sh
#

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Functions
print_header() {
    echo ""
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo ""
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

check_cuda() {
    if command -v nvidia-smi &> /dev/null; then
        print_success "NVIDIA GPU detected"
        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
        return 0
    else
        print_error "NVIDIA GPU not detected!"
        print_warning "This miner requires a GPU with 24GB+ VRAM"
        return 1
    fi
}

# Main script
print_header "VOCENCE HYBRID MINER - QUICK START"

echo "This script will:"
echo "  1. Check system requirements"
echo "  2. Create Python virtual environment"
echo "  3. Install dependencies"
echo "  4. Download T2 base model (4.4GB)"
echo "  5. Run local tests"
echo ""
read -p "Continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    exit 0
fi

# Step 1: Check system requirements
print_header "Step 1/5: Checking System Requirements"

# Check GPU
if ! check_cuda; then
    print_error "GPU check failed. Exiting."
    exit 1
fi

# Check VRAM
VRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n1)
if [ "$VRAM" -lt 24000 ]; then
    print_warning "GPU has ${VRAM}MB VRAM, but 24GB+ recommended"
    read -p "Continue anyway? This may fail. (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 0
    fi
else
    print_success "GPU has ${VRAM}MB VRAM (sufficient)"
fi

# Check Python version
if command -v python3.10 &> /dev/null; then
    PYTHON_CMD=python3.10
    print_success "Python 3.10 found"
elif command -v python3.11 &> /dev/null; then
    PYTHON_CMD=python3.11
    print_success "Python 3.11 found"
elif command -v python3 &> /dev/null; then
    PY_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
    if [[ "$PY_VERSION" == "3.10" ]] || [[ "$PY_VERSION" == "3.11" ]]; then
        PYTHON_CMD=python3
        print_success "Python $PY_VERSION found"
    else
        print_error "Python 3.10 or 3.11 required (found $PY_VERSION)"
        exit 1
    fi
else
    print_error "Python 3.10 or 3.11 not found"
    exit 1
fi

# Check disk space
DISK_FREE=$(df -BG . | tail -1 | awk '{print $4}' | sed 's/G//')
if [ "$DISK_FREE" -lt 50 ]; then
    print_warning "Only ${DISK_FREE}GB free disk space (50GB+ recommended)"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 0
    fi
else
    print_success "Sufficient disk space (${DISK_FREE}GB available)"
fi

# Step 2: Create virtual environment
print_header "Step 2/5: Creating Virtual Environment"

if [ -d "venv" ]; then
    print_warning "venv/ already exists, skipping creation"
else
    $PYTHON_CMD -m venv venv
    print_success "Virtual environment created"
fi

source venv/bin/activate
print_success "Virtual environment activated"

# Upgrade pip
pip install --upgrade pip -q
print_success "Pip upgraded"

# Step 3: Install dependencies
print_header "Step 3/5: Installing Dependencies (5-10 minutes)"

echo "Installing PyTorch with CUDA 12.1..."
pip install torch==2.1.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121 -q
print_success "PyTorch installed"

echo "Installing transformers ecosystem..."
pip install transformers==4.40.0 accelerate==0.27.0 huggingface_hub -q
print_success "Transformers installed"

echo "Installing Qwen-TTS..."
pip install qwen-tts -q
print_success "Qwen-TTS installed"

echo "Installing evaluation tools..."
pip install faster-whisper jiwer soundfile -q
print_success "Evaluation tools installed"

echo "Installing UTMOSv2 (may take a few minutes)..."
pip install git+https://github.com/sarulab-speech/UTMOSv2.git -q
print_success "UTMOSv2 installed"

echo "Installing additional utilities..."
pip install pyyaml numpy scipy librosa -q
print_success "Additional utilities installed"

# Verify CUDA
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print(f'PyTorch CUDA: {torch.version.cuda}')" || {
    print_error "PyTorch CUDA check failed!"
    exit 1
}
print_success "PyTorch CUDA verified"

# Step 4: Download T2 model
print_header "Step 4/5: Downloading T2 Base Model (4.4GB, 5-10 minutes)"

if [ -d "model" ] && [ -f "model/config.json" ]; then
    print_warning "model/ directory already exists, skipping download"
else
    python << 'EOF'
from huggingface_hub import snapshot_download
from pathlib import Path
import sys

try:
    print("Downloading macminix/qwen3_voice_design_t2...")
    local_dir = snapshot_download(
        repo_id="macminix/qwen3_voice_design_t2",
        local_dir=Path("./model"),
        ignore_patterns=["*.md", "*.txt", ".gitattributes"]
    )
    print(f"✓ Downloaded to: {local_dir}")
except Exception as e:
    print(f"✗ Download failed: {e}", file=sys.stderr)
    sys.exit(1)
EOF
    
    if [ $? -eq 0 ]; then
        print_success "T2 model downloaded"
    else
        print_error "Model download failed!"
        exit 1
    fi
fi

# Step 5: Run local tests
print_header "Step 5/5: Running Local Tests"

# Check if miner.py exists
if [ ! -f "miner.py" ]; then
    print_error "miner.py not found in current directory!"
    print_warning "Please copy miner.py, vocence_config.yaml to this directory"
    exit 1
fi

print_success "Found miner.py"

# Create test script
cat > test_local_quick.py << 'EOF'
from pathlib import Path
import soundfile as sf
import time
import sys

try:
    from miner import Miner
    
    print("\n" + "=" * 60)
    print("VOCENCE HYBRID MINER - QUICK TEST")
    print("=" * 60)
    
    # Initialize
    print("\n[1/3] Initializing miner...")
    miner = Miner(path_hf_repo=Path("./model"))
    print("✓ Miner initialized")
    
    # Warmup
    print("\n[2/3] Warming up (30-60s)...")
    start = time.time()
    miner.warmup()
    warmup_time = time.time() - start
    print(f"✓ Warmup complete in {warmup_time:.1f}s")
    
    # Single test generation
    print("\n[3/3] Running test generation...")
    instruction = "gender: female | age_group: adult | pitch: mid | speed: normal | emotion: happy | tone: casual | accent: us"
    text = "Hello! This is a quick test of the hybrid miner."
    
    start = time.time()
    wav, sr = miner.generate_wav(instruction=instruction, text=text)
    gen_time = time.time() - start
    
    duration = len(wav) / sr
    sf.write("test_output.wav", wav, sr)
    
    print(f"✓ Generated {duration:.2f}s audio in {gen_time:.1f}s")
    print(f"✓ Saved to: test_output.wav")
    
    # Summary
    print("\n" + "=" * 60)
    print("✓ ALL TESTS PASSED!")
    print("=" * 60)
    print(f"Warmup time: {warmup_time:.1f}s")
    print(f"Generation time: {gen_time:.1f}s")
    
    if gen_time > 130:
        print("\n⚠ WARNING: Generation time > 130s")
        print("Consider reducing num_candidates in vocence_config.yaml")
    else:
        print("\n✓ Generation time meets latency requirements")
    
    print("\nNext steps:")
    print("1. Listen to test_output.wav to verify quality")
    print("2. Review DEPLOYMENT_GUIDE.md for Chutes deployment")
    print("3. Join Vocence Discord: https://discord.gg/vocence")
    print("=" * 60)
    
except Exception as e:
    print(f"\n✗ Test failed: {e}", file=sys.stderr)
    import traceback
    traceback.print_exc()
    sys.exit(1)
EOF

python test_local_quick.py

if [ $? -eq 0 ]; then
    print_success "Local tests completed successfully!"
    
    # Final summary
    print_header "SETUP COMPLETE!"
    
    echo "Your hybrid miner is ready for local testing."
    echo ""
    echo "Generated files:"
    echo "  • test_output.wav - Test audio (listen to verify quality)"
    echo "  • venv/ - Python virtual environment"
    echo "  • model/ - T2 base model (4.4GB)"
    echo ""
    echo "Next steps:"
    echo "  1. Listen to test_output.wav"
    echo "  2. Review DEPLOYMENT_GUIDE.md for deployment instructions"
    echo "  3. Test with more diverse prompts (see README.md)"
    echo ""
    echo "To reactivate environment later:"
    echo "  source venv/bin/activate"
    echo ""
    echo "For deployment help:"
    echo "  cat DEPLOYMENT_GUIDE.md"
    echo ""
    print_success "Happy mining!"
else
    print_error "Tests failed! Check output above for errors."
    exit 1
fi
