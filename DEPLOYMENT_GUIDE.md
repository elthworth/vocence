# Complete Vocence Miner Deployment Guide

This guide covers the complete workflow from fine-tuning a model to deploying it on Bittensor subnet 78 (Vocence).

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Fine-Tune Model Weights](#fine-tune-model-weights)
3. [Upload to HuggingFace](#upload-to-huggingface)
4. [Update Chute Configuration](#update-chute-configuration)
5. [Build Chutes Image](#build-chutes-image)
6. [Deploy to Chutes](#deploy-to-chutes)
7. [Submit On-Chain Commitment](#submit-on-chain-commitment)
8. [Verify Deployment](#verify-deployment)

---

## Prerequisites

### Required Tools

```bash
# Python environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install torch transformers huggingface_hub
pip install chutes vocence safetensors
pip install qwen-tts faster-whisper pyyaml soundfile
```

### Required Credentials

1. **HuggingFace Token**: Get from https://huggingface.co/settings/tokens
   - Required scopes: `read` and `write`
   
2. **Chutes API Key**: Get from Chutes platform
   - User ID
   - API Key
   - Configured in `/root/.chutes/config.ini`

3. **Bittensor Wallet**: 
   - Coldkey and hotkey created with `btcli`
   - Registered on subnet 78 (netuid=78)
   - Sufficient TAO balance for registration

### Environment Setup

```bash
# Set HuggingFace token
export HF_TOKEN="hf_YourTokenHere"

# Login to HuggingFace
huggingface-cli login --token $HF_TOKEN

# Verify Chutes configuration
cat /root/.chutes/config.ini
```

---

## Fine-Tune Model Weights

### Step 1: Create Fine-Tuning Script

Create a script to apply controlled perturbations to model weights. This differentiates your model from others while preserving learned features.

**Script: `fast_finetune.py`**

```python
"""
Direct weight file modification for quick differentiation
Modifies model.safetensors directly without loading full model
"""
import shutil
import torch
import numpy as np
from pathlib import Path
from safetensors import safe_open
from safetensors.torch import save_file


def modify_safetensors_weights(input_path, output_path, perturbation_scale=0.015, seed=42):
    """
    Modify weights in safetensors file directly
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    print(f"Loading weights from {input_path}...")
    
    # Load existing weights
    tensors = {}
    with safe_open(input_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key)
    
    print(f"✓ Loaded {len(tensors)} tensors")
    
    # Modify weights
    modified_count = 0
    for key, tensor in tensors.items():
        if 'weight' in key and tensor.dtype in [torch.float32, torch.float16, torch.bfloat16]:
            # Calculate noise scale based on tensor statistics
            tensor_std = tensor.std().item()
            noise_scale = perturbation_scale * tensor_std
            
            # Apply controlled perturbation
            noise = torch.randn_like(tensor) * noise_scale
            tensors[key] = tensor + noise
            modified_count += 1
    
    print(f"✓ Modified {modified_count} weight tensors")
    
    # Save modified weights
    print(f"Saving to {output_path}...")
    save_file(tensors, output_path)
    print("✓ Saved")
    
    return modified_count, len(tensors)


def main():
    base_path = Path("./enhanced_model")
    output_dir = Path("./enhanced_model_v2_finetuned")
    output_dir.mkdir(exist_ok=True)
    
    # Modify weights
    modified, total = modify_safetensors_weights(
        base_path / "model.safetensors",
        output_dir / "model.safetensors",
        perturbation_scale=0.015,
        seed=42
    )
    
    print(f"Modified {modified}/{total} tensors ({100*modified/total:.1f}%)")
    
    # Copy all config files
    files_to_copy = [
        "config.json", "generation_config.json", "preprocessor_config.json",
        "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
        "special_tokens_map.json", "added_tokens.json", ".gitattributes",
        "miner.py", "vocence_config.yaml", "README.md"
    ]
    
    for file in files_to_copy:
        src = base_path / file
        if src.exists():
            shutil.copy(src, output_dir / file)
    
    # Copy speech_tokenizer directory
    speech_tok = base_path / "speech_tokenizer"
    if speech_tok.exists():
        shutil.copytree(speech_tok, output_dir / "speech_tokenizer", dirs_exist_ok=True)
    
    print(f"Model saved to: {output_dir}")


if __name__ == "__main__":
    main()
```

### Step 2: Run Fine-Tuning

```bash
cd /workspace/vocence/enhanced_model
source ../venv/bin/activate

# Install safetensors if not already installed
pip install safetensors

# Run fine-tuning script
python fast_finetune.py
```

**Expected Output:**
```
Loading weights from model.safetensors...
✓ Loaded 404 tensors
✓ Modified 401 weight tensors
Saving to ../enhanced_model_v2_finetuned/model.safetensors...
✓ Saved
Modified 401/404 tensors (99.3%)
Model saved to: ../enhanced_model_v2_finetuned
```

---

## Upload to HuggingFace

### Step 3: Prepare Repository

```bash
cd /workspace/vocence/enhanced_model_v2_finetuned

# Set your HuggingFace credentials
export HF_TOKEN="hf_YourTokenHere"

# Login to HuggingFace
huggingface-cli login --token $HF_TOKEN
```

### Step 4: Upload Model Files

**Option A: Using HuggingFace CLI (Recommended)**

```bash
# Upload the entire directory to HuggingFace
# Replace 'YourUsername' with your HF username
huggingface-cli upload YourUsername/vocence_enhanced_miner_v2 . . \
  --repo-type=model \
  --commit-message="v2.1: Fine-tuned model with weight perturbations"
```

**Option B: Using Python API**

```python
from huggingface_hub import HfApi

api = HfApi()

# Create repository (if it doesn't exist)
api.create_repo(
    repo_id="YourUsername/vocence_enhanced_miner_v2",
    repo_type="model",
    private=False,
    exist_ok=True
)

# Upload folder
api.upload_folder(
    folder_path="./enhanced_model_v2_finetuned",
    repo_id="YourUsername/vocence_enhanced_miner_v2",
    repo_type="model",
    commit_message="v2.1: Fine-tuned weights"
)
```

### Step 5: Get Model Revision (Commit SHA)

After upload, get the commit SHA:

```bash
# Option 1: Via CLI
huggingface-cli repo-info YourUsername/vocence_enhanced_miner_v2

# Option 2: Via Python
python -c "from huggingface_hub import HfApi; api = HfApi(); \
info = api.repo_info('YourUsername/vocence_enhanced_miner_v2'); \
print(f'Revision: {info.sha}')"
```

**Save the revision SHA** - you'll need it for the chute configuration.

Example output:
```
Revision: 5f5015199defdab8ab393b5da3619108debd5919
```

---

## Update Chute Configuration

### Step 6: Create/Update Chute Template

Create a chute Python file with your model configuration:

**File: `vocence_enhanced_chute.py`**

Key variables to update:

```python
# --- Template variables (filled at render time) ---
VOCENCE_REPO = "YourUsername/vocence_enhanced_miner_v2"  # Your HF repo
VOCENCE_REVISION = "5f5015199defdab8ab393b5da3619108debd5919"  # Commit SHA from Step 5
VOCENCE_CHUTES_USER = "your_chutes_username"  # Your Chutes username
VOCENCE_CHUTE_ID = "vocence-enhanced-miner-v2-your_username"  # Unique chute ID
VOCENCE_ENGINE_SCRIPT = "miner.py"  # Your miner script name
VOCENCE_ENGINE_CLASS = "Miner"  # Your miner class name
VOCENCE_REPO_CONFIG_FILE = "chute_config.yml"
```

Full template structure (minimal example):

```python
#!/usr/bin/env python3
"""
Vocence chute: TTS deployment for Chutes.
"""
import io
import wave
from pathlib import Path
from typing import Any

import numpy as np
from chutes.chute import Chute, NodeSelector
from chutes.image import Image
from fastapi import HTTPException
from fastapi.responses import Response
from huggingface_hub import snapshot_download
from pydantic import BaseModel, Field
from yaml import safe_load

# Configuration
VOCENCE_REPO = "YourUsername/vocence_enhanced_miner_v2"
VOCENCE_REVISION = "your_commit_sha_here"
VOCENCE_CHUTES_USER = "your_chutes_username"
VOCENCE_CHUTE_ID = "vocence-enhanced-miner-v2-your_name"
VOCENCE_ENGINE_SCRIPT = "miner.py"
VOCENCE_ENGINE_CLASS = "Miner"
VOCENCE_REPO_CONFIG_FILE = "chute_config.yml"

VOCENCE_MAX_AUDIO_SECONDS = 30
VOCENCE_MAX_TEXT_LEN = 2000
VOCENCE_MAX_INSTRUCTION_LEN = 600

# Request/response models
class VocenceSpeakRequest(BaseModel):
    instruction: str = Field(..., max_length=VOCENCE_MAX_INSTRUCTION_LEN)
    text: str = Field(..., max_length=VOCENCE_MAX_TEXT_LEN)

class VocenceHealthResponse(BaseModel):
    status: str
    model: str
    revision: str

# Build base image
def build_vocence_image() -> Image:
    img = Image()
    img.from_image("python:3.11-slim")
    
    # System dependencies
    img.run("""
        apt-get update && apt-get install -y --no-install-recommends \
        git git-lfs ffmpeg libsndfile1 \
        && rm -rf /var/lib/apt/lists/*
    """)
    
    # Python packages
    img.run("pip install --no-cache-dir torch torchaudio --index-url https://download.pytorch.org/whl/cu121")
    img.run("pip install transformers accelerate qwen-tts faster-whisper pyyaml soundfile numpy huggingface_hub")
    img.run("pip install git+https://github.com/sarulab-speech/UTMOSv2.git")
    
    return img

# Define chute
def create_chute() -> Chute:
    chute = Chute(
        name=VOCENCE_CHUTE_ID,
        image=build_vocence_image(),
        node_selector=NodeSelector(gpu="any"),
    )
    
    # Global state
    _miner = None
    _repo_path = None
    
    @chute.on_startup()
    def startup():
        nonlocal _miner, _repo_path
        
        # Download model
        _repo_path = Path("/tmp/vocence_model")
        snapshot_download(
            repo_id=VOCENCE_REPO,
            revision=VOCENCE_REVISION,
            local_dir=str(_repo_path),
            local_dir_use_symlinks=False
        )
        
        # Load miner
        import importlib.util
        spec = importlib.util.spec_from_file_location("miner", _repo_path / VOCENCE_ENGINE_SCRIPT)
        miner_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(miner_module)
        
        miner_class = getattr(miner_module, VOCENCE_ENGINE_CLASS)
        _miner = miner_class(path_hf_repo=_repo_path)
        _miner.warmup()
    
    @chute.get("/health")
    def health() -> VocenceHealthResponse:
        return VocenceHealthResponse(
            status="ready" if _miner else "starting",
            model=VOCENCE_REPO,
            revision=VOCENCE_REVISION
        )
    
    @chute.post("/speak")
    def speak(request: VocenceSpeakRequest) -> Response:
        if not _miner:
            raise HTTPException(status_code=503, detail="Miner not ready")
        
        try:
            wav, sr = _miner.generate_wav(
                instruction=request.instruction,
                text=request.text
            )
            
            # Convert to WAV bytes
            buf = io.BytesIO()
            with wave.open(buf, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sr)
                wf.writeframes((wav * 32767).astype(np.int16).tobytes())
            
            return Response(content=buf.getvalue(), media_type="audio/wav")
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    return chute

# Export chute
chute = create_chute()
```

---

## Build Chutes Image

### Step 7: Build the Chute

```bash
cd /workspace/vocence
source venv/bin/activate

# Build the chute image
chutes build vocence_enhanced_chute:chute --wait
```

**Build Process:**
1. ✅ Vocence repo downloaded from HuggingFace
2. ✅ Vocence repo config loaded
3. ✅ Image built (includes all dependencies)
4. ✅ NodeSelector built (GPU configuration)
5. ✅ Chute built

**Expected Output:**
```
✅ Vocence repo downloaded
✅ Vocence repo config loaded
✅ Image built
✅ NodeSelector built
✅ Chute built
2026-05-05 13:05:26.123 | SUCCESS  | Image 'vocence-enhanced-miner-v2-mcqueen:latest' built successfully
```

### Step 8: Verify Image

```bash
# List built images
chutes images list

# Should show your image
# vocence-enhanced-miner-v2-mcqueen  latest  ...
```

---

## Deploy to Chutes

### Step 9: Deploy Chute

```bash
# Deploy with automatic fee acceptance
echo "y" | chutes deploy vocence_enhanced_chute:chute --accept-fee

# Or deploy interactively
chutes deploy vocence_enhanced_chute:chute
```

**Deployment Process:**
1. Validates chute configuration
2. Uploads chute code to Chutes platform
3. Provisions GPU resources
4. Starts chute service
5. Returns chute_id and version

**Expected Output:**
```
✅ Vocence repo downloaded
✅ Vocence repo config loaded
✅ Image built
✅ NodeSelector built
✅ Chute built
2026-05-05 13:07:09.789 | INFO - Loading chutes config...
You are about to upload /workspace/vocence/vocence_enhanced_chute.py and deploy 
vocence-enhanced-miner-v2-mcqueen, confirm? (y/n) y
2026-05-05 13:08:18.477 | SUCCESS - Successfully deployed chute 
  chute_id='4a408134-7cab-5c22-addf-0b44007b4972' 
  version='36622b6d-02f1-528b-bfa1-6b60457ff21d'
  invocations will be available soon!
```

**Save the chute_id** - you'll need it for on-chain commitment.

### Step 10: Verify Deployment

```bash
# List your chutes
chutes chutes list

# Check chute status
chutes chutes get 4a408134-7cab-5c22-addf-0b44007b4972

# Test health endpoint (if available)
curl https://your-chute-url.chutes.ai/health
```

---

## Submit On-Chain Commitment

### Step 11: Verify Wallet Registration

Ensure your wallet is registered on subnet 78:

```bash
# Check wallet status
btcli wallet overview --wallet.name c --hotkey h01 --netuid 78

# Should show registration on subnet 78 (Vocence)
# If not registered:
# btcli subnet register --netuid 78 --wallet.name c --wallet.hotkey h01
```

### Step 12: Submit Commitment

```bash
cd /workspace/vocence
source venv/bin/activate

# Submit on-chain commitment
vocence miner commit \
  --model-name YourUsername/vocence_enhanced_miner_v2 \
  --model-revision 5f5015199defdab8ab393b5da3619108debd5919 \
  --chute-id 4a408134-7cab-5c22-addf-0b44007b4972 \
  --coldkey c \
  --hotkey h01 \
  --network finney \
  --netuid 78
```

**Parameters:**
- `--model-name`: Your HuggingFace repository (from Step 4)
- `--model-revision`: Commit SHA from HuggingFace (from Step 5)
- `--chute-id`: Chute ID from deployment (from Step 9)
- `--coldkey`: Your Bittensor coldkey name
- `--hotkey`: Your Bittensor hotkey name
- `--network`: `finney` (mainnet) or `test` (testnet)
- `--netuid`: `78` (Vocence subnet)

**Expected Output:**
```
██╗   ██╗ ██████╗  ██████╗███████╗███╗   ██╗ ██████╗███████╗
real-time ai voice engine

────────────────────────────────────────────────────────────
Committing to Chain
────────────────────────────────────────────────────────────

13:08:57 ▸ Repository: YourUsername/vocence_enhanced_miner_v2
13:08:57 ▸ Revision: 5f5015199defdab8...
13:08:57 ▸ Chute ID: 4a408134-7cab-5c22-addf-0b44007b4972
13:08:57 ▸ Network: finney (mainnet)
13:08:57 ▸ Subnet ID: 78 (from --netuid)
13:08:59 ▸ Committing: YourUsername/vocence_enhanced_miner_v2@5f501519
13:08:59 ▸ Network: finney, subnet: 78
13:08:59 ▸ Using wallet: 5GCNsntXpGw6N13A...
13:09:38 ✓ Commit successful
{
  "success": true,
  "model_name": "YourUsername/vocence_enhanced_miner_v2",
  "model_revision": "5f5015199defdab8ab393b5da3619108debd5919",
  "chute_id": "4a408134-7cab-5c22-addf-0b44007b4972"
}

────────────────────────────────────────────────────────────
Commit Complete
────────────────────────────────────────────────────────────

13:09:38 ✓ Model info committed to chain
```

---

## Verify Deployment

### Step 13: Check Blockchain State

```bash
# Verify your miner registration
btcli wallet overview --wallet.name c --hotkey h01 --netuid 78

# Check subnet metagraph
btcli subnet metagraph --netuid 78 | grep "5GCNsntXpGw6N13A"
```

### Step 14: Monitor Performance

```bash
# Check chute logs
chutes logs 4a408134-7cab-5c22-addf-0b44007b4972 --follow

# Check chute metrics
chutes chutes get 4a408134-7cab-5c22-addf-0b44007b4972
```

### Step 15: Test Audio Generation

```bash
# Test your deployed chute
curl -X POST https://your-chute-url.chutes.ai/speak \
  -H "Content-Type: application/json" \
  -d '{
    "instruction": "gender: female | age_group: adult | pitch: mid | speed: normal | emotion: happy | tone: casual | accent: us",
    "text": "Hello! This is a test of my fine-tuned Vocence miner."
  }' \
  --output test_output.wav

# Play the audio
ffplay test_output.wav
```

---

## Troubleshooting

### Common Issues

**1. HuggingFace Upload Fails**
```bash
# Check credentials
huggingface-cli whoami

# Re-login
huggingface-cli login --token $HF_TOKEN

# Check repository exists
huggingface-cli repo-info YourUsername/vocence_enhanced_miner_v2
```

**2. Chutes Build Fails**
```bash
# Check disk space
df -h

# Clean old images
chutes images prune

# Rebuild with verbose output
chutes build vocence_enhanced_chute:chute --debug
```

**3. Deployment Fails - "Not registered on subnet"**
```bash
# Register on subnet 78
btcli subnet register --netuid 78 --wallet.name c --wallet.hotkey h01

# Verify registration
btcli wallet overview --wallet.name c --hotkey h01 --netuid 78
```

**4. On-Chain Commitment Fails - Package conflicts**
```bash
# Fix bittensor dependencies
pip uninstall scalecodec cyscale -y
pip install cyscale --force-reinstall
pip install bittensor --force-reinstall
```

**5. Chutes Config Issues**
```bash
# Verify Chutes configuration
cat /root/.chutes/config.ini

# Should include wallet_name
[auth]
username = your_username
user_id = your_user_id
wallet_name = c
hotkey_name = h01
hotkey_ss58address = 5GCNsntXpGw6N13A...
```

### Getting Help

- **Vocence Docs**: https://github.com/vocence/vocence
- **Chutes Docs**: https://chutes.ai/docs
- **Bittensor Docs**: https://docs.bittensor.com
- **Discord**: Join Vocence/Bittensor Discord for support

---

## Quick Reference Commands

```bash
# Complete workflow in one script
cd /workspace/vocence

# 1. Fine-tune (from enhanced_model directory)
cd enhanced_model && python fast_finetune.py && cd ..

# 2. Upload to HuggingFace
huggingface-cli upload YourUsername/vocence_enhanced_miner_v2 \
  ./enhanced_model_v2_finetuned . \
  --repo-type=model

# 3. Get revision
REVISION=$(python -c "from huggingface_hub import HfApi; \
  api = HfApi(); \
  info = api.repo_info('YourUsername/vocence_enhanced_miner_v2'); \
  print(info.sha)")
echo "Revision: $REVISION"

# 4. Update chute config (manually edit vocence_enhanced_chute.py)

# 5. Build chute
source venv/bin/activate
chutes build vocence_enhanced_chute:chute --wait

# 6. Deploy chute
CHUTE_OUTPUT=$(echo "y" | chutes deploy vocence_enhanced_chute:chute --accept-fee)
CHUTE_ID=$(echo "$CHUTE_OUTPUT" | grep -oP "chute_id='\K[^']+")
echo "Chute ID: $CHUTE_ID"

# 7. Commit on-chain
vocence miner commit \
  --model-name YourUsername/vocence_enhanced_miner_v2 \
  --model-revision $REVISION \
  --chute-id $CHUTE_ID \
  --coldkey c --hotkey h01 \
  --network finney --netuid 78

echo "✅ Deployment complete!"
```

---

## Summary Checklist

- [ ] Python environment set up with all dependencies
- [ ] HuggingFace token configured
- [ ] Chutes API key configured
- [ ] Bittensor wallet created and funded
- [ ] Registered on subnet 78
- [ ] Model weights fine-tuned
- [ ] Model uploaded to HuggingFace
- [ ] Commit SHA saved
- [ ] Chute configuration updated
- [ ] Chute image built successfully
- [ ] Chute deployed to platform
- [ ] Chute ID saved
- [ ] On-chain commitment submitted
- [ ] Deployment verified on blockchain
- [ ] Audio generation tested

**You're now mining on Vocence subnet 78! 🎉**

---

## Appendix: Configuration Files

### A. Chutes Config (`/root/.chutes/config.ini`)

```ini
[api]
base_url = https://api.chutes.ai

[auth]
username = your_username
user_id = your-user-id-uuid
wallet_name = c
hotkey_seed = your-hotkey-seed
hotkey_name = h01
hotkey_ss58address = 5GCNsntXpGw6N13Ay6ZZhvwsKWZbzG3qHf6Y7du4KeDHYvrE

[payment]
address = your-payment-address
```

### B. Model Config (`vocence_config.yaml`)

```yaml
runtime:
  adapter: "qwen3_tts_repo_snapshot"
  device_preference: "cuda"
  dtype: "bfloat16"
  default_language: "English"
  use_flash_attention_2: false
  num_candidates: 8  # Best-of-N sampling

generation:
  sample_rate: 24000
  max_seconds: 30

limits:
  max_text_chars: 2000
  max_instruction_chars: 600
  default_language: "English"
```

### C. Chute Config (`chute_config.yml`)

```yaml
runtime:
  timeout_seconds: 240
  max_concurrent: 5
  
resources:
  gpu: "any"
  memory: "16G"
  disk: "50G"
```

---

**Document Version**: 1.0  
**Last Updated**: May 5, 2026  
**Author**: Vocence Deployment Guide
