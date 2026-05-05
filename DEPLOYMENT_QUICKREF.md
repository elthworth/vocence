# Vocence Deployment Quick Reference

**Quick commands for deploying fine-tuned models to Bittensor subnet 78**

---

## 🚀 Complete Deployment (One-Liner Per Step)

### 1️⃣ Fine-Tune Model Weights
```bash
cd /workspace/vocence/enhanced_model
source ../venv/bin/activate
python fast_finetune.py
```

### 2️⃣ Upload to HuggingFace
```bash
cd /workspace/vocence/enhanced_model_v2_finetuned
export HF_TOKEN="hf_YourTokenHere"
huggingface-cli upload YourUsername/vocence_enhanced_miner_v2 . . --repo-type=model --commit-message="Fine-tuned weights"
```

### 3️⃣ Get Commit SHA
```bash
python -c "from huggingface_hub import HfApi; api = HfApi(); info = api.repo_info('YourUsername/vocence_enhanced_miner_v2'); print(f'Revision: {info.sha}')"
```

### 4️⃣ Update Chute Config
```bash
# Edit vocence_enhanced_chute.py and update:
# VOCENCE_REPO = "YourUsername/vocence_enhanced_miner_v2"
# VOCENCE_REVISION = "your_commit_sha_from_step_3"
```

### 5️⃣ Build Chute Image
```bash
cd /workspace/vocence
source venv/bin/activate
chutes build vocence_enhanced_chute:chute --wait
```

### 6️⃣ Deploy to Chutes
```bash
echo "y" | chutes deploy vocence_enhanced_chute:chute --accept-fee
# Save the chute_id from output
```

### 7️⃣ Submit On-Chain Commitment
```bash
vocence miner commit \
  --model-name YourUsername/vocence_enhanced_miner_v2 \
  --model-revision YOUR_COMMIT_SHA \
  --chute-id YOUR_CHUTE_ID \
  --coldkey c --hotkey h01 \
  --network finney --netuid 78
```

---

## 📋 Verification Commands

```bash
# Check HuggingFace upload
huggingface-cli repo-info YourUsername/vocence_enhanced_miner_v2

# List Chutes images
chutes images list

# List deployed chutes
chutes chutes list

# Check wallet registration
btcli wallet overview --wallet.name c --hotkey h01 --netuid 78

# Monitor chute logs
chutes logs YOUR_CHUTE_ID --follow
```

---

## 🔧 Troubleshooting

### Fix package conflicts:
```bash
pip uninstall scalecodec cyscale -y
pip install cyscale --force-reinstall
pip install bittensor --force-reinstall
```

### Clean disk space:
```bash
chutes images prune
rm -rf ~/.cache/huggingface/hub/*
```

### Re-register on subnet:
```bash
btcli subnet register --netuid 78 --wallet.name c --wallet.hotkey h01
```

### Update Chutes config:
```bash
# Add wallet_name to /root/.chutes/config.ini
[auth]
wallet_name = c
```

---

## 📊 Key Information to Save

After each deployment, save these values:

| Item | Value | Used For |
|------|-------|----------|
| HF Repo | `YourUsername/vocence_enhanced_miner_v2` | Model reference |
| Commit SHA | `5f5015199defdab8...` | Model version |
| Chute ID | `4a408134-7cab-5c22...` | Deployment reference |
| Coldkey | `c` | Wallet identification |
| Hotkey | `h01` | Mining identity |
| UID | `167` | Subnet position |

---

## 🎯 Success Indicators

✅ **Fine-Tuning**: Modified 99%+ of weight tensors  
✅ **HuggingFace**: Repository shows new commit with updated files  
✅ **Chutes Build**: "Image built successfully" message  
✅ **Chutes Deploy**: Returns chute_id and version  
✅ **On-Chain**: "Commit successful" with transaction details  

---

**Full documentation**: [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md)
