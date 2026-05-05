#!/usr/bin/env python3
"""
Simple Model Inspector - Check downloaded models without running inference
"""
import json
from pathlib import Path
from huggingface_hub import snapshot_download


def inspect_model(repo_id: str, model_name: str):
    """Inspect a model's structure and configuration"""
    print(f"\n{'='*70}")
    print(f"{model_name}")
    print(f"{'='*70}")
    print(f"Repository: {repo_id}")
    
    try:
        # Download to cache
        local_path = Path(snapshot_download(repo_id=repo_id))
        print(f"Local path: {local_path}")
        
        # List files
        print(f"\n📁 Files:")
        all_files = sorted(local_path.rglob("*"))
        important_files = [
            f for f in all_files 
            if f.is_file() and f.suffix in ['.py', '.json', '.yml', '.yaml', '.md', '.txt']
            and f.stat().st_size < 1_000_000  # Under 1MB
        ]
        
        for f in important_files[:20]:  # Show first 20
            rel_path = f.relative_to(local_path)
            size = f.stat().st_size
            print(f"  - {rel_path} ({size:,} bytes)")
        
        # Check for config files
        config_file = local_path / "config.json"
        if config_file.exists():
            print(f"\n⚙️  Model Configuration:")
            with open(config_file) as f:
                config = json.load(f)
                for key in ['model_type', 'architectures', '_name_or_path', 
                           'hidden_size', 'num_hidden_layers', 'vocab_size']:
                    if key in config:
                        print(f"  {key}: {config[key]}")
        
        # Check vocence config
        vocence_config = local_path / "vocence_config.yaml"
        if vocence_config.exists():
            print(f"\n🎯 Vocence Configuration:")
            with open(vocence_config) as f:
                content = f.read()
                print(f"{content}")
        
        # Check miner.py
        miner_file = local_path / "miner.py"
        if miner_file.exists():
            print(f"\n🔧 Miner Implementation:")
            with open(miner_file) as f:
                lines = f.readlines()
                # Show first 50 lines
                print("".join(lines[:50]))
                if len(lines) > 50:
                    print(f"\n  ... ({len(lines) - 50} more lines)")
        
        # Check model sizes
        print(f"\n💾 Model Weights:")
        safetensors_files = list(local_path.rglob("*.safetensors"))
        total_size = 0
        for st_file in safetensors_files:
            size = st_file.stat().st_size
            total_size += size
            rel_path = st_file.relative_to(local_path)
            print(f"  - {rel_path}: {size / (1024**3):.2f} GB")
        
        print(f"  Total model size: {total_size / (1024**3):.2f} GB")
        
        # Check chute config
        chute_config = local_path / "chute_config.yml"
        if chute_config.exists():
            print(f"\n🚀 Chute Configuration:")
            with open(chute_config) as f:
                content = f.read()
                print(content)
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error inspecting model: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main inspection function"""
    print("="*70)
    print("Vocence Model Inspector")
    print("="*70)
    print("\nThis script inspects model structure without running inference.")
    print("Full inference requires GPU and proper qwen-tts setup.\n")
    
    models = [
        ("diegosilvabit904/vocence-tts-sft-v1", "Model 1: Top Miner (Best-of-5)"),
        ("macminix/qwen3_voice_design_t2", "Model 2: T2 Candidate (Gender-Parity)"),
    ]
    
    for repo_id, model_name in models:
        success = inspect_model(repo_id, model_name)
        if success:
            print(f"\n✅ Successfully inspected {model_name}")
        else:
            print(f"\n❌ Failed to inspect {model_name}")
    
    print(f"\n{'='*70}")
    print("Summary")
    print(f"{'='*70}")
    print("""
Key Differences:

Model 1 (diegosilvabit904/vocence-tts-sft-v1):
  - Base: T1 checkpoint-600 (merged LoRA)
  - Strategy: Best-of-5 sequential sampling
  - Selection: UTMOSv2 + Whisper composite scoring
  - Complexity: 31.6 KB miner.py (complex implementation)
  - Expected pass rate: ~90%
  
Model 2 (macminix/qwen3_voice_design_t2):
  - Base: T2 (gender-parity trained, 3 epochs)
  - Strategy: Single generation
  - Selection: None
  - Complexity: 5.46 KB miner.py (simple implementation)
  - UTMOS: 3.086 (+2.8% vs baseline)
  - Expected pass rate: ~75%

Recommendation for building a better model:
  → Combine T2's superior base with Model 1's best-of-N sampling
  → See BUILD_BETTER_MODEL_GUIDE.md for details
    """)


if __name__ == "__main__":
    main()
