"""
Merge LoRA adapter weights into base model

Usage:
    python merge_lora.py \
        --base_model downloaded_models/top_model \
        --adapter finetuned_models/phase1_expresso/final_model \
        --output finetuned_models/phase1_expresso/merged_model
"""
import argparse
import json
from pathlib import Path
from datetime import datetime
import torch
from transformers import AutoConfig, AutoModel
from peft import PeftModel


def merge_lora_weights(
    base_model_path: str,
    adapter_path: str,
    output_path: str,
    dtype: str = "bfloat16"
):
    """
    Merge LoRA adapter weights into base model
    
    Args:
        base_model_path: Path to base model directory
        adapter_path: Path to LoRA adapter directory
        output_path: Path to save merged model
        dtype: Data type for merged model (bfloat16, float16, float32)
    """
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("LORA WEIGHT MERGING")
    print("="*80)
    print(f"\nBase model: {base_model_path}")
    print(f"Adapter: {adapter_path}")
    print(f"Output: {output_path}")
    print(f"Data type: {dtype}")
    
    # Map dtype string to torch dtype
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32
    }
    torch_dtype = dtype_map.get(dtype, torch.bfloat16)
    
    # Load base model
    print("\n[1/4] Loading base model...")
    try:
        config = AutoConfig.from_pretrained(base_model_path, trust_remote_code=True)
        base_model = AutoModel.from_pretrained(
            base_model_path,
            config=config,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
            device_map="auto"
        )
        print("✓ Base model loaded successfully")
    except Exception as e:
        print(f"✗ Error loading base model: {e}")
        raise
    
    # Load LoRA adapter
    print("\n[2/4] Loading LoRA adapter...")
    try:
        model = PeftModel.from_pretrained(
            base_model,
            adapter_path,
            torch_dtype=torch_dtype
        )
        print("✓ LoRA adapter loaded successfully")
        
        # Print LoRA configuration
        print("\nLoRA Configuration:")
        lora_config = model.peft_config['default']
        print(f"  Rank (r): {lora_config.r}")
        print(f"  Alpha: {lora_config.lora_alpha}")
        print(f"  Dropout: {lora_config.lora_dropout}")
        print(f"  Target modules: {lora_config.target_modules}")
        
    except Exception as e:
        print(f"✗ Error loading adapter: {e}")
        raise
    
    # Merge weights
    print("\n[3/4] Merging LoRA weights into base model...")
    try:
        merged_model = model.merge_and_unload()
        print("✓ Weights merged successfully")
    except Exception as e:
        print(f"✗ Error merging weights: {e}")
        raise
    
    # Save merged model
    print(f"\n[4/4] Saving merged model to {output_path}...")
    try:
        merged_model.save_pretrained(
            output_path,
            safe_serialization=True,
            max_shard_size="5GB"
        )
        print("✓ Model saved successfully")
        
        # Also save the config
        config.save_pretrained(output_path)
        print("✓ Config saved successfully")
        
    except Exception as e:
        print(f"✗ Error saving model: {e}")
        raise
    
    # Copy additional files from base model
    print("\n[5/5] Copying supporting files...")
    
    files_to_copy = [
        "tokenizer.json",
        "vocab.json",
        "merges.txt",
        "added_tokens.json",
        "special_tokens_map.json",
        "tokenizer_config.json",
        "preprocessor_config.json",
        "generation_config.json",
        "vocence_config.yaml",
        "chute_config.yml",
        "miner.py"
    ]
    
    base_path = Path(base_model_path)
    
    import shutil
    
    for filename in files_to_copy:
        src = base_path / filename
        if src.exists():
            dst = output_dir / filename
            shutil.copy2(src, dst)
            print(f"  ✓ Copied {filename}")
    
    # Copy speech_tokenizer directory
    speech_tokenizer_src = base_path / "speech_tokenizer"
    if speech_tokenizer_src.exists():
        speech_tokenizer_dst = output_dir / "speech_tokenizer"
        if speech_tokenizer_dst.exists():
            shutil.rmtree(speech_tokenizer_dst)
        shutil.copytree(speech_tokenizer_src, speech_tokenizer_dst)
        print("  ✓ Copied speech_tokenizer/")
    
    # Create merge info
    print("\n[6/6] Creating merge info...")
    
    merge_info = {
        "base_model_path": str(base_model_path),
        "adapter_path": str(adapter_path),
        "merged_path": str(output_path),
        "merged_dtype": dtype,
        "merged_at_utc": datetime.utcnow().isoformat() + "+00:00",
        "tool": "merge_lora.py",
        "adapter_config": {
            "r": lora_config.r,
            "lora_alpha": lora_config.lora_alpha,
            "lora_dropout": lora_config.lora_dropout,
            "target_modules": lora_config.target_modules,
            "bias": lora_config.bias,
            "task_type": lora_config.task_type,
        }
    }
    
    merge_info_path = output_dir / "merge_info.json"
    with open(merge_info_path, 'w') as f:
        json.dump(merge_info, f, indent=2)
    
    print(f"  ✓ Saved merge info to {merge_info_path}")
    
    # Print summary
    print("\n" + "="*80)
    print("MERGE COMPLETE")
    print("="*80)
    print(f"\nMerged model saved to: {output_path}")
    print("\nFiles created:")
    print("  - model.safetensors (merged weights)")
    print("  - config.json")
    print("  - tokenizer files")
    print("  - speech_tokenizer/")
    print("  - vocence_config.yaml")
    print("  - miner.py")
    print("  - merge_info.json")
    print("\nYou can now test the merged model with:")
    print(f"  python test_model_inference.py --model_path {output_path}")
    
    return merge_info


def main():
    parser = argparse.ArgumentParser(description="Merge LoRA adapter weights into base model")
    parser.add_argument(
        "--base_model",
        type=str,
        required=True,
        help="Path to base model directory"
    )
    parser.add_argument(
        "--adapter",
        type=str,
        required=True,
        help="Path to LoRA adapter directory"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to save merged model"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="Data type for merged model"
    )
    
    args = parser.parse_args()
    
    merge_lora_weights(
        base_model_path=args.base_model,
        adapter_path=args.adapter,
        output_path=args.output,
        dtype=args.dtype
    )


if __name__ == "__main__":
    main()
