#!/usr/bin/env python3
"""
Simple model test script - directly import from downloaded models
"""
import sys
from pathlib import Path
import numpy as np
import soundfile as sf

def test_model(repo_id: str, model_name: str):
    """Test a single model"""
    print(f"\n{'='*70}")
    print(f"Testing: {model_name}")
    print(f"Repo: {repo_id}")
    print(f"{'='*70}")
    
    try:
        # Download model
        from huggingface_hub import snapshot_download
        local_path = Path(snapshot_download(repo_id=repo_id))
        print(f"✓ Downloaded to: {local_path}")
        
        # Add to path and import miner
        sys.path.insert(0, str(local_path))
        from miner import Miner
        
        # Initialize
        print("✓ Initializing miner...")
        miner = Miner(local_path)
        
        # Warmup
        print("✓ Running warmup...")
        miner.warmup()
        
        # Generate test audio
        print("✓ Generating test audio...")
        text = "Hello, this is a test."
        instruction = "gender: female | pitch: mid | speed: normal | age_group: adult | emotion: neutral | tone: friendly | accent: us"
        
        waveform, sr = miner.generate_wav(instruction=instruction, text=text)
        
        # Save output
        output_file = f"{model_name.replace(' ', '_')}_test.wav"
        sf.write(output_file, waveform, sr)
        
        print(f"✓ SUCCESS! Generated {len(waveform)/sr:.2f}s audio")
        print(f"✓ Saved to: {output_file}")
        print(f"  Sample rate: {sr} Hz")
        print(f"  Duration: {len(waveform)/sr:.2f}s")
        print(f"  Peak: {np.max(np.abs(waveform)):.4f}")
        print(f"  RMS: {np.sqrt(np.mean(np.square(waveform))):.4f}")
        
        # Clean up path
        sys.path.remove(str(local_path))
        
        return True
        
    except Exception as e:
        print(f"✗ ERROR: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("="*70)
    print("Vocence Model Testing")
    print("="*70)
    
    models = [
        ("diegosilvabit904/vocence-tts-sft-v1", "Model_1_diegosilvabit904"),
        ("macminix/qwen3_voice_design_t2", "Model_2_macminix"),
    ]
    
    results = {}
    for repo_id, model_name in models:
        results[model_name] = test_model(repo_id, model_name)
    
    # Summary
    print(f"\n{'='*70}")
    print("Summary")
    print(f"{'='*70}")
    for model_name, success in results.items():
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status} - {model_name}")


if __name__ == "__main__":
    main()
