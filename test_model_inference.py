"""
Model Testing Script for Qwen3-TTS
Tests model with various prompts and generates audio samples

Usage:
    python test_model_inference.py --model_path downloaded_models/top_model --output_dir test_outputs
"""
import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import soundfile as sf
import torch


def load_model(model_path: str):
    """Load Qwen3-TTS model from local path or HuggingFace"""
    try:
        from transformers import AutoModel, AutoConfig
        print(f"Loading model from: {model_path}")
        
        # Try loading with transformers first
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModel.from_pretrained(
            model_path,
            config=config,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        print(f"✓ Model loaded successfully")
        return model
    except Exception as e:
        print(f"Error loading with transformers: {e}")
        print("Trying alternative loading method (qwen_tts)...")
        
        try:
            # Try qwen_tts package if available
            from qwen_tts import Qwen3TTSModel
            model = Qwen3TTSModel.from_pretrained(model_path)
            print(f"✓ Model loaded successfully with qwen_tts")
            return model
        except Exception as e2:
            print(f"Error loading with qwen_tts: {e2}")
            raise RuntimeError(f"Failed to load model from {model_path}")


def generate_audio(model, text: str, instruction: str, language: str = "english") -> Tuple[np.ndarray, int]:
    """Generate audio from text with instruction-guided voice"""
    try:
        # Check if model has generate_voice_design method (qwen_tts package)
        if hasattr(model, 'generate_voice_design'):
            wavs, sr = model.generate_voice_design(
                text=text,
                instruct=instruction,
                language=language,
                temperature=0.9,
                top_p=1.0,
                top_k=50,
                repetition_penalty=1.05,
                max_new_tokens=600,
            )
            return wavs[0], sr
        
        # Otherwise, try standard transformers interface
        elif hasattr(model, 'generate'):
            # This would require proper preprocessing - simplified here
            print("Warning: Using simplified generation - may not work correctly")
            # You'd need to properly tokenize and prepare inputs here
            raise NotImplementedError("Direct transformers generation requires custom preprocessing")
        
        else:
            raise AttributeError("Model doesn't have expected generation methods")
            
    except Exception as e:
        raise RuntimeError(f"Failed to generate audio: {e}")


# Test prompts covering various scenarios
TEST_PROMPTS = [
    {
        "name": "male_excited",
        "text": "Come and look at this, you are not going to believe it.",
        "instruction": "gender: male | pitch: mid | speed: normal | age_group: adult | emotion: excited | tone: casual | accent: us",
        "expected": {
            "gender": "male",
            "pitch": "mid",
            "speed": "normal",
            "age_group": "adult",
            "emotion": "excited",
            "tone": "casual",
            "accent": "us"
        }
    },
    {
        "name": "female_calm_formal",
        "text": "The weather today is absolutely beautiful with clear blue skies.",
        "instruction": "gender: female | pitch: mid | speed: normal | age_group: adult | emotion: calm | tone: formal | accent: uk",
        "expected": {
            "gender": "female",
            "pitch": "mid",
            "speed": "normal",
            "age_group": "adult",
            "emotion": "calm",
            "tone": "formal",
            "accent": "uk"
        }
    },
    {
        "name": "male_angry",
        "text": "I told you a hundred times not to do that!",
        "instruction": "gender: male | pitch: low | speed: fast | age_group: adult | emotion: angry | tone: authoritative | accent: us",
        "expected": {
            "gender": "male",
            "pitch": "low",
            "speed": "fast",
            "age_group": "adult",
            "emotion": "angry",
            "tone": "authoritative",
            "accent": "us"
        }
    },
    {
        "name": "female_sad",
        "text": "I can't believe you're really leaving. This is so hard.",
        "instruction": "gender: female | pitch: mid | speed: slow | age_group: adult | emotion: sad | tone: warm | accent: us",
        "expected": {
            "gender": "female",
            "pitch": "mid",
            "speed": "slow",
            "age_group": "adult",
            "emotion": "sad",
            "tone": "warm",
            "accent": "us"
        }
    },
    {
        "name": "male_happy",
        "text": "We did it! The results are finally here and they're exactly what we hoped for!",
        "instruction": "gender: male | pitch: high | speed: fast | age_group: young_adult | emotion: happy | tone: friendly | accent: au",
        "expected": {
            "gender": "male",
            "pitch": "high",
            "speed": "fast",
            "age_group": "young_adult",
            "emotion": "happy",
            "tone": "friendly",
            "accent": "au"
        }
    },
    {
        "name": "female_serious",
        "text": "We need to address this situation immediately before it gets any worse.",
        "instruction": "gender: female | pitch: low | speed: normal | age_group: adult | emotion: serious | tone: authoritative | accent: us",
        "expected": {
            "gender": "female",
            "pitch": "low",
            "speed": "normal",
            "age_group": "adult",
            "emotion": "serious",
            "tone": "authoritative",
            "accent": "us"
        }
    },
    {
        "name": "male_fearful",
        "text": "Did you hear that sound? I think someone's in the house.",
        "instruction": "gender: male | pitch: mid | speed: fast | age_group: adult | emotion: fearful | tone: casual | accent: us",
        "expected": {
            "gender": "male",
            "pitch": "mid",
            "speed": "fast",
            "age_group": "adult",
            "emotion": "fearful",
            "tone": "casual",
            "accent": "us"
        }
    },
    {
        "name": "female_neutral",
        "text": "The meeting is scheduled for tomorrow at three o'clock in conference room B.",
        "instruction": "gender: female | pitch: mid | speed: normal | age_group: adult | emotion: neutral | tone: formal | accent: us",
        "expected": {
            "gender": "female",
            "pitch": "mid",
            "speed": "normal",
            "age_group": "adult",
            "emotion": "neutral",
            "tone": "formal",
            "accent": "us"
        }
    },
]


def test_model(model, output_dir: Path, num_tests: int = None):
    """Generate audio for test cases"""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    test_cases = TEST_PROMPTS if num_tests is None else TEST_PROMPTS[:num_tests]
    
    print(f"\n{'='*80}")
    print(f"TESTING MODEL WITH {len(test_cases)} PROMPTS")
    print(f"{'='*80}\n")
    
    results = []
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"[{i}/{len(test_cases)}] Generating: {test_case['name']}")
        print(f"  Text: {test_case['text'][:60]}...")
        print(f"  Instruction: {test_case['instruction'][:80]}...")
        
        try:
            start_time = time.time()
            
            # Generate audio
            audio, sample_rate = generate_audio(
                model,
                text=test_case['text'],
                instruction=test_case['instruction'],
                language="english"
            )
            
            generation_time = time.time() - start_time
            
            # Save audio
            output_path = output_dir / f"{test_case['name']}.wav"
            sf.write(output_path, audio, sample_rate)
            
            # Calculate basic metrics
            duration = len(audio) / sample_rate
            audio_rms = np.sqrt(np.mean(audio**2))
            audio_peak = np.max(np.abs(audio))
            
            result = {
                "name": test_case['name'],
                "text": test_case['text'],
                "instruction": test_case['instruction'],
                "expected": test_case['expected'],
                "output_file": str(output_path),
                "generation_time": round(generation_time, 2),
                "duration": round(duration, 2),
                "sample_rate": sample_rate,
                "audio_rms": round(float(audio_rms), 4),
                "audio_peak": round(float(audio_peak), 4),
                "success": True
            }
            
            print(f"  ✓ Generated in {generation_time:.2f}s")
            print(f"  ✓ Duration: {duration:.2f}s, RMS: {audio_rms:.4f}, Peak: {audio_peak:.4f}")
            print(f"  ✓ Saved to: {output_path}\n")
            
        except Exception as e:
            print(f"  ✗ Error: {e}\n")
            result = {
                "name": test_case['name'],
                "text": test_case['text'],
                "instruction": test_case['instruction'],
                "expected": test_case['expected'],
                "error": str(e),
                "success": False
            }
        
        results.append(result)
    
    # Save results JSON
    results_path = output_dir / "test_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print(f"\n{'='*80}")
    print("TEST SUMMARY")
    print(f"{'='*80}\n")
    
    successful = [r for r in results if r.get('success')]
    failed = [r for r in results if not r.get('success')]
    
    print(f"  Total tests: {len(results)}")
    print(f"  ✓ Successful: {len(successful)}")
    print(f"  ✗ Failed: {len(failed)}")
    
    if successful:
        avg_time = np.mean([r['generation_time'] for r in successful])
        avg_duration = np.mean([r['duration'] for r in successful])
        print(f"\n  Average generation time: {avg_time:.2f}s")
        print(f"  Average audio duration: {avg_duration:.2f}s")
    
    print(f"\n  Results saved to: {results_path}")
    print(f"  Audio files saved to: {output_dir}/")
    
    if failed:
        print(f"\n  Failed tests:")
        for r in failed:
            print(f"    • {r['name']}: {r['error']}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Test Qwen3-TTS model with various prompts")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to model directory or HuggingFace model ID"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="test_outputs",
        help="Directory to save generated audio files"
    )
    parser.add_argument(
        "--num_tests",
        type=int,
        default=None,
        help="Number of tests to run (default: all)"
    )
    
    args = parser.parse_args()
    
    print("="*80)
    print("QWEN3-TTS MODEL TESTING SCRIPT")
    print("="*80)
    print(f"\nModel: {args.model_path}")
    print(f"Output: {args.output_dir}")
    print(f"Tests: {args.num_tests if args.num_tests else 'All'}")
    
    # Load model
    model = load_model(args.model_path)
    
    # Run tests
    output_dir = Path(args.output_dir)
    results = test_model(model, output_dir, args.num_tests)
    
    print(f"\n{'='*80}")
    print("TESTING COMPLETE")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
