#!/usr/bin/env python3
"""
Vocence Model Evaluation Script
Tests both top miner models against Vocence scoring criteria
"""
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch


# Test prompts matching Vocence evaluation format
TEST_CASES = [
    {
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
        "text": "The weather today is absolutely beautiful with clear blue skies.",
        "instruction": "gender: female | pitch: high | speed: slow | age_group: young_adult | emotion: happy | tone: warm | accent: uk",
        "expected": {
            "gender": "female",
            "pitch": "high",
            "speed": "slow",
            "age_group": "young_adult",
            "emotion": "happy",
            "tone": "warm",
            "accent": "uk"
        }
    },
    {
        "text": "This is a critical situation that requires immediate attention.",
        "instruction": "gender: male | pitch: low | speed: fast | age_group: senior | emotion: serious | tone: authoritative | accent: neutral",
        "expected": {
            "gender": "male",
            "pitch": "low",
            "speed": "fast",
            "age_group": "senior",
            "emotion": "serious",
            "tone": "authoritative",
            "accent": "neutral"
        }
    },
    {
        "text": "I'm so sorry to hear that happened.",
        "instruction": "gender: female | pitch: mid | speed: normal | age_group: adult | emotion: sad | tone: friendly | accent: us",
        "expected": {
            "gender": "female",
            "pitch": "mid",
            "speed": "normal",
            "age_group": "adult",
            "emotion": "sad",
            "tone": "friendly",
            "accent": "us"
        }
    },
    {
        "text": "Watch out! Something's coming!",
        "instruction": "gender: neutral | pitch: high | speed: fast | age_group: young_adult | emotion: fearful | tone: casual | accent: neutral",
        "expected": {
            "gender": "neutral",
            "pitch": "high",
            "speed": "fast",
            "age_group": "young_adult",
            "emotion": "fearful",
            "tone": "casual",
            "accent": "neutral"
        }
    }
]


def load_model_1(repo_path: str = "diegosilvabit904/vocence-tts-sft-v1"):
    """Load Model 1: Top miner with best-of-N sampling"""
    print(f"\n[Model 1] Loading {repo_path}...")
    from huggingface_hub import snapshot_download
    local_path = snapshot_download(repo_id=repo_path)
    
    # Import miner from downloaded repo
    import sys
    sys.path.insert(0, local_path)
    from miner import Miner
    
    miner = Miner(Path(local_path))
    miner.warmup()
    print("[Model 1] Ready!")
    return miner


def load_model_2(repo_path: str = "macminix/qwen3_voice_design_t2"):
    """Load Model 2: T2 candidate with gender-parity training"""
    print(f"\n[Model 2] Loading {repo_path}...")
    from huggingface_hub import snapshot_download
    local_path = snapshot_download(repo_id=repo_path)
    
    import sys
    sys.path.insert(0, local_path)
    from miner import Miner
    
    miner = Miner(Path(local_path))
    miner.warmup()
    print("[Model 2] Ready!")
    return miner


def calculate_wer(reference: str, hypothesis: str) -> float:
    """Calculate Word Error Rate (matches Vocence scoring)"""
    import re
    word_re = re.compile(r"\w+", re.UNICODE)
    ref = word_re.findall(reference.lower())
    hyp = word_re.findall(hypothesis.lower())
    
    if not ref:
        return 1.0 if hyp else 0.0
    
    n, m = len(ref), len(hyp)
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        curr = [i] + [0] * m
        for j in range(1, m + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
        prev = curr
    
    return min(1.0, prev[m] / n)


def score_with_utmos(waveform: np.ndarray, sample_rate: int) -> float:
    """Score naturalness with UTMOSv2"""
    try:
        import utmosv2
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = utmosv2.create_model(pretrained=True, device=device)
        
        wav = np.ascontiguousarray(waveform, dtype=np.float32)
        mos = model.predict(data=wav, sr=int(sample_rate))
        
        if hasattr(mos, "item"):
            mos = float(mos.item() if mos.ndim == 0 else mos.flatten()[0].item())
        elif isinstance(mos, np.ndarray):
            mos = float(mos.flatten()[0])
        else:
            mos = float(mos)
        
        return max(0.0, min(1.0, mos / 5.0))  # Normalize to [0, 1]
    except Exception as e:
        print(f"  WARNING: UTMOSv2 failed ({e})")
        return 0.5


def transcribe_audio(waveform: np.ndarray, sample_rate: int) -> str:
    """Transcribe audio with faster-whisper"""
    try:
        from faster_whisper import WhisperModel
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        compute = "float16" if device == "cuda" else "int8"
        model = WhisperModel("base", device=device, compute_type=compute)
        
        # Resample to 16kHz if needed
        if sample_rate != 16000:
            import torchaudio.functional as AF
            t = torch.from_numpy(waveform).unsqueeze(0)
            waveform = AF.resample(t, sample_rate, 16000).squeeze(0).numpy()
        
        segments, _ = model.transcribe(waveform, language="en", beam_size=1)
        return " ".join(seg.text for seg in segments).strip()
    except Exception as e:
        print(f"  WARNING: Whisper failed ({e})")
        return ""


def evaluate_model(miner: Any, test_cases: list, model_name: str, output_dir: Path):
    """Evaluate model on test cases"""
    print(f"\n{'='*60}")
    print(f"Evaluating {model_name}")
    print(f"{'='*60}")
    
    results = []
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for i, case in enumerate(test_cases):
        print(f"\n[Test {i+1}/{len(test_cases)}]")
        print(f"  Text: {case['text']}")
        print(f"  Instruction: {case['instruction']}")
        
        # Generate audio
        t0 = time.time()
        try:
            waveform, sr = miner.generate_wav(
                instruction=case['instruction'],
                text=case['text']
            )
            gen_time = time.time() - t0
            print(f"  Generation time: {gen_time:.2f}s")
        except Exception as e:
            print(f"  ERROR: Generation failed: {e}")
            results.append({
                "test_id": i,
                "error": str(e),
                "score": 0.0
            })
            continue
        
        # Save audio
        audio_path = output_dir / f"{model_name.replace(' ', '_')}_test_{i}.wav"
        sf.write(str(audio_path), waveform, sr)
        print(f"  Saved: {audio_path}")
        
        # Score with UTMOSv2
        utmos_score = score_with_utmos(waveform, sr)
        print(f"  UTMOS (naturalness): {utmos_score:.4f}")
        
        # Transcribe and calculate WER
        transcript = transcribe_audio(waveform, sr)
        wer = calculate_wer(case['text'], transcript)
        print(f"  Transcript: {transcript}")
        print(f"  WER: {wer:.4f}")
        
        # Calculate composite score (matching Vocence weighting)
        # Script: 30%, Naturalness: 15% (simplified, full eval needs all traits)
        script_score = max(0.0, 1.0 - wer)
        composite = 0.3 * script_score + 0.15 * utmos_score
        print(f"  Composite (partial): {composite:.4f}")
        
        # Validity checks
        duration = len(waveform) / sr
        rms = float(np.sqrt(np.mean(np.square(waveform))))
        peak = float(np.max(np.abs(waveform)))
        
        valid = True
        if duration < 2.0 or duration > 29.5:
            print(f"  WARNING: Duration out of range: {duration:.2f}s")
            valid = False
        if rms < 1e-3:
            print(f"  WARNING: RMS too low: {rms:.6f}")
            valid = False
        if peak >= 0.99:
            print(f"  WARNING: Peak clipping: {peak:.4f}")
            valid = False
        
        results.append({
            "test_id": i,
            "text": case['text'],
            "instruction": case['instruction'],
            "generation_time_s": gen_time,
            "duration_s": duration,
            "sample_rate": sr,
            "transcript": transcript,
            "wer": wer,
            "script_score": script_score,
            "utmos_score": utmos_score,
            "composite_score": composite,
            "rms": rms,
            "peak": peak,
            "valid": valid,
            "audio_file": str(audio_path)
        })
    
    # Summary
    valid_results = [r for r in results if r.get('valid', False) and 'error' not in r]
    if valid_results:
        avg_wer = np.mean([r['wer'] for r in valid_results])
        avg_utmos = np.mean([r['utmos_score'] for r in valid_results])
        avg_composite = np.mean([r['composite_score'] for r in valid_results])
        avg_time = np.mean([r['generation_time_s'] for r in valid_results])
        
        print(f"\n{'='*60}")
        print(f"{model_name} Summary")
        print(f"{'='*60}")
        print(f"Valid samples: {len(valid_results)}/{len(test_cases)}")
        print(f"Average WER: {avg_wer:.4f}")
        print(f"Average UTMOS: {avg_utmos:.4f}")
        print(f"Average Composite: {avg_composite:.4f}")
        print(f"Average Generation Time: {avg_time:.2f}s")
        print(f"Pass rate (score ≥ 0.9): {sum(r['composite_score'] >= 0.9 for r in valid_results)}/{len(valid_results)}")
    
    # Save results
    results_path = output_dir / f"{model_name.replace(' ', '_')}_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")
    
    return results


def main():
    """Main evaluation pipeline"""
    output_dir = Path("./evaluation_results")
    output_dir.mkdir(exist_ok=True)
    
    print("="*60)
    print("Vocence Model Evaluation")
    print("="*60)
    
    # Model 1: Top miner
    try:
        model1 = load_model_1()
        results1 = evaluate_model(model1, TEST_CASES, "Model_1_Top_Miner", output_dir)
    except Exception as e:
        print(f"\nModel 1 failed to load: {e}")
        results1 = None
    
    # Model 2: T2 candidate
    try:
        model2 = load_model_2()
        results2 = evaluate_model(model2, TEST_CASES, "Model_2_T2_Candidate", output_dir)
    except Exception as e:
        print(f"\nModel 2 failed to load: {e}")
        results2 = None
    
    # Comparison
    if results1 and results2:
        print(f"\n{'='*60}")
        print("Head-to-Head Comparison")
        print(f"{'='*60}")
        
        valid1 = [r for r in results1 if r.get('valid', False) and 'error' not in r]
        valid2 = [r for r in results2 if r.get('valid', False) and 'error' not in r]
        
        if valid1 and valid2:
            print(f"\n{'Metric':<25} | Model 1 | Model 2 | Winner")
            print("-" * 60)
            
            metrics = [
                ('WER (lower better)', 'wer', True),
                ('UTMOS (higher better)', 'utmos_score', False),
                ('Composite (higher better)', 'composite_score', False),
                ('Gen Time (lower better)', 'generation_time_s', True),
            ]
            
            for metric_name, key, lower_better in metrics:
                val1 = np.mean([r[key] for r in valid1])
                val2 = np.mean([r[key] for r in valid2])
                
                if lower_better:
                    winner = "Model 1" if val1 < val2 else "Model 2" if val2 < val1 else "Tie"
                else:
                    winner = "Model 1" if val1 > val2 else "Model 2" if val2 > val1 else "Tie"
                
                print(f"{metric_name:<25} | {val1:7.4f} | {val2:7.4f} | {winner}")


if __name__ == "__main__":
    main()
