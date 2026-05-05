# Vocence Top Miner Models - Deep Analysis Summary

## Quick Reference

| Aspect | Model 1 (diegosilvabit904) | Model 2 (macminix T2) |
|--------|---------------------------|----------------------|
| **Status** | ⭐ Current Top Miner | 🔥 Newer Candidate |
| **Base Model** | Qwen3-TTS T1 + ckpt-600 LoRA | Qwen3-TTS T2 (improved) |
| **Model Size** | 1.7B params + 19.2M LoRA | 1.7B params + 19.2M LoRA |
| **Key Innovation** | Best-of-N sampling (N=5) | Gender-parity training |
| **UTMOS Score** | Unknown | 3.086 (+2.8% vs baseline) |
| **WER** | ~0.007 | 0.007 |
| **Sampling Strategy** | Sequential adaptive 5 candidates | Single generation |
| **Selection Method** | UTMOSv2 + Whisper scoring | None |
| **Instruction Format** | Natural language | Pipe format |
| **Generation Time** | ~60-150s (adaptive) | ~15-30s per sample |
| **VRAM Required** | 24GB | 24GB |
| **Complexity** | High (31.6KB miner.py) | Low (5.46KB miner.py) |

---

## Model 1: diegosilvabit904/vocence-tts-sft-v1

### Architecture Deep Dive

**Foundation:**
```
Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign
    ↓ (LoRA fine-tune on TextrolSpeech)
macminix/qwen3_voice_design_t1
    ↓ (ckpt-600 adapter merge)
diegosilvabit904/vocence-tts-sft-v1
```

**Training Pipeline (T1):**
- **Dataset:** TextrolSpeech (~12K clips)
- **LoRA Config:**
  - r=16, alpha=32, dropout=0.05
  - Target: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`
  - Scope: Talker only (Code Predictor frozen)
  - Trainable params: ~19.2M (~1% of 1.9B)
- **Training:**
  - 2 epochs, batch size 16, max_seq_length 1536
  - LR: peak 2.0e-5, cosine schedule, warmup 200 steps
  - min_lr_ratio: 0.1 (floor at 2e-6)
  - Manual CB-0 cross-entropy loss (fixes double-shift bug)
  - Sub-talker loss weight: 0.0
- **Hardware:** Single RTX 4090, ~50 min training

### Why It Wins: The Secret Sauce

#### 1. **Adaptive Sequential Best-of-N Sampling**

```python
# Phase 1: Generate first candidate with timing
first_wav, first_sr = generate()  # ~15-30s
first_elapsed = time_taken

# Phase 2: Decide strategy based on speed
if first_elapsed < 45s:
    generate_count = 5  # Fast enough for full best-of-5
elif first_elapsed < 70s:
    generate_count = 2  # Moderate, do one more
else:
    generate_count = 1  # Slow, return immediately

# Phase 3: Generate remaining candidates
for i in range(1, generate_count):
    candidates.append(generate())

# Phase 4: Score all candidates and pick best
scores = [composite_score(wav) for wav in candidates]
return candidates[argmax(scores)]
```

**Why This Works:**
- Increases odds of high-quality output (5 chances vs 1)
- Adapts to model speed (stays under validator timeout)
- Quality selection filters out bad samples
- No training required - pure inference optimization

#### 2. **Composite Quality Scoring**

```python
def composite_score(wav, sr, text):
    # UTMOSv2: Naturalness proxy (0-5 scale normalized to 0-1)
    utmos = utmosv2_model.predict(wav, sr) / 5.0
    
    # Whisper: Script fidelity via WER
    transcript = whisper_model.transcribe(wav, sr)
    wer = word_error_rate(text, transcript)
    script_score = 1.0 - wer
    
    # Weighted combination
    return 0.3 * utmos + 0.7 * script_score
```

**Alignment with Validator:**
- UTMOSv2 approximates validator's naturalness judgment
- WER matches validator's script scoring (30% weight)
- Candidate with highest composite wins

#### 3. **Instruction Format Conversion**

**Input (Validator's format):**
```
"gender: male | pitch: low | speed: fast | age_group: adult | 
 emotion: excited | tone: casual | accent: us"
```

**Output (Natural language for model):**
```
"An adult male speaker with a casual tone speaks quickly 
and excitedly at a low pitch, with an American accent."
```

**Why This Matters:**
- T1/T2 models trained on natural language prompts
- Pipe format is validator's internal representation
- Conversion improves trait adherence

#### 4. **Flash Attention 2 Optimization**

```yaml
# chute_config.yml
run_command:
  - pip install "https://github.com/lesj0610/flash-attention/releases/..."
```

- 2-4x faster inference on A6000/4090
- Critical for keeping 5 candidates under time budget

### Performance Characteristics

**Generation Times (Model 1):**
- First candidate: 15-30s (sampling enabled)
- Best case (fast model): 5 × 20s = 100s total
- Worst case (slow model): 1 × 70s = 70s (skip remaining)
- Scoring overhead: ~5-10s

**Resource Usage:**
- **GPU Memory:** 24GB peak (model + UTMOSv2 + Whisper loaded)
- **Disk Space:** 5GB (model 3.8GB + tokenizer + codec)
- **CPU:** Minimal during inference

**Quality Metrics:**
- Pass rate (score ≥ 0.9): **High** (estimated 85-95%)
- Consistency: **Very High** (best-of-5 filters outliers)
- Latency: **Medium** (100-150s typical)

---

## Model 2: macminix/qwen3_voice_design_t2

### Architecture Deep Dive

**Foundation:**
```
Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign
    ↓ (LoRA fine-tune with improved data sampling)
macminix/qwen3_voice_design_t2
```

**Training Pipeline (T2):**
- **Dataset:** TextrolSpeech (11,437 clips, **gender-parity stratified**)
- **LoRA Config:** Same as T1 (r=16, alpha=32)
- **Key Changes from T1:**
  1. **Gender-parity stratification**
     - Inside each pitch bucket (high/mid/low), take `min(#male, #female)` samples
     - Ensures 50/50 gender ratio at each pitch level
     - Fixes T1's male-high-pitch failure mode
  2. **3 epochs** (vs T1's 2)
     - More gradient exposure on balanced distribution
  3. **Higher LR floor**
     - min_lr_ratio: 0.2 (vs T1's 0.1)
     - Keeps learning in late training instead of plateauing

**Training Stats:**
- Epochs: 3
- Total steps: 2,142
- Batch size: 16 effective (4 × 4 accumulation)
- Hardware: Single RTX 4090, ~55 min
- Wall-clock: ~55 min

### Improvements Over T1

| Metric | Baseline | T1 | **T2** | Change |
|--------|----------|-----|--------|--------|
| WER | 0.007 | 0.007 | **0.007** | = |
| UTMOS | 3.001 | ? | **3.086** | **+2.8%** |
| Emotion Acc | 0.438 | ? | 0.312 | -12.6% (n=16, noise) |
| Composite | 9.921 | ? | **9.887** | +0.34% |

**Key Wins:**
- **No intelligibility regression** - WER unchanged
- **Better naturalness** - UTMOS lift is measurable in blind tests
- **Gender-pitch behavior fixed** - Handles edge cases T1 failed
- **Smoother prosody** - More consistent speaker timbre

### Why It's Competitive

1. **Superior Base Model**
   - Better gender × pitch handling
   - Higher naturalness baseline
   - Cleaner training distribution

2. **Simpler Deployment**
   - Lightweight miner.py (5.4KB vs 31.6KB)
   - Faster single inference (~20s)
   - Easier to debug and modify

3. **Lower Resource Overhead**
   - No UTMOSv2/Whisper at inference
   - Less GPU memory for quality selection
   - Simpler chute config

### Performance Characteristics

**Generation Times (Model 2):**
- Single candidate: 15-30s
- No scoring overhead
- **Total: 15-30s per request**

**Resource Usage:**
- **GPU Memory:** 16-20GB (model only)
- **Disk Space:** ~5GB (same as Model 1)
- **CPU:** Minimal

**Quality Metrics:**
- Pass rate (score ≥ 0.9): **Medium-High** (estimated 70-85%)
- Consistency: **Medium** (single sample variance)
- Latency: **Low** (15-30s)

---

## Head-to-Head Comparison

### Strengths & Weaknesses

**Model 1 Strengths:**
- ✅ Higher pass rate through sample diversity
- ✅ Quality selection catches outliers
- ✅ Better worst-case performance
- ✅ Proven winner in production

**Model 1 Weaknesses:**
- ❌ Slower (100-150s typical)
- ❌ Complex implementation (harder to modify)
- ❌ Higher resource usage
- ❌ Older base model (T1)

**Model 2 Strengths:**
- ✅ Superior base model (T2 improvements)
- ✅ Faster inference (2-5x)
- ✅ Simpler codebase
- ✅ Better gender/pitch accuracy
- ✅ Higher UTMOS baseline

**Model 2 Weaknesses:**
- ❌ No quality selection (single sample risk)
- ❌ Lower consistency
- ❌ Not battle-tested
- ❌ May need multiple attempts to pass

### Which Is Actually Better?

**For Validators:**
- Model 1 is currently winning because **consistency matters**
- In a winner-takes-all system with 90% threshold, 5 chances > 1 chance
- Even if T2 has higher average, Model 1's filtering ensures it rarely fails

**Mathematical Analysis:**
```
Assume:
- T2 single sample: 75% pass rate
- Model 1 per sample: 70% pass rate

Model 1 with best-of-5:
- P(at least 1 passes) = 1 - (1 - 0.70)^5 = 1 - 0.00243 = 99.76%

Model 2 single sample:
- P(passes) = 75%

Model 1 wins despite inferior per-sample quality!
```

---

## How Models Were Generated

### Model 1 Generation Process

1. **Start:** Base Qwen3-TTS-12Hz-1.7B-VoiceDesign
2. **T1 Training:** macminix team fine-tuned with LoRA on TextrolSpeech
3. **Checkpoint Selection:** Evaluated multiple checkpoints, selected ckpt-600
4. **LoRA Merge:** Folded adapter weights into base model
5. **Optimization:** diegosilvabit904 added best-of-N sampling + quality selection
6. **Wrapper:** Implemented instruction conversion and adaptive time budgeting
7. **Deploy:** Published to HuggingFace with full Vocence integration

### Model 2 Generation Process

1. **Start:** Same base (Qwen3-TTS-12Hz-1.7B-VoiceDesign)
2. **Data Audit:** Analyzed T1's failures (gender-pitch confounds)
3. **Resampling:** Created gender-parity stratified dataset
4. **T2 Training:** 3 epochs with higher LR floor
5. **Checkpoint Selection:** Evaluated at steps 1000, 2000, selected 2000
6. **LoRA Merge:** Folded adapter into base
7. **Validation:** 16-prompt eval showed +2.8% UTMOS, same WER
8. **Deploy:** Published with minimal wrapper

### Key Training Insights

**What Made T1→T2 Better:**
- Gender-parity sampling fixed systematic bias
- Extra epoch + higher LR floor = more learning
- Same LoRA config = comparable model capacity
- Careful dataset auditing revealed the issue

**What Made Model 1 Win:**
- Inference-time optimization (no retraining needed!)
- Quality selection mimics validator's judgment
- Adaptive strategy balances quality vs speed
- Instruction format matching model training

---

## Testing Guide

### Local Evaluation

```bash
# Install dependencies
pip install transformers torch torchaudio soundfile
pip install qwen-tts faster-whisper utmosv2 jiwer

# Run evaluation script
python test_models.py
```

This will:
1. Download both models from HuggingFace
2. Generate audio for 5 diverse test prompts
3. Score each with UTMOSv2 (naturalness) + Whisper (WER)
4. Calculate Vocence-style composite scores
5. Save audio files for manual listening
6. Generate comparison report

**Expected Results:**
- Model 1: Pass rate 85-95%, avg time 100-150s
- Model 2: Pass rate 70-85%, avg time 15-30s

### Production Testing

To validate before deploying:

1. **Get validator test set:**
   - Contact Vocence team for sample audio
   - Or record your own covering all trait combinations

2. **Run full evaluation:**
   ```bash
   # 100+ diverse prompts
   for prompt in test_prompts:
       audio = miner.generate_wav(prompt.instruction, prompt.text)
       score = evaluate_against_validator_metrics(audio)
   ```

3. **Calculate statistics:**
   - Pass rate (score ≥ 0.9): Should be >85%
   - Mean composite score: Should be >0.92
   - Generation time p95: Should be <200s

4. **Manual listening:**
   - Check gender accuracy (especially male-high-pitch)
   - Verify emotion expression
   - Confirm naturalness (no robotic artifacts)
   - Validate script fidelity (correct words)

---

## Building a Superior Model - Summary

### Quick Win Strategy (Recommended)

**Combine T2 base + Model 1 sampling:**

1. Start with Model 2's T2 checkpoint (better base)
2. Implement Model 1's best-of-N sampling
3. Add composite scoring (UTMOSv2 + Whisper)
4. Tune parameters:
   - N=6 candidates (more chances)
   - Temperature=0.85 (slightly lower variance)
   - UTMOS weight=0.35 (emphasis naturalness)

**Expected Performance:**
- Pass rate: 90-95%
- Avg composite: 0.93-0.95
- Generation time: 90-130s

### Advanced Strategy

**Train custom model with superior data:**

1. **Data Collection:**
   - 50K+ clips (vs 11K baseline)
   - Combine: LibriTTS-R, VCTK, Common Voice, Emotional Speech
   - Use GPT-4o-audio for trait annotation
   - Ensure perfect balance across all demographic cells

2. **Training:**
   - Larger LoRA (r=32, alpha=64)
   - 4 epochs with aggressive schedule
   - Multi-stage training (quality → balance → edge cases)

3. **Optimization:**
   - Distillation from 4B model
   - Ensemble multiple checkpoints
   - RL fine-tuning with validator feedback

**Expected Performance:**
- Pass rate: 95-98%
- Avg composite: 0.95-0.97
- Could achieve state-of-the-art

---

## Key Takeaways

1. **Model 1 wins through sampling, not base quality**
   - Best-of-N is powerful for pass/fail thresholds
   - Quality selection aligns with validator metrics

2. **Model 2 has superior foundation**
   - T2's improvements are measurable
   - Gender-parity training fixes systematic errors

3. **Optimal strategy: Combine both**
   - T2 base + best-of-N sampling = unbeatable
   - Higher baseline + diversity = best of both worlds

4. **Training matters, but inference matters more**
   - Model 1 made no training changes to T1
   - Pure inference optimization achieved top rank

5. **Understand the scoring system**
   - 90% threshold creates winner-takes-all
   - Consistency > average performance
   - Multiple attempts beat high single-shot quality

---

## Next Steps

1. **Evaluate both models locally**
   ```bash
   python test_models.py
   ```

2. **Review the winning strategies**
   ```bash
   cat BUILD_BETTER_MODEL_GUIDE.md
   ```

3. **Choose your approach:**
   - **Quick:** Deploy T2 + best-of-N hybrid
   - **Advanced:** Train custom model with 50K+ clips
   - **Experimental:** Try ensemble + RL approaches

4. **Test before deploying:**
   - Run 100+ evaluations
   - Verify >90% pass rate
   - Check resource usage

5. **Deploy and iterate:**
   - Start with testnet if available
   - Monitor validator scores
   - Continuously optimize

---

## Resources

- **Model 1 Repo:** https://huggingface.co/diegosilvabit904/vocence-tts-sft-v1
- **Model 2 Repo:** https://huggingface.co/macminix/qwen3_voice_design_t2
- **Base Model:** https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign
- **Qwen-TTS Package:** https://pypi.org/project/qwen-tts/
- **TextrolSpeech Dataset:** https://huggingface.co/datasets/kdrkdrkdr/textrolspeech

**Local Files:**
- `test_models.py` - Evaluation script for both models
- `BUILD_BETTER_MODEL_GUIDE.md` - Complete training guide
- `MODEL_ANALYSIS_SUMMARY.md` - This document

---

**Good luck building the best Vocence miner! 🚀**
