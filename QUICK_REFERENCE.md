# Vocence Mining Quick Reference Card

## 🎯 Goal
Score ≥ 0.90 on validator evaluations to win rewards in winner-takes-all subnet

---

## 📊 Scoring Breakdown (How Validators Judge You)

| Element | Weight | How It's Scored |
|---------|--------|-----------------|
| **Script** | 30% | `1 - WER` (Word Error Rate via Whisper) |
| **Naturalness** | 15% | GPT-4o pairwise vs source audio |
| **Gender** | 10% | Exact match (male/female/neutral) |
| **Speed** | 10% | Ordinal match (slow/normal/fast) |
| **Emotion** | 10% | Exact match (8 emotions) |
| **Age Group** | 10% | Ordinal match (child/young_adult/adult/senior) |
| **Pitch** | 5% | Ordinal match (low/mid/high) |
| **Accent** | 5% | Exact match (us/uk/au/in/neutral/other) |
| **Tone** | 5% | Exact match (6 tones) |
| **TOTAL** | **100%** | Must achieve ≥ 0.90 to "win" |

**Pass Threshold:** Score ≥ 0.90 = Win, < 0.90 = Loss

---

## 🏆 Current Top 2 Models

### Model 1: diegosilvabit904/vocence-tts-sft-v1 ⭐ TOP MINER

**Strategy:** Quality through sampling diversity
- Base: Qwen3-TTS T1 + ckpt-600
- Samples: **5 candidates** per request
- Selection: UTMOSv2 + Whisper composite scoring
- Time: 100-150s per request
- Pass Rate: **~90%** (estimated)

**Why It Wins:**
```python
# Best-of-5 gives 5 chances to pass
P(at least 1 ≥ 0.90) = 1 - (1 - p_single)^5
# Even if p_single = 0.7, best-of-5 = 0.997 (99.7%)
```

### Model 2: macminix/qwen3_voice_design_t2 🔥 CHALLENGER

**Strategy:** Quality through superior training
- Base: Qwen3-TTS T2 (gender-parity trained)
- Samples: **1 candidate** per request
- Selection: None
- Time: 15-30s per request
- Pass Rate: **~75%** (estimated)

**Advantages:**
- +2.8% UTMOS over T1
- Better gender × pitch accuracy
- 5x faster inference

---

## 🚀 How to Beat Them Both

### **Option A: Quick Win (Hybrid)** ⚡ RECOMMENDED

Combine T2 base + best-of-N sampling:

```bash
# 1. Clone T2 repo
git clone https://huggingface.co/macminix/qwen3_voice_design_t2

# 2. Replace miner.py with hybrid version
# (implements best-of-6 with UTMOSv2+Whisper scoring)

# 3. Deploy
chutes deploy your_hybrid:chute --accept-fee
```

**Expected:** 90-95% pass rate, 0.93-0.95 avg score

### **Option B: Advanced Training** 🎓

Train your own superior model:

1. **Data:** 50K+ clips (vs 11K baseline)
   - LibriTTS-R, VCTK, Common Voice, Emotional Speech
   - Gender-parity stratified per pitch bucket
   
2. **Training:** 
   - LoRA: r=32, alpha=64 (2x capacity)
   - 4 epochs, higher LR floor (0.25)
   - Multi-stage: quality → balance → edge cases

3. **Inference:**
   - Best-of-6 sampling
   - Optimized composite scoring

**Expected:** 95-98% pass rate, 0.95-0.97 avg score

---

## ⚙️ Implementation Checklist

### For Hybrid Model (Quick Win)

- [ ] Download T2 base model
- [ ] Implement best-of-N sampling (N=6)
- [ ] Add UTMOSv2 scorer (naturalness)
- [ ] Add faster-whisper scorer (WER)
- [ ] Implement instruction format conversion
- [ ] Add adaptive time budgeting
- [ ] Test locally with 100+ prompts
- [ ] Verify >90% pass rate
- [ ] Deploy to Chutes
- [ ] Register on subnet

### For Custom Training

- [ ] Collect 50K+ audio clips
- [ ] Annotate with GPT-4o-audio
- [ ] Balance gender/pitch/age/emotion
- [ ] Setup training environment (24GB+ GPU)
- [ ] Configure improved LoRA (r=32)
- [ ] Train for 4 epochs
- [ ] Evaluate checkpoints
- [ ] Select best via validation set
- [ ] Merge LoRA into base
- [ ] Add inference optimizations
- [ ] Test thoroughly
- [ ] Deploy

---

## 🔧 Critical Implementation Details

### Instruction Format Conversion

Validators send:
```
"gender: male | pitch: low | speed: fast | age_group: adult | 
 emotion: excited | tone: casual | accent: us"
```

You must convert to:
```
"An adult male speaker with a casual tone speaks quickly 
and excitedly at a low pitch, with an American accent."
```

Models are trained on natural language, not pipe format!

### Generation Parameters

**Recommended:**
```python
temperature = 0.85      # Lower = more consistent
top_p = 0.95            # Nucleus sampling
top_k = 50              # Limit choices
repetition_penalty = 1.05
max_new_tokens = 600    # Enough for 30s audio
do_sample = True        # Essential for diversity
```

### Quality Scoring (for best-of-N)

```python
def composite_score(wav, sr, text):
    # Naturalness (0-1 scale)
    utmos = utmosv2_model.predict(wav, sr) / 5.0
    
    # Script fidelity (0-1 scale)
    transcript = whisper.transcribe(wav)
    wer = word_error_rate(text, transcript)
    script = 1.0 - wer
    
    # Weighted composite
    return 0.35 * utmos + 0.65 * script
```

### Validity Checks

Before scoring, ensure:
```python
duration = len(wav) / sample_rate
assert 2.0 <= duration <= 29.5  # Seconds
assert rms(wav) >= 1e-3          # Not silent
assert max(abs(wav)) < 0.99      # Not clipping
```

---

## 📈 Performance Targets

To reliably win in a winner-takes-all subnet:

| Metric | Minimum | Recommended |
|--------|---------|-------------|
| Pass Rate (≥0.90) | 85% | **90%+** |
| Avg Composite Score | 0.91 | **0.93+** |
| Generation Time | <200s | **<150s** |
| Resource Usage | 24GB | 24GB |
| Uptime | 95% | **99%+** |

---

## 🎯 Testing Protocol

Before deploying to mainnet:

```bash
# 1. Local evaluation
python test_models.py

# 2. Extended testing (100+ prompts)
python extensive_eval.py --prompts 100

# 3. Check metrics
# - Pass rate > 90%
# - Avg score > 0.93
# - p95 latency < 200s
# - No crashes/errors

# 4. Gradual rollout
# - Deploy to testnet first (if available)
# - Monitor for 24h
# - Check validator scores
# - If stable, deploy to mainnet
```

---

## 💡 Pro Tips

1. **Sampling > Training (for now)**
   - Model 1 made NO training changes to T1
   - Pure inference optimization got #1 rank
   - Start with sampling, train later

2. **Consistency Beats Average**
   - 90% pass threshold = consistency critical
   - Better to score 0.91 always than 0.95 sometimes

3. **Monitor Validator Feedback**
   - Track your scores in validator buckets
   - Identify weak areas (which traits fail?)
   - Iterate on those specific cases

4. **Resource Management**
   - Keep GPU usage smooth (no OOM)
   - Implement proper timeout handling
   - Cache models in memory (warmup)

5. **Eligibility Requirements**
   - Need 41+ evals in 3+ validator buckets
   - Takes time to accumulate
   - Be patient, focus on quality

---

## 🚨 Common Pitfalls

❌ **Don't ignore instruction conversion**
- Pipe format ≠ model training format
- Will hurt trait adherence scores

❌ **Don't skip quality selection**
- Single sample has high variance
- Best-of-N is proven to work

❌ **Don't deploy without testing**
- 100+ diverse prompts minimum
- Check ALL trait combinations

❌ **Don't modify wrapper improperly**
- Only 4 approved variables
- Owner checks wrapper integrity

❌ **Don't expect instant profits**
- Need 41+ evals to compete
- Winner-takes-all = only #1 earns

---

## 📚 Key Files Created

1. **`test_models.py`** - Evaluate both top models locally
2. **`BUILD_BETTER_MODEL_GUIDE.md`** - Complete training guide
3. **`MODEL_ANALYSIS_SUMMARY.md`** - Deep technical analysis
4. **`QUICK_REFERENCE.md`** - This document

---

## 🔗 Essential Links

- **Vocence Docs:** [docs/](docs/)
- **Model 1:** https://huggingface.co/diegosilvabit904/vocence-tts-sft-v1
- **Model 2:** https://huggingface.co/macminix/qwen3_voice_design_t2
- **Base Model:** https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign
- **Qwen-TTS Package:** https://pypi.org/project/qwen-tts/

---

## 🎬 Quick Start Command

```bash
# Evaluate both models and generate comparison report
cd /home/ubuntu/workspace/vocence
python test_models.py

# Read the results
cat evaluation_results/Model_1_Top_Miner_results.json
cat evaluation_results/Model_2_T2_Candidate_results.json

# Build your winning strategy
cat BUILD_BETTER_MODEL_GUIDE.md
```

---

**Remember:** The current top miner wins through **sampling strategy**, not superior training. Your fastest path to victory is combining T2's better base model with best-of-N sampling. Train custom models only after proving the hybrid approach works.

**Good luck! 🚀**
