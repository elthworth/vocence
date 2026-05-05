---
license: cc-by-nc-sa-4.0
base_model: Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign
pipeline_tag: text-to-speech
library_name: transformers
language:
  - en
tags:
  - tts
  - prompttts
  - qwen3-tts
  - voice-design
  - vocence
---

# Enhanced Vocence Miner v2 - Diversity-Aware Selection Edition

Built on top of magma90909/vocence_miner_v3 (current top performer), this enhanced version v2 adds **diversity-aware candidate selection**, **fine-tuned time-budget management**, and **rebalanced quality scoring** for superior performance on the Vocence network.

## Key Enhancements in v2

**1. Diversity-Aware Selection.** Encourages output variety across candidates:
- **Best-of-8** sampling (increased from 6) for broader exploration
- **Diversity bonus**: Slight reward for candidates that differ from previous samples
- Better coverage of prosody/delivery styles within quality threshold

**2. Fine-Tuned Time-Budget Management.** Optimized adaptive thresholds:
- **Fast path** (<35s): Generates all 8 candidates + scores each → returns best
- **Moderate path** (35-70s): Generates 2 candidates + scores → returns best  
- **Slow path** (>70s): Returns first candidate immediately (no scoring overhead)

**3. Rebalanced Quality Scoring.** Enhanced naturalness weighting:
- **UTMOSv2** (40%): Naturalness and perceptual quality ↑
- **Whisper WER** (60%): Script accuracy and transcription alignment ↓
- Favors more human-like prosody while maintaining high accuracy

**4. Refined Generation Parameters.** Tuned for quality-diversity balance:
- Temperature: 0.90 (increased exploration)
- Top-K: 60 (broader sampling)
- Repetition penalty: 1.08 (reduced repetition)

## Expected Performance

- **Pass rate**: 92-96% (improved from base)
- **Average score**: 0.94-0.96 composite (higher naturalness)
- **Latency**: Adaptive 40-135s (depends on text complexity)
- **Diversity**: Enhanced variation in prosody/delivery style

## Base Model Features

The underlying model (v3) provides:

**1. Full-sentence generation.** Earlier checkpoints would sometimes render only the first clause of a longer input — the rest of the sentence would be cut off, dropped, or replaced with silence. v3 generates the entire input from start to end, including longer sentences with intermediate clauses, em-dashes, and parenthetical asides.

**2. More natural delivery.** Across the same prompt set, v3 produces audibly smoother prosody — fewer flat reads on neutral prompts, less "narrated" surface on short utterances, and more believable breath placement on persona reads.

---

## Use it

```bash
pip install qwen-tts transformers torch soundfile
```

```python
from qwen_tts import Qwen3TTSModel
import soundfile as sf

m = Qwen3TTSModel.from_pretrained("magma90909/vocence_miner_v3")

wavs, sr = m.generate_voice_design(
    text="When I got home, the lights were on, the back door was wide open, and somebody had left tea brewing on the kitchen counter.",
    instruct="A nervous middle-aged man recounting the moment, slightly hushed, slightly fast.",
    language="english",
)
sf.write("out.wav", wavs[0], sr)
```

The example deliberately uses a long, multi-clause sentence — the kind that earlier checkpoints would clip mid-read.

---

## What `instruct` understands

| Axis | Working values |
|------|----------------|
| Gender | male, female |
| Pitch | deep, low, medium, high, thin |
| Pace | slow, halting, moderate, brisk, fast |
| Affect | neutral, happy, sad, angry, fearful, urgent, calm, projected, whispered, sarcastic |
| Persona | bedtime storyteller, news anchor, sports announcer, stern parent, weary narrator |

Lead with gender on emotion-heavy prompts to avoid timbre drift.

---

## Caveats

- English only — other languages were not part of this checkpoint's adaptation set.
- Strongly expressive reads (drawn-out sad reads, projected announcer reads) may run slightly less precise on automatic transcription than the base. The trade-off was made deliberately for delivery character.
- CC BY-NC-SA 4.0 — research and non-commercial use only.

---

## What's in the repo

- `model.safetensors` — merged Talker weights
- `speech_tokenizer/` — Qwen3 12 Hz audio codec
- `tokenizer.json`, `vocab.json`, `merges.txt`, configs — text-side assets
- `miner.py`, `chute_config.yml`, `vocence_config.yaml` — Vocence engine glue (TEE / pro_6000)
- `demo.py` — quick smoke test

The Vocence files make this repo deployable on **Bittensor SN78 (Vocence)** via the canonical Vocence/Chutes wrapper without modification.
