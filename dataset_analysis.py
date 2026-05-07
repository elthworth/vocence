"""
Comprehensive analysis of available TTS datasets for fine-tuning
Focus: Naturalness, Gender accuracy, and Emotion expression
"""
from typing import Dict, List, Any

# Comprehensive dataset analysis for TTS fine-tuning
DATASETS = {
    "libritts_r": {
        "name": "LibriTTS-R",
        "hf_id": "cdminix/libritts-r-aligned",
        "size": "~585 hours",
        "speakers": "2,456 speakers",
        "sample_rate": "24kHz",
        "quality": "High quality audiobook recordings, cleaned and enhanced",
        "gender_balance": "Balanced (male/female)",
        "emotion_coverage": "Limited (mostly neutral reading)",
        "naturalness": "★★★★★ (Professional audiobook narration)",
        "pros": [
            "Highest audio quality in the open domain",
            "Multi-speaker with consistent quality",
            "Well-aligned text-audio pairs",
            "Perfect for improving naturalness and prosody",
            "Good gender representation"
        ],
        "cons": [
            "Limited emotional range (mostly neutral/narrative)",
            "Formal reading style may lack conversational naturalness",
            "May not help with emotion accuracy"
        ],
        "best_for": ["naturalness", "prosody", "clarity"],
        "fine_tuning_priority": "HIGH - Best for naturalness improvements"
    },
    
    "emov_db": {
        "name": "EmoV-DB",
        "hf_id": "speechcolab/emov-db",
        "size": "~11 hours",
        "speakers": "5 speakers (3 female, 2 male)",
        "sample_rate": "16kHz",
        "quality": "Clean studio recordings",
        "gender_balance": "Slight female bias (3F/2M)",
        "emotion_coverage": "★★★★★ (5 emotions: amused, angry, disgusted, neutral, sleepy)",
        "naturalness": "★★★★ (Studio quality, somewhat acted)",
        "pros": [
            "Explicitly labeled emotions with clear expression",
            "Clean, controlled recordings",
            "Same text read in multiple emotions",
            "Perfect for emotion classification improvements"
        ],
        "cons": [
            "Small dataset size",
            "Limited speakers",
            "Emotions may sound slightly acted",
            "16kHz (lower than Qwen3-TTS native 24kHz)"
        ],
        "best_for": ["emotion", "gender"],
        "fine_tuning_priority": "HIGH - Best for emotion accuracy"
    },
    
    "expresso": {
        "name": "Expresso",
        "hf_id": "ylacombe/expresso",
        "size": "~40 hours",
        "speakers": "2 speakers (1 female, 1 male)",
        "sample_rate": "24kHz",
        "quality": "High-quality studio recordings",
        "gender_balance": "Perfectly balanced (1F/1M)",
        "emotion_coverage": "★★★★★ (Rich prosodic and emotional variation)",
        "naturalness": "★★★★★ (Diverse natural speaking styles)",
        "pros": [
            "Wide range of speaking styles (reading, conversational, storytelling, etc.)",
            "Rich emotional and prosodic annotations",
            "Perfect for gender-balanced training",
            "Native 24kHz matches Qwen3-TTS",
            "Includes metadata: pitch, energy, speaking rate"
        ],
        "cons": [
            "Only 2 speakers (limited speaker diversity)",
            "Moderate size"
        ],
        "best_for": ["naturalness", "emotion", "gender", "prosody"],
        "fine_tuning_priority": "VERY HIGH - Best overall for all three weaknesses"
    },
    
    "meld": {
        "name": "MELD (Multimodal EmotionLines Dataset)",
        "hf_id": "sanchit-gandhi/meld",
        "size": "~13 hours",
        "speakers": "Multiple (from TV show)",
        "sample_rate": "16kHz (original), can be upsampled",
        "quality": "TV audio quality (variable)",
        "gender_balance": "Mixed",
        "emotion_coverage": "★★★★★ (7 emotions: anger, disgust, fear, joy, neutral, sadness, surprise)",
        "naturalness": "★★★★★ (Real conversational speech)",
        "pros": [
            "Authentic conversational emotion",
            "Natural dialogue context",
            "Multiple speakers with personality",
            "Real-world emotion expression"
        ],
        "cons": [
            "Variable audio quality",
            "Background noise/music",
            "Need preprocessing",
            "16kHz audio"
        ],
        "best_for": ["emotion", "conversational_naturalness"],
        "fine_tuning_priority": "MEDIUM - Good for emotion but needs preprocessing"
    },
    
    "voxceleb": {
        "name": "VoxCeleb1 & 2",
        "hf_id": "Benjamin-Walker/voxceleb",
        "size": "~2,000 hours",
        "speakers": "7,000+ speakers",
        "sample_rate": "16kHz",
        "quality": "In-the-wild YouTube videos (variable)",
        "gender_balance": "Reasonable mix",
        "emotion_coverage": "Varied (natural, uncontrolled)",
        "naturalness": "★★★★★ (Real speech)",
        "pros": [
            "Massive speaker diversity",
            "Real-world speaking conditions",
            "Many accents and styles",
            "Good gender diversity"
        ],
        "cons": [
            "No emotion annotations",
            "Variable quality",
            "Background noise",
            "Need transcription",
            "May introduce unwanted variability"
        ],
        "best_for": ["speaker_diversity", "robustness"],
        "fine_tuning_priority": "LOW - Too noisy for current goals"
    },
    
    "jenny": {
        "name": "Jenny Dataset",
        "hf_id": "reach-vb/jenny-tts-dataset",
        "size": "~25 hours",
        "speakers": "1 speaker (female)",
        "sample_rate": "22.05kHz",
        "quality": "Professional studio quality",
        "gender_balance": "Female only",
        "emotion_coverage": "Limited (mostly neutral)",
        "naturalness": "★★★★★ (Exceptionally natural)",
        "pros": [
            "Extremely high quality",
            "Very natural speaking style",
            "Clean recordings",
            "Good for naturalness"
        ],
        "cons": [
            "Single speaker (female only)",
            "Limited emotion range",
            "Won't help with gender accuracy",
            "Slightly lower than 24kHz"
        ],
        "best_for": ["naturalness", "female_voice"],
        "fine_tuning_priority": "MEDIUM - Good for naturalness but gender-imbalanced"
    },
    
    "ljspeech": {
        "name": "LJSpeech",
        "hf_id": "lj_speech",
        "size": "~24 hours",
        "speakers": "1 speaker (female)",
        "sample_rate": "22.05kHz",
        "quality": "High-quality audiobook",
        "gender_balance": "Female only",
        "emotion_coverage": "Very limited (neutral reading)",
        "naturalness": "★★★★ (Clear but monotonous)",
        "pros": [
            "Clean, consistent quality",
            "Well-aligned text",
            "Widely used benchmark"
        ],
        "cons": [
            "Single speaker",
            "Very limited prosody",
            "No emotion variation",
            "Won't help with gender or emotion"
        ],
        "best_for": ["clarity", "alignment"],
        "fine_tuning_priority": "LOW - Not suitable for current goals"
    },
    
    "ravdess": {
        "name": "RAVDESS",
        "hf_id": "LIMUNIMI/RAVDESS",
        "size": "~3 hours",
        "speakers": "24 speakers (12 female, 12 male)",
        "sample_rate": "48kHz (can downsample to 24kHz)",
        "quality": "Studio quality",
        "gender_balance": "★★★★★ Perfectly balanced (12F/12M)",
        "emotion_coverage": "★★★★★ (8 emotions: calm, happy, sad, angry, fearful, surprise, disgust, neutral)",
        "naturalness": "★★★ (Acted emotions, may sound theatrical)",
        "pros": [
            "Perfect gender balance",
            "Rich emotion annotations with intensity levels",
            "High sample rate (48kHz)",
            "Same sentence in multiple emotions",
            "Multiple takes per emotion"
        ],
        "cons": [
            "Small dataset",
            "Emotions are acted (theatrical)",
            "May sound unnatural for TTS",
            "Limited vocabulary"
        ],
        "best_for": ["emotion", "gender_balance"],
        "fine_tuning_priority": "MEDIUM - Good for gender/emotion but small and theatrical"
    },
    
    "common_voice": {
        "name": "Common Voice",
        "hf_id": "mozilla-foundation/common_voice_16_1",
        "size": "~3,000+ hours (English)",
        "speakers": "Tens of thousands",
        "sample_rate": "48kHz",
        "quality": "Variable (crowdsourced)",
        "gender_balance": "Mixed (can filter)",
        "emotion_coverage": "Mostly neutral",
        "naturalness": "★★★ (Variable, often stiff reading)",
        "pros": [
            "Massive scale",
            "Huge speaker diversity",
            "Multi-accent coverage",
            "Gender/age metadata available"
        ],
        "cons": [
            "Variable quality",
            "Many recordings sound unnatural (reading)",
            "Background noise",
            "No emotion annotations",
            "Requires heavy filtering"
        ],
        "best_for": ["accent_diversity", "speaker_coverage"],
        "fine_tuning_priority": "LOW - Too variable, won't improve naturalness/emotion"
    },
    
    "gigaspeech": {
        "name": "GigaSpeech",
        "hf_id": "speechcolab/gigaspeech",
        "size": "~10,000 hours",
        "speakers": "Thousands",
        "sample_rate": "16kHz",
        "quality": "Variable (audiobooks, podcasts, YouTube)",
        "gender_balance": "Mixed",
        "emotion_coverage": "Natural variation",
        "naturalness": "★★★★ (Podcast style can be very natural)",
        "pros": [
            "Massive scale",
            "Diverse speaking styles",
            "Conversational and narrative styles",
            "Good for learning naturalness"
        ],
        "cons": [
            "16kHz only",
            "Variable quality",
            "No emotion annotations",
            "Computational cost",
            "Need filtering for quality"
        ],
        "best_for": ["naturalness", "speaking_style_diversity"],
        "fine_tuning_priority": "MEDIUM - Good for naturalness with proper filtering"
    }
}


def get_recommended_datasets_by_goal() -> Dict[str, List[str]]:
    """Return recommended datasets prioritized by improvement goal"""
    return {
        "improve_naturalness": [
            "expresso",  # Best overall
            "libritts_r",  # High quality prosody
            "gigaspeech",  # Diverse natural styles (filtered)
            "jenny",  # Excellent quality but single speaker
        ],
        "improve_emotion": [
            "expresso",  # Rich emotional annotations
            "emov_db",  # Explicit emotion labels
            "ravdess",  # Emotion-focused
            "meld",  # Natural conversational emotion
        ],
        "improve_gender": [
            "expresso",  # Perfect 1F/1M balance
            "ravdess",  # Perfect 12F/12M balance
            "libritts_r",  # Good gender coverage
            "emov_db",  # 3F/2M reasonable
        ],
        "all_three_combined": [
            "expresso",  # BEST - hits all three goals
            "libritts_r + emov_db",  # Combo: naturalness + emotion
            "libritts_r + ravdess",  # Combo: naturalness + emotion/gender
        ]
    }


def generate_training_strategy() -> str:
    """Generate recommended fine-tuning strategy"""
    return """
RECOMMENDED FINE-TUNING STRATEGY FOR IMPROVING NATURALNESS, GENDER & EMOTION
============================================================================

PHASE 1: PRIMARY DATASET - Expresso (HIGHEST PRIORITY)
------------------------------------------------------
Dataset: ylacombe/expresso
Duration: 40 hours
Why: Hits all three weaknesses simultaneously
- Exceptional naturalness with diverse speaking styles
- Perfect gender balance (1F/1M)
- Rich emotional and prosodic variation
- Native 24kHz (matches model)
- Includes pitch, energy, speaking rate metadata

Training approach:
- LoRA rank: 8-16 (start with 8 to match original)
- Learning rate: 1e-4 to 5e-4
- Batch size: 4-8 (depending on GPU)
- Epochs: 2-4
- Target modules: Same as original (q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj)
- Special focus: Attention layers (for prosody/naturalness)


PHASE 2: SUPPLEMENTARY - LibriTTS-R (for naturalness boost)
-----------------------------------------------------------
Dataset: cdminix/libritts-r-aligned
Duration: 585 hours (use subset: ~50-100 hours)
Why: Professional-quality prosody and naturalness
- Highest audio quality available
- Excellent for prosody learning
- Good gender balance

Training approach:
- Continue from Phase 1 checkpoint
- Smaller learning rate: 5e-5 to 1e-4
- 1-2 epochs
- Focus on solidifying naturalness gains


PHASE 3: EMOTION-SPECIFIC - EmoV-DB (for emotion accuracy)
----------------------------------------------------------
Dataset: speechcolab/emov-db
Duration: 11 hours
Why: Explicit emotion labels with clear distinctions
- Same text in multiple emotions
- Direct emotion-label training
- Clean recordings

Training approach:
- Continue from Phase 2 checkpoint
- Learning rate: 5e-5
- 3-5 epochs (small dataset, more epochs OK)
- May help model distinguish emotions better


ALTERNATIVE: COMBINED APPROACH
------------------------------
Mix datasets in proportion:
- 50% Expresso (balanced focus)
- 30% LibriTTS-R (naturalness)
- 20% EmoV-DB (emotion)

Single training run with mixed data.

Epochs: 3-4
Learning rate: 1e-4 with cosine decay

This approach may be more efficient but Phase-wise training allows
you to evaluate improvement at each stage.


DATA PREPROCESSING REQUIREMENTS
--------------------------------
1. Resample all audio to 24kHz (match model native rate)
2. Normalize audio levels (-23 LUFS target)
3. Remove silence at start/end
4. Filter out:
   - Clips < 1 second or > 30 seconds
   - Clips with excessive background noise
   - Clips with poor transcription alignment
5. Create instruction templates matching Vocence format:
   "gender: {gender} | pitch: {pitch} | speed: {speed} | age_group: {age_group} | emotion: {emotion} | tone: {tone} | accent: {accent}"
6. Balance dataset by:
   - Gender (50/50 male/female)
   - Emotion distribution (uniform if possible)
   - Avoid over-representation of neutral emotion


EVALUATION METRICS
------------------
After each phase, test on held-out set with:
1. UTMOSv2 naturalness scores
2. Gender classification accuracy (separate classifier)
3. Emotion classification accuracy (separate classifier)
4. WER (script accuracy - should not degrade)
5. Human listening tests

Target improvements:
- Naturalness: +10-15% (UTMOSv2 score increase)
- Gender accuracy: +15-20%
- Emotion accuracy: +15-20%
- Script accuracy: maintain within -2%


COMPUTATIONAL REQUIREMENTS
---------------------------
- GPU: 1x A100 (40GB) or 2x RTX 4090
- Training time per phase: 12-24 hours
- Storage: ~100GB for datasets + checkpoints
- RAM: 64GB+ recommended for data loading
"""


def print_analysis():
    """Print comprehensive dataset analysis"""
    print("="*80)
    print("TTS FINE-TUNING DATASET ANALYSIS")
    print("="*80)
    print("\nGoal: Improve NATURALNESS (15% weight), GENDER (10% weight), and EMOTION (10% weight)\n")
    
    # Print priority sorted datasets
    priority_order = ["VERY HIGH", "HIGH", "MEDIUM", "LOW"]
    for priority in priority_order:
        datasets_at_priority = [(name, info) for name, info in DATASETS.items() 
                               if info["fine_tuning_priority"].startswith(priority)]
        
        if datasets_at_priority:
            print(f"\n{'='*80}")
            print(f"PRIORITY: {priority}")
            print(f"{'='*80}\n")
            
            for name, info in datasets_at_priority:
                print(f"📊 {info['name']}")
                print(f"   HuggingFace: {info['hf_id']}")
                print(f"   Size: {info['size']}")
                print(f"   Speakers: {info['speakers']}")
                print(f"   Sample Rate: {info['sample_rate']}")
                print(f"   Gender Balance: {info['gender_balance']}")
                print(f"   Emotion Coverage: {info['emotion_coverage']}")
                print(f"   Naturalness: {info['naturalness']}")
                print(f"   Best For: {', '.join(info['best_for'])}")
                print(f"\n   ✅ PROS:")
                for pro in info['pros']:
                    print(f"      • {pro}")
                print(f"\n   ❌ CONS:")
                for con in info['cons']:
                    print(f"      • {con}")
                print(f"\n   💡 Priority: {info['fine_tuning_priority']}")
                print("\n" + "-"*80 + "\n")
    
    # Print recommendations by goal
    print("\n" + "="*80)
    print("RECOMMENDATIONS BY IMPROVEMENT GOAL")
    print("="*80 + "\n")
    
    recommendations = get_recommended_datasets_by_goal()
    for goal, datasets in recommendations.items():
        print(f"🎯 {goal.replace('_', ' ').upper()}:")
        for i, dataset in enumerate(datasets, 1):
            print(f"   {i}. {dataset}")
        print()
    
    # Print training strategy
    print("="*80)
    print(generate_training_strategy())


if __name__ == "__main__":
    print_analysis()
