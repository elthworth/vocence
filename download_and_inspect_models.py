"""
Download and inspect both TTS models from HuggingFace
"""
import os
from huggingface_hub import snapshot_download, list_repo_files
import json

# Model repositories
TOP_MODEL = "ratrys/sft-tts-800"
BASE_MODEL = "magma90909/vocence_miner_v3"

# Create a directory for models
MODELS_DIR = "downloaded_models"
os.makedirs(MODELS_DIR, exist_ok=True)

def download_model(repo_id, local_dir):
    """Download a model from HuggingFace"""
    print(f"\n{'='*60}")
    print(f"Downloading {repo_id}...")
    print(f"{'='*60}")
    
    try:
        # List files in the repo first
        files = list_repo_files(repo_id)
        print(f"\nFiles in repository:")
        for file in files:
            print(f"  - {file}")
        
        # Download the model
        local_path = snapshot_download(
            repo_id=repo_id,
            local_dir=local_dir,
            local_dir_use_symlinks=False
        )
        print(f"\n✓ Downloaded to: {local_path}")
        return local_path
    except Exception as e:
        print(f"✗ Error downloading {repo_id}: {str(e)}")
        return None

def inspect_model_directory(model_path, model_name):
    """Inspect the downloaded model directory"""
    print(f"\n{'='*60}")
    print(f"Inspecting {model_name}")
    print(f"{'='*60}")
    
    if not os.path.exists(model_path):
        print(f"✗ Model path does not exist: {model_path}")
        return
    
    # List all files and their sizes
    print(f"\nDirectory structure:")
    for root, dirs, files in os.walk(model_path):
        level = root.replace(model_path, '').count(os.sep)
        indent = ' ' * 2 * level
        print(f"{indent}{os.path.basename(root)}/")
        subindent = ' ' * 2 * (level + 1)
        for file in files:
            file_path = os.path.join(root, file)
            file_size = os.path.getsize(file_path)
            size_mb = file_size / (1024 * 1024)
            print(f"{subindent}{file} ({size_mb:.2f} MB)")
    
    # Check for common model files
    config_file = os.path.join(model_path, "config.json")
    if os.path.exists(config_file):
        print(f"\n{'='*40}")
        print("CONFIG.JSON CONTENTS:")
        print(f"{'='*40}")
        with open(config_file, 'r') as f:
            config = json.load(f)
            print(json.dumps(config, indent=2))
    
    # Check for README
    readme_file = os.path.join(model_path, "README.md")
    if os.path.exists(readme_file):
        print(f"\n{'='*40}")
        print("README.MD CONTENTS:")
        print(f"{'='*40}")
        with open(readme_file, 'r') as f:
            readme_content = f.read()
            # Print first 2000 characters
            print(readme_content[:2000])
            if len(readme_content) > 2000:
                print(f"\n... (truncated, total length: {len(readme_content)} characters)")
    
    # Check for model card
    model_card = os.path.join(model_path, "model_card.md")
    if os.path.exists(model_card):
        print(f"\n{'='*40}")
        print("MODEL_CARD.MD CONTENTS:")
        print(f"{'='*40}")
        with open(model_card, 'r') as f:
            print(f.read())

def main():
    print("="*60)
    print("MODEL DOWNLOAD AND INSPECTION TOOL")
    print("="*60)
    
    # Download both models
    top_model_path = os.path.join(MODELS_DIR, "top_model")
    base_model_path = os.path.join(MODELS_DIR, "base_model")
    
    print("\n[1/2] Downloading top performing model...")
    top_path = download_model(TOP_MODEL, top_model_path)
    
    print("\n[2/2] Downloading base model...")
    base_path = download_model(BASE_MODEL, base_model_path)
    
    # Inspect both models
    if top_path:
        inspect_model_directory(top_path, "TOP MODEL (ratrys/sft-tts-800)")
    
    if base_path:
        inspect_model_directory(base_path, "BASE MODEL (magma90909/vocence_miner_v3)")
    
    print(f"\n{'='*60}")
    print("DOWNLOAD AND INSPECTION COMPLETE")
    print(f"{'='*60}")
    print(f"\nModels saved in: {os.path.abspath(MODELS_DIR)}")

if __name__ == "__main__":
    main()
