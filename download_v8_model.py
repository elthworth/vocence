"""
Download and inspect the new top model (v8) and compare with v3
"""
import os
import json
from huggingface_hub import snapshot_download, list_repo_files
from pathlib import Path

# Model repositories
V8_MODEL = "magma90909/vocence_miner_v8"
V3_MODEL = "magma90909/vocence_miner_v3"

# Directory for model
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
        return None
    
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
    config_data = None
    if os.path.exists(config_file):
        print(f"\n{'='*40}")
        print("CONFIG.JSON CONTENTS:")
        print(f"{'='*40}")
        with open(config_file, 'r') as f:
            config_data = json.load(f)
            print(json.dumps(config_data, indent=2))
    
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
    
    # Check for merge_info or training_info
    merge_info_file = os.path.join(model_path, "merge_info.json")
    if os.path.exists(merge_info_file):
        print(f"\n{'='*40}")
        print("MERGE_INFO.JSON CONTENTS:")
        print(f"{'='*40}")
        with open(merge_info_file, 'r') as f:
            merge_info = json.load(f)
            print(json.dumps(merge_info, indent=2))
    
    training_info_file = os.path.join(model_path, "training_info.json")
    if os.path.exists(training_info_file):
        print(f"\n{'='*40}")
        print("TRAINING_INFO.JSON CONTENTS:")
        print(f"{'='*40}")
        with open(training_info_file, 'r') as f:
            training_info = json.load(f)
            print(json.dumps(training_info, indent=2))
    
    return config_data

def compare_configs(v3_config, v8_config):
    """Compare configurations between v3 and v8"""
    print(f"\n{'='*60}")
    print("CONFIGURATION COMPARISON: v3 vs v8")
    print(f"{'='*60}\n")
    
    # Check for new or different keys
    v3_keys = set(v3_config.keys()) if v3_config else set()
    v8_keys = set(v8_config.keys()) if v8_config else set()
    
    new_in_v8 = v8_keys - v3_keys
    removed_in_v8 = v3_keys - v8_keys
    common_keys = v3_keys & v8_keys
    
    if new_in_v8:
        print(f"✨ NEW in v8:")
        for key in sorted(new_in_v8):
            print(f"  + {key}: {v8_config[key]}")
        print()
    
    if removed_in_v8:
        print(f"🗑️  REMOVED from v8:")
        for key in sorted(removed_in_v8):
            print(f"  - {key}: {v3_config[key]}")
        print()
    
    print(f"📊 CHANGED values:")
    for key in sorted(common_keys):
        if v3_config[key] != v8_config[key]:
            print(f"  {key}:")
            print(f"    v3: {v3_config[key]}")
            print(f"    v8: {v8_config[key]}")
    
    # Check model architecture differences
    if 'talker_config' in v3_config and 'talker_config' in v8_config:
        print(f"\n{'='*40}")
        print("TALKER CONFIG COMPARISON:")
        print(f"{'='*40}")
        
        v3_talker = v3_config['talker_config']
        v8_talker = v8_config['talker_config']
        
        important_keys = [
            'hidden_size', 'num_hidden_layers', 'num_attention_heads',
            'num_key_value_heads', 'intermediate_size', 'dtype'
        ]
        
        for key in important_keys:
            v3_val = v3_talker.get(key, 'N/A')
            v8_val = v8_talker.get(key, 'N/A')
            changed = "✓" if v3_val == v8_val else "→"
            print(f"  {changed} {key}: {v3_val} {'→ ' + str(v8_val) if v3_val != v8_val else ''}")

def main():
    print("="*60)
    print("MODEL v8 DOWNLOAD AND COMPARISON TOOL")
    print("="*60)
    
    # Download v8 model
    v8_path = os.path.join(MODELS_DIR, "v8_model")
    print("\n[1/2] Downloading v8 model (new top model)...")
    v8_local = download_model(V8_MODEL, v8_path)
    
    # Check if v3 exists
    v3_path = os.path.join(MODELS_DIR, "base_model")
    if not os.path.exists(v3_path):
        print("\n[2/2] v3 model not found, downloading...")
        v3_local = download_model(V3_MODEL, v3_path)
    else:
        print("\n[2/2] v3 model already exists, skipping download...")
        v3_local = v3_path
    
    # Inspect v8
    v8_config = None
    if v8_local:
        v8_config = inspect_model_directory(v8_local, "V8 MODEL (magma90909/vocence_miner_v8)")
    
    # Inspect v3
    v3_config = None
    if v3_local:
        v3_config = inspect_model_directory(v3_local, "V3 MODEL (magma90909/vocence_miner_v3)")
    
    # Compare
    if v3_config and v8_config:
        compare_configs(v3_config, v8_config)
    
    print(f"\n{'='*60}")
    print("DOWNLOAD AND COMPARISON COMPLETE")
    print(f"{'='*60}")
    print(f"\nModels saved in: {os.path.abspath(MODELS_DIR)}")
    print(f"  - v8 (new top): {v8_path}")
    print(f"  - v3 (previous): {v3_path}")

if __name__ == "__main__":
    main()
