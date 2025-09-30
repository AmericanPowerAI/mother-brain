# dataset_manager.py
import os
import requests
import json
from pathlib import Path
import zipfile
import tempfile

class DatasetManager:
    def __init__(self):
        self.data_dir = Path("training_data")
        self.data_dir.mkdir(exist_ok=True)
        
        # Dataset URLs - these are safe, manageable datasets
        self.datasets = {
            "tinystories": "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-28k.zip",
            "wikitext_tiny": "https://huggingface.co/datasets/wikitext/resolve/main/data/wikitext-103-raw-v1.zip"
        }
    
    def download_dataset(self, dataset_name):
        """Download dataset to training_data/ folder"""
        if dataset_name not in self.datasets:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        url = self.datasets[dataset_name]
        filename = self.data_dir / f"{dataset_name}.zip"
        
        print(f"📥 Downloading {dataset_name} from {url}...")
        
        # Download the file
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(filename, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        # Extract if it's a zip file
        if filename.suffix == '.zip':
            print(f"📦 Extracting {filename}...")
            with zipfile.ZipFile(filename, 'r') as zip_ref:
                zip_ref.extractall(self.data_dir / dataset_name)
        
        print(f"✅ Successfully downloaded {dataset_name}")
        return filename
    
    def load_into_mother(self, mother_instance):
        """Load downloaded datasets into MOTHER AI's knowledge base"""
        knowledge_updates = {}
        
        # Load TinyStories (conversational data)
        stories_path = self.data_dir / "tinystories" / "TinyStoriesV2-GPT4-28k.txt"
        if stories_path.exists():
            print("📚 Loading TinyStories into knowledge base...")
            with open(stories_path, 'r', encoding='utf-8') as f:
                content = f.read()[:50000]  # Load first 50KB
                knowledge_updates["TRAINING_STORIES"] = content
                print(f"✅ Loaded {len(content)} characters from TinyStories")
        
        # Load WikiText (general knowledge)
        wiki_path = self.data_dir / "wikitext_tiny"
        if wiki_path.exists():
            # Look for text files in the extracted folder
            for file in wiki_path.rglob("*.txt"):
                try:
                    with open(file, 'r', encoding='utf-8') as f:
                        content = f.read()[:30000]  # Load first 30KB per file
                        key = f"WIKIPEDIA_{file.stem.upper()}"
                        knowledge_updates[key] = content
                        print(f"✅ Loaded {len(content)} characters from {file.name}")
                except Exception as e:
                    print(f"⚠️ Could not load {file}: {e}")
        
        # Update mother's knowledge
        if knowledge_updates:
            mother_instance.knowledge.update(knowledge_updates)
            mother_instance._save_to_github()  # Persist to GitHub
            
        return len(knowledge_updates)
    
    def get_available_datasets(self):
        """List available datasets for download"""
        return list(self.datasets.keys())
    
    def cleanup(self):
        """Remove downloaded datasets (optional)"""
        import shutil
        if self.data_dir.exists():
            shutil.rmtree(self.data_dir)
            print("🧹 Cleaned up training data")

# Simple function to check if setup is needed
def needs_training_data():
    """Check if training data needs to be downloaded"""
    data_dir = Path("training_data")
    return not data_dir.exists() or not any(data_dir.iterdir())

if __name__ == "__main__":
    # Test the dataset manager
    dm = DatasetManager()
    print("Available datasets:", dm.get_available_datasets())
