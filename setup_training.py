#!/usr/bin/env python3
"""
setup_training.py - Run this once to download training data for MOTHER AI
This downloads Wikipedia-style knowledge and conversation data to train your AI
"""

import os
import sys
from pathlib import Path

# Add current directory to path so we can import dataset_manager
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from dataset_manager import DatasetManager
except ImportError:
    print("❌ Error: Could not import dataset_manager. Make sure it's in the same directory.")
    sys.exit(1)

def main():
    print("🚀 MOTHER AI Training Data Setup")
    print("=" * 50)
    print("This will download datasets to train MOTHER AI with Wikipedia-level knowledge")
    print("Datasets will be saved to: training_data/")
    print("This may take a few minutes depending on your internet connection...")
    print()
    
    # Create dataset manager
    dm = DatasetManager()
    
    print("📋 Available datasets:")
    for i, dataset in enumerate(dm.get_available_datasets(), 1):
        print(f"   {i}. {dataset}")
    print()
    
    # Download essential datasets
    datasets_to_download = ["tinystories", "wikitext_tiny"]
    
    success_count = 0
    for dataset in datasets_to_download:
        try:
            print(f"📥 Downloading {dataset}...")
            result = dm.download_dataset(dataset)
            print(f"✅ Successfully downloaded {dataset}")
            success_count += 1
        except Exception as e:
            print(f"❌ Failed to download {dataset}: {e}")
            print("   This dataset will be skipped, but MOTHER AI will still work.")
    
    print()
    print("=" * 50)
    
    if success_count > 0:
        print(f"🎉 Successfully downloaded {success_count}/{len(datasets_to_download)} datasets!")
        print()
        print("📚 What was downloaded:")
        print("   • TinyStories: 28K AI-generated stories for conversation training")
        print("   • WikiText: Wikipedia articles for general knowledge")
        print()
        print("🔧 Next steps:")
        print("   1. Run your MOTHER AI as normal: python mother-20.py")
        print("   2. The AI will automatically load this new knowledge")
        print("   3. Use '/train/start' endpoint to trigger learning")
    else:
        print("⚠️ No datasets were downloaded successfully.")
        print("   MOTHER AI will still work with its existing knowledge base.")
    
    print()
    print("💡 Note: Training data is saved in 'training_data/' folder")
    print("   This folder is ignored by git (see .gitignore)")

if __name__ == "__main__":
    main()
