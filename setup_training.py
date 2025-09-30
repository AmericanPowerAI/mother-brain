#!/usr/bin/env python3
"""
setup_training.py - Download WEB-SCALE training data for MOTHER AI
This gives your AI true internet knowledge, not just Wikipedia!
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
    print("🚀 MOTHER AI WEB-SCALE Training Data Setup")
    print("=" * 60)
    print("This will download REAL INTERNET DATA to train MOTHER AI")
    print("Your AI will learn from actual web pages, news, code, and conversations!")
    print()
    print("📁 Datasets will be saved to: training_data/")
    print("⏱️  This may take 5-10 minutes depending on your internet...")
    print()
    
    # Create dataset manager
    dm = DatasetManager()
    
    print("🌐 AVAILABLE WEB DATASETS:")
    print("-" * 40)
    
    categories = dm.get_available_datasets()
    for category, datasets in categories.items():
        print(f"\n{category}:")
        for dataset in datasets:
            print(f"   📦 {dataset}")
    print()
    
    # Download ALL web-scale datasets
    all_datasets = []
    for category_datasets in categories.values():
        all_datasets.extend(category_datasets)
    
    print("📥 DOWNLOADING WEB-SCALE KNOWLEDGE...")
    print("-" * 40)
    
    success_count = 0
    for dataset in all_datasets:
        try:
            print(f"\n🌐 Downloading {dataset}...")
            result = dm.download_dataset(dataset)
            print(f"✅ Success: {dataset}")
            success_count += 1
        except Exception as e:
            print(f"⚠️  Skipped {dataset}: {e}")
            print("   MOTHER AI will still work with other datasets.")
    
    print()
    print("=" * 60)
    
    if success_count > 0:
        print(f"🎉 WEB-SCALE TRAINING COMPLETE!")
        print(f"   Successfully downloaded {success_count}/{len(all_datasets)} datasets!")
        print()
        print("📊 KNOWLEDGE NOW INCLUDES:")
        print("   🌐 Common Crawl - Real web pages from across the internet")
        print("   📰 Real News - Current events and journalism")  
        print("   💬 Reddit - Social conversations and trends")
        print("   💻 GitHub - Programming code and technical knowledge")
        print("   📚 Wikipedia - General encyclopedia knowledge")
        print("   🗣️ Conversation Data - AI dialogue training")
        print()
        print("🔧 NEXT STEPS:")
        print("   1. Run MOTHER AI: python mother-20.py")
        print("   2. AI will automatically load this WEB knowledge")
        print("   3. Use '/train/start' endpoint to begin learning")
        print("   4. Your AI now has TRUE internet-scale knowledge!")
    else:
        print("⚠️ No datasets were downloaded successfully.")
        print("   MOTHER AI will use its existing knowledge base.")
    
    print()
    print("💡 TIP: Run this script again anytime to update knowledge")
    print("📁 Data location: 'training_data/' (ignored by git)")

if __name__ == "__main__":
    main()
