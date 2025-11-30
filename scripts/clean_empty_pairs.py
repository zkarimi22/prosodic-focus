#!/usr/bin/env python3
"""
Clean up empty pair directories that don't contain any .wav files.
Deletes directories in data/minimal_pairs/pair_XXXX that have no audio files.
"""

import os
import shutil
from pathlib import Path

# Get the minimal_pairs directory
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
MINIMAL_PAIRS_DIR = PROJECT_ROOT / "data" / "minimal_pairs"

def has_wav_files(directory):
    """Check if a directory contains any .wav files."""
    return any(directory.glob("*.wav"))

def clean_empty_pairs():
    """Remove pair directories that don't contain any .wav files."""
    if not MINIMAL_PAIRS_DIR.exists():
        print(f"Error: {MINIMAL_PAIRS_DIR} does not exist!")
        return
    
    deleted_count = 0
    kept_count = 0
    
    # Find all pair_XXXX directories
    pair_dirs = sorted([d for d in MINIMAL_PAIRS_DIR.iterdir() 
                       if d.is_dir() and d.name.startswith("pair_")])
    
    print(f"Found {len(pair_dirs)} pair directories to check...\n")
    
    for pair_dir in pair_dirs:
        if has_wav_files(pair_dir):
            kept_count += 1
            print(f"✓ Keeping {pair_dir.name} (has .wav files)")
        else:
            deleted_count += 1
            print(f"✗ Deleting {pair_dir.name} (no .wav files)")
            shutil.rmtree(pair_dir)
    
    print(f"\nSummary:")
    print(f"  Kept: {kept_count} directories")
    print(f"  Deleted: {deleted_count} directories")

if __name__ == "__main__":
    clean_empty_pairs()

