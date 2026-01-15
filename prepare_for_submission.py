#!/usr/bin/env python3
"""
Prepare repository for submission by separating GitHub and Zenodo files.

This script:
1. Creates a 'zenodo_upload/' folder with all large files
2. Removes large files from the main repo (keeps small metadata files)
3. Creates a summary of what goes where

Usage:
    python prepare_for_submission.py           # Dry run (shows what will happen)
    python prepare_for_submission.py --execute # Actually move files

After running with --execute:
1. Upload contents of 'zenodo_upload/' to Zenodo
2. Get the Zenodo record ID
3. Update ZENODO_RECORD_ID in download_data.py
4. Push the repo to anonymous GitHub
"""

import os
import sys
import shutil
import argparse
from pathlib import Path

# Files to move to Zenodo (>8 MB)
# Format: (source_path, zenodo_filename)
ZENODO_FILES = [
    # Russian corpus
    ("data/russian/embeddings_gemini.npy", "russian_embeddings_gemini.npy"),
    ("data/russian/embeddings_openai.npy", "russian_embeddings_openai.npy"),
    ("data/russian/embeddings_voyage.npy", "russian_embeddings_voyage.npy"),
    ("data/russian/embeddings_qwen8b.npy", "russian_embeddings_qwen8b.npy"),
    ("data/russian/embeddings_e5-large.npy", "russian_embeddings_e5-large.npy"),
    ("data/russian/embeddings_bge-m3.npy", "russian_embeddings_bge-m3.npy"),
    ("data/russian/poems.json", "russian_poems.json"),
    ("data/russian/linguistic_features.json", "russian_linguistic_features.json"),

    # Italian corpus
    ("data/italian/embeddings_gemini.npy", "italian_embeddings_gemini.npy"),
    ("data/italian/embeddings_openai.npy", "italian_embeddings_openai.npy"),
    ("data/italian/embeddings_voyage.npy", "italian_embeddings_voyage.npy"),
    ("data/italian/embeddings_qwen8b.npy", "italian_embeddings_qwen8b.npy"),
    ("data/italian/embeddings_e5-large.npy", "italian_embeddings_e5-large.npy"),
    ("data/italian/poems.json", "italian_poems.json"),
    ("data/italian/linguistic_features.json", "italian_linguistic_features.json"),
]

# Files to keep in GitHub (<8 MB)
GITHUB_KEEP = [
    # Russian metadata
    "data/russian/dataset_metadata.json",
    "data/russian/prosody.json",
    "data/russian/labels.npy",

    # Italian metadata
    "data/italian/dataset_metadata.json",
    "data/italian/labels.npy",
]


def get_file_size_mb(filepath):
    """Get file size in MB."""
    if filepath.exists():
        return filepath.stat().st_size / (1024 * 1024)
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Prepare repository for GitHub + Zenodo submission"
    )
    parser.add_argument(
        "--execute", action="store_true",
        help="Actually move files (default is dry run)"
    )
    parser.add_argument(
        "--zenodo-dir", default="zenodo_upload",
        help="Directory for Zenodo files (default: zenodo_upload)"
    )

    args = parser.parse_args()

    base_dir = Path(__file__).parent
    zenodo_dir = base_dir / args.zenodo_dir

    print("=" * 70)
    print("PREPARE REPOSITORY FOR SUBMISSION")
    print("=" * 70)

    if not args.execute:
        print("\n*** DRY RUN - No files will be moved ***")
        print("*** Run with --execute to actually move files ***\n")

    # Calculate sizes
    zenodo_size = 0
    github_size = 0

    print("\n[Files to move to Zenodo]")
    print("-" * 70)

    files_to_move = []
    for src_path, zenodo_name in ZENODO_FILES:
        full_path = base_dir / src_path
        if full_path.exists():
            size = get_file_size_mb(full_path)
            zenodo_size += size
            files_to_move.append((src_path, zenodo_name, size))
            print(f"  {src_path:<50} {size:>6.1f} MB")
        else:
            print(f"  {src_path:<50} [NOT FOUND]")

    print(f"\n  {'TOTAL':<50} {zenodo_size:>6.1f} MB")

    print("\n[Files to keep in GitHub]")
    print("-" * 70)

    for filepath in GITHUB_KEEP:
        full_path = base_dir / filepath
        if full_path.exists():
            size = get_file_size_mb(full_path)
            github_size += size
            print(f"  {filepath:<50} {size:>6.1f} MB")
        else:
            print(f"  {filepath:<50} [NOT FOUND]")

    # Add code files to GitHub size estimate
    zenodo_src_paths = [src for src, _ in ZENODO_FILES]
    code_size = 0
    for pattern in ["**/*.py", "**/*.json", "**/*.md", "**/*.txt", "**/*.png"]:
        for f in base_dir.glob(pattern):
            if "zenodo" not in str(f) and "__pycache__" not in str(f):
                rel_path = str(f.relative_to(base_dir))
                if rel_path not in zenodo_src_paths:
                    code_size += get_file_size_mb(f)

    github_size += code_size
    print(f"  {'+ Code, results, figures':<50} {code_size:>6.1f} MB")
    print(f"\n  {'TOTAL (GitHub)':<50} {github_size:>6.1f} MB")

    if args.execute:
        print("\n[Executing...]")
        print("-" * 70)

        # Create Zenodo directory
        zenodo_dir.mkdir(exist_ok=True)
        print(f"  Created: {zenodo_dir}/")

        # Move files
        moved_count = 0
        for src_path, zenodo_name, size in files_to_move:
            src = base_dir / src_path
            # Use prefixed names for Zenodo to avoid conflicts
            dst = zenodo_dir / zenodo_name

            if src.exists():
                shutil.move(str(src), str(dst))
                print(f"  Moved: {src_path} -> {args.zenodo_dir}/{zenodo_name}")
                moved_count += 1

        print(f"\n  Moved {moved_count} files to {args.zenodo_dir}/")

        # Create a README in Zenodo folder
        zenodo_readme = zenodo_dir / "README.txt"
        with open(zenodo_readme, 'w') as f:
            f.write("Zenodo Data Files for: Stylometry or Embeddings?\n")
            f.write("=" * 50 + "\n\n")
            f.write("Upload all .npy and .json files to Zenodo.\n\n")
            f.write("Files:\n")
            for src_path, zenodo_name, size in files_to_move:
                f.write(f"  - {zenodo_name} ({size:.1f} MB)\n")
            f.write(f"\nTotal: {zenodo_size:.1f} MB\n")
        print(f"  Created: {zenodo_dir}/README.txt")

        print("\n" + "=" * 70)
        print("DONE!")
        print("=" * 70)
        print(f"""
Next steps:

1. UPLOAD TO ZENODO:
   - Go to https://sandbox.zenodo.org (for review) or https://zenodo.org (final)
   - Create a new upload
   - Upload all files from: {zenodo_dir}/
   - Note the record ID (e.g., 1234567)

2. UPDATE download_data.py:
   - Open download_data.py
   - Change: ZENODO_RECORD_ID = "XXXXXXX"
   - To:     ZENODO_RECORD_ID = "<your-record-id>"

3. PUSH TO GITHUB:
   - The repo is now ready for anonymous submission
   - Total size: ~{github_size:.1f} MB (under 8 MB limit)

4. TEST:
   - Clone fresh copy
   - Run: python download_data.py --minimal
   - Run: python experiments/exp7_waterfall/run_extended.py
""")

    else:
        print("\n" + "=" * 70)
        print("DRY RUN COMPLETE")
        print("=" * 70)
        print(f"""
Summary:
  - Zenodo: {zenodo_size:.1f} MB ({len(files_to_move)} files)
  - GitHub: {github_size:.1f} MB

Run with --execute to actually move files:
  python prepare_for_submission.py --execute
""")

    return 0


if __name__ == "__main__":
    sys.exit(main())
