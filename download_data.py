#!/usr/bin/env python3
"""
Download large data files from Zenodo.

This script downloads the embedding files and large data files that are
hosted on Zenodo due to file size limitations on GitHub.

Usage:
    python download_data.py              # Get what the experiments need, then
                                         # verify every file's checksum.
                                         # Safe to re-run: already-downloaded
                                         # files are checked, not re-fetched.
    python download_data.py --all        # Also the other embedding models
    python download_data.py --verify     # Check only, download nothing
    python download_data.py --force      # Re-download even if files exist

    --russian / --italian restrict any of the above to one corpus.
"""

import os
import sys
import json
import hashlib
import argparse
import urllib.request
from pathlib import Path


# ============================================================================
# CONFIGURATION - Update this after uploading to Zenodo
# ============================================================================

# Zenodo record URL (update after creating the deposit)
# For anonymous review, use Zenodo Sandbox: https://sandbox.zenodo.org
# For final publication, use: https://zenodo.org
ZENODO_RECORD_ID = "18260458"  # Zenodo record for stylometry-embeddings data
ZENODO_BASE_URL = "https://zenodo.org/records"  # or https://sandbox.zenodo.org/records

# File manifest with checksums (MD5)
# Format: local_path -> {zenodo_filename, size_mb, md5, required, description}
# Zenodo files use prefixed names (russian_, italian_) to avoid conflicts
FILES = {
    # Russian corpus - embeddings
    "data/russian/embeddings_gemini.npy": {
        "zenodo_name": "russian_embeddings_gemini.npy",
        "size_mb": 68,
        "md5": "401c5c4fd1faa6a02b591cdd7a4b2534",
        "required": True,
        "description": "Gemini embeddings (5800 x 3072)"
    },
    "data/russian/embeddings_openai.npy": {
        "zenodo_name": "russian_embeddings_openai.npy",
        "size_mb": 136,
        "md5": "834f687ea0510959653b7c23d785b4e6",
        "required": False,
        "description": "OpenAI embeddings (5800 x 3072)"
    },
    "data/russian/embeddings_voyage.npy": {
        "zenodo_name": "russian_embeddings_voyage.npy",
        "size_mb": 23,
        "md5": "b73106221f695f6b5ab6330eaf6f15cd",
        "required": False,
        "description": "Voyage embeddings (5800 x 1024)"
    },
    "data/russian/embeddings_qwen8b.npy": {
        "zenodo_name": "russian_embeddings_qwen8b.npy",
        "size_mb": 91,
        "md5": "4834e06cbf7a174ea6730cd1be6165fa",
        "required": False,
        "description": "Qwen3-8B embeddings (5800 x 4096)"
    },
    "data/russian/embeddings_e5-large.npy": {
        "zenodo_name": "russian_embeddings_e5-large.npy",
        "size_mb": 23,
        "md5": "28cf11920721963d7b17a6a244c78afa",
        "required": False,
        "description": "E5-large embeddings (5800 x 1024)"
    },
    "data/russian/embeddings_bge-m3.npy": {
        "zenodo_name": "russian_embeddings_bge-m3.npy",
        "size_mb": 23,
        "md5": "aacba13ff5feb8a6650b5351096cade2",
        "required": False,
        "description": "BGE-M3 embeddings (5800 x 1024)"
    },
    # Russian corpus - poems
    "data/russian/poems.json": {
        "zenodo_name": "russian_poems.json",
        "size_mb": 10,
        "md5": "a14c133fe0652c419f872042c88a11f1",
        "required": True,
        "description": "Russian poems with text and metadata"
    },
    "data/russian/linguistic_features.json": {
        "zenodo_name": "russian_linguistic_features.json",
        "size_mb": 14,
        "md5": "a2ce664fe956608754ea133778fb72ea",
        "required": True,
        "description": "Linguistic features (POS, lemmas, morphology) - Tier 3"
    },

    # Italian corpus - embeddings
    "data/italian/embeddings_gemini.npy": {
        "zenodo_name": "italian_embeddings_gemini.npy",
        "size_mb": 122,
        "md5": "335536e797205e90f3fde292d60a897e",
        "required": True,
        "description": "Gemini embeddings (10400 x 3072)"
    },
    "data/italian/embeddings_openai.npy": {
        "zenodo_name": "italian_embeddings_openai.npy",
        "size_mb": 122,
        "md5": "c034b2d9e6395683ea42597ef773af54",
        "required": False,
        "description": "OpenAI embeddings (10400 x 3072)"
    },
    "data/italian/embeddings_voyage.npy": {
        "zenodo_name": "italian_embeddings_voyage.npy",
        "size_mb": 41,
        "md5": "915dac0a86e6930bdf129e7030d0d4c5",
        "required": False,
        "description": "Voyage embeddings (10400 x 1024)"
    },
    "data/italian/embeddings_qwen8b.npy": {
        "zenodo_name": "italian_embeddings_qwen8b.npy",
        "size_mb": 163,
        "md5": "d52424d8d790ae57567c8fee8a66efd2",
        "required": False,
        "description": "Qwen3-8B embeddings (10400 x 4096)"
    },
    "data/italian/embeddings_e5-large.npy": {
        "zenodo_name": "italian_embeddings_e5-large.npy",
        "size_mb": 41,
        "md5": "2b27a24c9ed8abe26002d017ea23a3d0",
        "required": False,
        "description": "E5-large embeddings (10400 x 1024)"
    },
    # Italian corpus - poems and features
    "data/italian/poems.json": {
        "zenodo_name": "italian_poems.json",
        "size_mb": 20,
        "md5": "b32f21b5a88edbcfbe9cb279cdd411f7",
        "required": True,
        "description": "Italian poems with text"
    },
    "data/italian/linguistic_features.json": {
        "zenodo_name": "italian_linguistic_features.json",
        "size_mb": 17,
        "md5": "dc1f110b3b39dbf06f028b19b157695a",
        "required": True,
        "description": "Linguistic features (POS, morphology) - Tier 3"
    },
}

# ============================================================================


def get_zenodo_download_url(record_id: str, filename: str) -> str:
    """Construct Zenodo download URL for a file."""
    # Zenodo direct download URL format
    return f"{ZENODO_BASE_URL}/{record_id}/files/{filename}?download=1"


def compute_md5(filepath: Path) -> str:
    """Compute MD5 hash of a file."""
    hash_md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def download_file(url: str, dest: Path, expected_size_mb: int = None) -> bool:
    """Download a file with progress indicator."""
    dest.parent.mkdir(parents=True, exist_ok=True)

    try:
        print(f"  Downloading: {dest.name}", end="", flush=True)
        if expected_size_mb:
            print(f" ({expected_size_mb} MB)", end="", flush=True)
        print(" ... ", end="", flush=True)

        urllib.request.urlretrieve(url, dest)
        print("done")
        return True

    except Exception as e:
        print(f"FAILED: {e}")
        return False


def verify_file(filepath: Path, expected_md5: str = None) -> bool:
    """Verify file exists and optionally check MD5."""
    if not filepath.exists():
        return False

    if expected_md5:
        actual_md5 = compute_md5(filepath)
        if actual_md5 != expected_md5:
            print(f"  WARNING: MD5 mismatch for {filepath.name}")
            print(f"    Expected: {expected_md5}")
            print(f"    Got: {actual_md5}")
            return False

    return True


def verify_all(files, base_dir):
    """Check every file's checksum. Returns 0 if all good, 1 otherwise."""
    print("\n[Verifying files...]")
    all_ok = True
    for filepath, info in files.items():
        full_path = base_dir / filepath
        if verify_file(full_path, info.get("md5")):
            print(f"  OK: {filepath}")
        else:
            print(f"  MISSING/INVALID: {filepath}")
            all_ok = False
    if all_ok:
        print(f"\nAll {len(files)} files present and verified.")
    else:
        print("\nSome files are missing or corrupt. Re-run with --force to "
              "re-download them.")
    return 0 if all_ok else 1


def main():
    parser = argparse.ArgumentParser(
        description="Download large data files from Zenodo"
    )
    parser.add_argument(
        "--russian", action="store_true",
        help="Download Russian corpus only"
    )
    parser.add_argument(
        "--italian", action="store_true",
        help="Download Italian corpus only"
    )
    parser.add_argument(
        "--minimal", action="store_true",
        help="Download only required files (Gemini embeddings + poems)"
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Download all files including optional embeddings"
    )
    parser.add_argument(
        "--verify", action="store_true",
        help="Verify existing files without downloading"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-download even if files exist"
    )

    args = parser.parse_args()

    # Determine base directory
    base_dir = Path(__file__).parent

    print("=" * 60)
    print("Downloading data files from Zenodo")
    print("=" * 60)

    if ZENODO_RECORD_ID == "XXXXXXX":
        print("\nERROR: Zenodo record ID not configured!")
        print("Please update ZENODO_RECORD_ID in this script after uploading to Zenodo.")
        print("\nFor reviewers: Contact the authors for the Zenodo access link.")
        return 1

    # Filter files based on arguments
    files_to_download = {}

    for filepath, info in FILES.items():
        # Filter by corpus
        if args.russian and not filepath.startswith("data/russian"):
            continue
        if args.italian and not filepath.startswith("data/italian"):
            continue

        # Filter by required/optional
        if args.minimal and not info["required"]:
            continue

        # If neither --all nor --minimal, download required files
        if not args.all and not args.minimal and not info["required"]:
            continue

        files_to_download[filepath] = info

    # Calculate total size
    total_size = sum(f["size_mb"] for f in files_to_download.values())
    print(f"\nFiles to download: {len(files_to_download)}")
    print(f"Total size: ~{total_size} MB")

    # Check existing files
    existing = []
    missing = []
    for filepath in files_to_download:
        full_path = base_dir / filepath
        if full_path.exists():
            existing.append(filepath)
        else:
            missing.append(filepath)

    # --verify must run before the "already downloaded" short-circuit below,
    # otherwise a complete-but-corrupt download reports success without any
    # checksum being computed.
    if args.verify:
        return verify_all(files_to_download, base_dir)

    if existing and not args.force:
        print(f"\nAlready downloaded: {len(existing)} files")
        if not missing:
            # Everything is on disk. Confirm it is intact rather than trusting
            # that the files exist, then stop -- there is nothing to fetch.
            print("Nothing to download; checking the files that are here.")
            return verify_all(files_to_download, base_dir)

    # Download missing files
    print(f"\n[Downloading {len(missing)} files...]")

    failed = []
    for filepath in missing:
        info = files_to_download[filepath]
        full_path = base_dir / filepath

        # Construct download URL
        # Note: Zenodo files use prefixed names (russian_, italian_)
        zenodo_filename = info["zenodo_name"]
        url = get_zenodo_download_url(ZENODO_RECORD_ID, zenodo_filename)

        if not download_file(url, full_path, info["size_mb"]):
            failed.append(filepath)
            continue

        # Verify if MD5 is available
        if info.get("md5"):
            if not verify_file(full_path, info["md5"]):
                failed.append(filepath)

    # Summary
    print("\n" + "=" * 60)
    if failed:
        print(f"COMPLETED WITH ERRORS: {len(failed)} files failed")
        for f in failed:
            print(f"  - {f}")
        return 1
    print("SUCCESS: All files downloaded")
    rc = verify_all(files_to_download, base_dir)
    if rc == 0:
        print("\nYou can now run the experiments:")
        print("  python experiments/exp7_waterfall/run_extended.py")
    return rc


if __name__ == "__main__":
    sys.exit(main())
