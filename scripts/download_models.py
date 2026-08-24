#!/usr/bin/env python3
"""download_models.py — Pre-download safety-classifier weights.

RedEdit downloads conventional-classifier weights automatically on first use
(see rededit/classifiers/conventional.py). This script lets you pull them all
upfront so the first run is fully offline.

Usage::
    # Download everything
    python scripts/download_models.py --all

    # Download a single classifier
    python scripts/download_models.py Q16

Models are stored under ``checkpoints/``:
  - Q16        → yiting/Q16                (prompts.p)
  - MultiHeaded→ yiting/MultiHeaded        (5 × .pt heads)
  - SD_Filter  → CompVis/stable-diffusion-safety-checker
  - NSFW_Detector → yiting/NSFWDetector    (clip_autokeras_binary_nsfw.pth)
  - NudeNet    → yiting/NudeNet            (classifier_model.h5)
  - FalconsaiNSFW → Falconsai/nsfw_image_detection (via transformers)
"""

import argparse
import os
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_CKPT = _PROJECT_ROOT / "checkpoints"


def _hf_snapshot(repo_id: str, target: str) -> None:
    from huggingface_hub import snapshot_download
    dest = _CKPT / target
    dest.mkdir(parents=True, exist_ok=True)
    print(f"[download] {repo_id} → {dest}")
    snapshot_download(repo_id=repo_id, repo_type="model", local_dir=str(dest))


def download(name: str) -> None:
    name = name.lower().replace("-", "_")
    if name in ("q16", "all"):
        _hf_snapshot("yiting/Q16", "Q16")
    if name in ("multiheaded", "all"):
        _hf_snapshot("yiting/MultiHeaded", "MultiHeaded")
    if name in ("sd_filter", "all"):
        _hf_snapshot("CompVis/stable-diffusion-safety-checker",
                     "CompVis/stable-diffusion-safety-checker")
    if name in ("nsfw_detector", "all"):
        _hf_snapshot("yiting/NSFWDetector", "NSFWDetector")
    if name in ("nudenet", "all"):
        _hf_snapshot("yiting/NudeNet", "NudeNet")
    if name == "all":
        # Falconsai is fetched by transformers directly; touch it here too.
        print("[download] FalconsaiNSFW is auto-downloaded by transformers on first use.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name", nargs="?", default="all",
                        help="Classifier name (Q16, MultiHeaded, SD_Filter, "
                             "NSFW_Detector, NudeNet) or 'all'.")
    args = parser.parse_args()
    download(args.name)
    print("[download] Done. Classifier weights are under checkpoints/.")


if __name__ == "__main__":
    main()
