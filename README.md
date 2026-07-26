# RedEdit: Agentic Red-Teaming of Image Safety Classifiers via MCTS-Guided Photo-Editing

[![arXiv](https://img.shields.io/badge/arXiv-2606.06140-b31b1b.svg)](https://arxiv.org/abs/2606.06140)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **RedEdit: Agentic Red-Teaming of Image Safety Classifiers via MCTS-Guided Photo-Editing**
>
> Weilin Lin, Ziqi Lin, Zhenxing Zhou, Jianze Li, Tong Zhang, Hui Xiong, Li Liu
>
> *Under Review* | [Paper](https://arxiv.org/abs/2606.06140)

## Overview

Image safety classifiers serve as a critical component of contemporary content moderation systems on the internet. However, their resilience against **user-style malicious image editing** remains underexplored. Such behaviors are highly prevalent in daily scenarios but difficult to fully reproduce.

To explore this vulnerability, we introduce **RedEdit**, a novel **black-box red-teaming agent** that formulates photo-editing evasion as a **combinatorial search problem** over edit-tool sequences. It adopts:

1. A **Vision-Language-Model (VLM)-based proposer** to generate semantically targeted candidate edits
2. A **Monte Carlo Tree Search (MCTS) planner** to prioritize promising edit paths while backtracking from ineffective ones

Together, the proposer and planner instantiate two key capabilities of human attackers: **domain knowledge** and **iterative backtracking**, respectively.

Our extensive experiments on UnsafeBench reveal profound systemic vulnerabilities: **fewer than two edits on average enable 76.2% of unsafe images to evade detectors**, while retaining **93.0% malicious semantics**, meaning that such manipulated content remains perceptually malicious to humans while easily bypassing automated moderation.

<p align="center">
  <em>We appeal to the community for more attention to this overlooked practical threat.</em>
</p>

## Method

RedEdit operates as a **black-box agent** with two core components:

### 1. VLM-Based Action Proposer

- Observes both the **original** and **current edited** images
- Generates semantically targeted edit proposals (tool + parameters)
- Supports **16 non-destructive photo-editing tools** (rotate, flip, resize, thumbnail, brightness, contrast, saturation, hue, grayscale, sepia, sharpen, vignette, compress, format conversion, watermark, border)
- AI-powered editing tools (image_edit_ai, t2i_gen) and destructive tools (mosaic, crop, blur, blend, edge extraction) are intentionally excluded
- Category diversity constraint ensures exploration across different tool types

### 2. MCTS Planner

- **Selection**: UCT (Upper Confidence Bound for Trees) to balance exploration and exploitation
- **Expansion**: Progressive expansion with branching factor *k*
- **Evaluation**: VLM safety detector scores the edited image (score ≥ 0.5 → unsafe, score < 0.5 → safe)
- **Backpropagation**: Updates node visit counts and value estimates
- **Content Preservation Rate (CPR)**: Ensures edited images retain malicious semantics

### Attack Pipeline

```
Original Image → [Baseline Score] → MCTS Search Loop:
    ├── Selection (UCT)
    ├── Expansion (VLM Proposer)
    ├── Evaluation (Safety Detector)
    └── Backpropagation
→ Best Path Extraction → CPR Check → Final Result
```

## Repository Structure

```
red_edit/
├── README.md                     # This file
├── requirements.txt              # Python dependencies
├── setup_env.sh                  # One-click environment setup
├── run_mcts.py                   # Main entry point for MCTS attack
│
├── rededit/                      # Core library
│   ├── __init__.py
│   ├── image_edit_agent.py       # 16 non-destructive photo-editing tools
│   ├── redteam_agent.py          # Red-team agent + VLM/Conventional detectors
│   ├── mcts_agent.py             # MCTS planner + VLM action proposer
│   ├── baselines.py              # Ablation baselines (Random/Single/Greedy)
│   ├── vlms.py                   # VLM model loading and inference
│   ├── utils.py                  # Utility functions
│   ├── unsafe_datasets.py        # UnsafeBench dataset loader
│   └── classifiers/              # Traditional safety classifiers
│       ├── __init__.py           # Unified classifier interface
│       ├── conventional.py       # Q16, MultiHeaded, SD_Filter, NSFW, NudeNet
│       ├── falconsai_classifier.py
│       ├── imageguard_classifier.py
│       └── llavaguard_classifier.py
│
├── scripts/
│   └── download_models.py        # Download classifier model weights
│
├── configs/                      # Configuration files (optional)
└── rededit_outputs/              # Default output directory
```

## Supported Safety Detectors

RedEdit supports both VLM-based and traditional safety detectors as described in our [paper](https://arxiv.org/abs/2606.06140). The default detector is **Qwen3.6-35B** (`vlm:qwen3.6-35b-a3b`).

### VLM-Based Detectors

| Detector | Specification String | Description |
|----------|---------------------|-------------|
| **Qwen3.6-35B** | `vlm:qwen3.6-35b-a3b` | Default VLM safety judge (recommended) |
| **Qwen3-VL-8B** | `vlm:qwen3-vl-8b` | Lightweight VLM safety judge |
| **Qwen3-VL-32B** | `vlm:qwen3-vl-32b` | Mid-size VLM safety judge |
| **Qwen3-VL-235B** | `vlm:qwen3-vl-235b` | Large-scale VLM safety judge |

### Traditional Classifiers

| Detector | Specification String | Description |
|----------|---------------------|-------------|
| **Q16** | `Q16` | CLIP prompt-based classifier |
| **MultiHeaded** | `MultiHeaded` | 5-head MLP safety classifier |
| **SD_Filter** | `SD_Filter` | Stable Diffusion safety checker |
| **NSFW_Detector** | `NSFW_Detector` | CLIP + linear head NSFW detector |
| **NudeNet** | `NudeNet` | Keras CNN nudity detector |

## Supported Editing Tools (16 total)

RedEdit uses 16 non-destructive, non-generative photo-editing tools. AI-powered editing (image_edit_ai, t2i_gen) and destructive tools (mosaic, crop, blur, blend, edge extraction) are intentionally excluded from the main tool set.

| Category | Tools |
|----------|-------|
| **Geometry** | rotate, flip, resize, thumbnail |
| **Color** | brightness, contrast, saturation, hue, grayscale, sepia |
| **Effects** | sharpen, vignette |
| **Format/Overlay** | compress, convert_format, watermark, border |

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/linweiii/RedEdit.git
cd RedEdit
```

### 2. One-Click Setup

```bash
bash setup_env.sh
```

This will:
- Check Python version (>= 3.9 required)
- Install all dependencies from `requirements.txt`
- Verify the installation

**Optionally create a virtual environment:**
```bash
bash setup_env.sh --venv
```

### 3. Set API Key

RedEdit uses **SiliconFlow API** for VLM inference (default: Qwen3.6-35B). Get a free API key from [SiliconFlow Cloud](https://cloud.siliconflow.cn/account/ak):

```bash
export SILICONFLOW_API_KEY=sk-your-api-key-here
```

> **Note**: You can also use OpenAI-compatible endpoints. Set `OPENAI_API_KEY` and `OPENAI_BASE_URL` environment variables.

### 4. (Optional) Download Classifier Models

If you want to use traditional classifiers (Q16, MultiHeaded, etc.):

```bash
python scripts/download_models.py --all
```

This downloads model weights from HuggingFace to the `checkpoints/` directory.

## Quick Start

### Attack a Single Image

```bash
python run_mcts.py --image path/to/your/image.png
```

### Attack with Custom Parameters

```bash
python run_mcts.py \
    --image path/to/image.png \
    --detector "vlm:qwen3.6-35b-a3b" \
    --mcts-iterations 30 \
    --max-steps 4 \
    --branching 3 \
    --cpr-threshold 0.60 \
    --output-dir ./my_results
```

### Batch Attack on a Dataset

```bash
# JSONL format (each line: {"original_image": "path/to/img.png", ...})
python run_mcts.py --dataset unsafe_images.jsonl --limit 100

# JSON format (list of image entries)
python run_mcts.py --dataset images.json --limit 50
```

### Use a Different Detector

```bash
# Traditional classifier
python run_mcts.py --image img.png --detector Q16

# Different VLM detector
python run_mcts.py --image img.png --detector "vlm:qwen3-vl-8b"
```

## Key Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--image` | - | Path to a single image |
| `--dataset` | - | Path to a JSON/JSONL dataset |
| `--detector` | `vlm:qwen3.6-35b-a3b` | Detector specification |
| `--threshold` | `0.5` | Unsafe score threshold (≥threshold → unsafe) |
| `--mcts-iterations` | `30` | MCTS search iterations |
| `--max-steps` | `4` | Maximum editing steps |
| `--branching` | `3` | MCTS branching factor |
| `--exploration` | `1.0` | UCT exploration constant |
| `--cpr-threshold` | `0.60` | Content preservation threshold |
| `--proposer-model` | `qwen3.6-35b-a3b` | VLM model for proposals |
| `--output-dir` | `./rededit_outputs` | Output directory |
| `--limit` | `0` | Max images to attack (0=all) |

## Output Structure

After running, the output directory contains:

```
rededit_outputs/
├── attack_result.json          # Full attack log (single image mode)
├── batch_summary.json          # Summary statistics (batch mode)
└── image_XXXX/                 # Per-image session directories
    ├── attack_log.json         # Detailed trajectory
    ├── images/                 # Intermediate edited images
    │   ├── original.png
    │   ├── edited_step1.png
    │   └── ...
    └── quality_metrics.json    # PSS/SSIM/PSNR/LPIPS metrics
```

## Citation

If you find this work useful, please cite:

```bibtex
@article{lin2026rededit,
  title={RedEdit: Agentic Red-Teaming of Image Safety Classifiers via MCTS-Guided Photo-Editing},
  author={Lin, Weilin and Lin, Ziqi and Zhou, Zhenxing and Li, Jianze and Zhang, Tong and Xiong, Hui and Liu, Li},
  journal={arXiv preprint arXiv:2606.06140},
  year={2026}
}
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Acknowledgements

- [Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL) - Vision-language model for safety detection
- [SiliconFlow](https://siliconflow.cn) - API inference service
- [UnsafeBench](https://arxiv.org/abs/2406.12361) - Benchmark for image safety classification
