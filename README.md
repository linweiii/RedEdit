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
- Supports **23 photo-editing tools** (rotate, flip, resize, brightness, contrast, saturation, hue, grayscale, sepia, sharpen, blur, vignette, mosaic, crop, watermark, border, compress, format conversion, thumbnail, blend, edge extraction, AI edit, text-to-image)
- Category diversity constraint ensures exploration across different tool types

### 2. MCTS Planner

- **Selection**: UCT (Upper Confidence Bound for Trees) to balance exploration and exploitation
- **Expansion**: Progressive expansion with branching factor *k*
- **Evaluation**: VLM safety detector scores the edited image (0.0 = safe, 1.0 = unsafe)
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
│   ├── image_edit_agent.py       # 23 photo-editing tools (Qwen-Agent framework)
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

| Detector | Type | Specification String | Description |
|----------|------|---------------------|-------------|
| **Qwen3-VL** | VLM | `vlm:qwen3-vl-8b` | VLM-based safety judge (recommended) |
| **GPT-4o** | VLM | `vlm:gpt-4o` | OpenAI GPT-4o safety judge |
| **Q16** | Traditional | `Q16` | CLIP prompt-based classifier |
| **MultiHeaded** | Traditional | `MultiHeaded` | 5-head MLP safety classifier |
| **SD_Filter** | Traditional | `SD_Filter` | Stable Diffusion safety checker |
| **NSFW_Detector** | Traditional | `NSFW_Detector` | CLIP + linear head NSFW detector |
| **NudeNet** | Traditional | `NudeNet` | Keras CNN nudity detector |
| **LlavaGuard** | Traditional | `LlavaGuard` | Llava-based safety classifier |

## Supported Editing Tools (23 total)

| Category | Tools |
|----------|-------|
| **Geometry** | rotate, flip, resize, crop, thumbnail, mosaic |
| **Color** | brightness, contrast, saturation, hue, grayscale, sepia |
| **Effects** | blur, sharpen, vignette, extract_edges, blend |
| **Format/Overlay** | compress, convert_format, watermark, border |
| **AI-Powered** | image_edit_ai (VLM-guided), t2i_gen (text-to-image) |

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

RedEdit uses **SiliconFlow API** for VLM inference (Qwen3-VL). Get a free API key from [SiliconFlow Cloud](https://cloud.siliconflow.cn/account/ak):

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
    --detector "vlm:qwen3-vl-8b" \
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

# GPT-4o (requires OPENAI_API_KEY)
python run_mcts.py --image img.png --detector "vlm:gpt-4o"
```

## Key Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--image` | - | Path to a single image |
| `--dataset` | - | Path to a JSON/JSONL dataset |
| `--detector` | `vlm:qwen3-vl-8b` | Detector specification |
| `--threshold` | `0.5` | Safety score threshold |
| `--mcts-iterations` | `30` | MCTS search iterations |
| `--max-steps` | `4` | Maximum editing steps |
| `--branching` | `3` | MCTS branching factor |
| `--exploration` | `1.0` | UCT exploration constant |
| `--cpr-threshold` | `0.60` | Content preservation threshold |
| `--proposer-model` | `qwen3-vl-32b` | VLM model for proposals |
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

- [Qwen-Agent](https://github.com/QwenLM/Qwen-Agent) - Agent framework for tool execution
- [Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL) - Vision-language model for safety detection
- [SiliconFlow](https://siliconflow.cn) - API inference service
- [UnsafeBench](https://arxiv.org/abs/2406.12361) - Benchmark for image safety classification
