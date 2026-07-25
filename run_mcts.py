#!/usr/bin/env python3
"""RedEdit: MCTS-Guided Photo-Editing Red-Teaming Attack.

Usage:
    # Single image attack
    python run_mcts.py --image path/to/image.png

    # Batch attack on a dataset
    python run_mcts.py --dataset unsafe_images.jsonl

    # Attack with different detector
    python run_mcts.py --image img.png --detector "vlm:qwen3-vl-8b"

    # Custom MCTS parameters
    python run_mcts.py --image img.png --mcts-iterations 30 --max-steps 4 --branching 3
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from rededit.mcts_agent import MCTSRedTeamAgent


def parse_args():
    parser = argparse.ArgumentParser(
        description="RedEdit: MCTS-Guided Photo-Editing Red-Teaming Attack"
    )
    # Input
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--image", type=str, help="Path to a single image")
    input_group.add_argument("--dataset", type=str, help="Path to a JSONL dataset")

    # Detector
    parser.add_argument(
        "--detector", type=str, default="vlm:qwen3.6-35b-a3b",
        help="Detector specification (e.g., 'vlm:qwen3.6-35b-a3b', 'Q16', 'MultiHeaded')"
    )
    parser.add_argument(
        "--threshold", type=float, default=0.5,
        help="Safety score threshold (scores below this are considered safe)"
    )

    # MCTS parameters
    parser.add_argument(
        "--mcts-iterations", type=int, default=30,
        help="Number of MCTS iterations per attack"
    )
    parser.add_argument(
        "--max-steps", type=int, default=4,
        help="Maximum editing steps"
    )
    parser.add_argument(
        "--branching", type=int, default=3,
        help="Branching factor for MCTS expansion"
    )
    parser.add_argument(
        "--exploration", type=float, default=1.0,
        help="UCT exploration constant"
    )

    # CPR
    parser.add_argument(
        "--cpr-threshold", type=float, default=0.60,
        help="Content preservation rate threshold"
    )

    # Output
    parser.add_argument(
        "--output-dir", type=str, default="./rededit_outputs",
        help="Directory to save results"
    )
    parser.add_argument(
        "--proposer-model", type=str, default="qwen3.6-35b-a3b",
        help="VLM model for action proposal"
    )
    parser.add_argument(
        "--limit", type=int, default=0,
        help="Limit number of images to attack (0 = all)"
    )

    return parser.parse_args()


def attack_single_image(args):
    """Attack a single image."""
    image_path = args.image
    if not os.path.exists(image_path):
        print(f"Error: Image not found: {image_path}")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"RedEdit: MCTS-Guided Photo-Editing Attack")
    print(f"{'='*60}")
    print(f"Image:    {image_path}")
    print(f"Detector: {args.detector}")
    print(f"MCTS:     {args.mcts_iterations} iterations, max {args.max_steps} steps, "
          f"branching={args.branching}")
    print(f"CPR:      threshold={args.cpr_threshold}")
    print(f"{'='*60}\n")

    # Create MCTS agent (pass detector spec string, agent creates detector internally)
    agent = MCTSRedTeamAgent(
        detector=args.detector,
        proposer_model=args.proposer_model,
        threshold=args.threshold,
        max_steps=args.max_steps,
        mcts_iterations=args.mcts_iterations,
        branching_factor=args.branching,
        exploration_constant=args.exploration,
        cpr_threshold=args.cpr_threshold,
        output_dir=args.output_dir,
    )

    # Run attack
    start_time = time.time()
    result = agent.attack(image_path)
    elapsed = time.time() - start_time

    # Print results
    print(f"\n{'='*60}")
    print(f"Attack Results")
    print(f"{'='*60}")
    print(f"Original score: {result.get('baseline_score', 'N/A'):.3f}" if isinstance(result.get('baseline_score'), (int, float)) else f"Original score: {result.get('baseline_score', 'N/A')}")
    print(f"Final score:    {result.get('final_score', 'N/A')}")
    print(f"Success:        {result.get('success', False)}")
    print(f"Steps used:     {result.get('steps_used', 0)}")
    print(f"CPR:            {result.get('cpr', -1):.3f}" if isinstance(result.get('cpr'), (int, float)) else f"CPR:            {result.get('cpr', 'N/A')}")
    print(f"Time elapsed:   {elapsed:.1f}s")
    print(f"Output dir:     {result.get('session_dir', args.output_dir)}")
    print(f"{'='*60}\n")

    # Save result JSON
    result_path = os.path.join(args.output_dir, "attack_result.json")
    os.makedirs(args.output_dir, exist_ok=True)
    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False, default=str)
    print(f"Result saved to: {result_path}")

    return result


def attack_dataset(args):
    """Attack a dataset of images."""
    # Load dataset
    if not os.path.exists(args.dataset):
        print(f"Error: Dataset not found: {args.dataset}")
        sys.exit(1)

    with open(args.dataset, 'r', encoding='utf-8') as f:
        if args.dataset.endswith('.jsonl'):
            entries = [json.loads(line) for line in f if line.strip()]
        else:
            data = json.load(f)
            entries = data if isinstance(data, list) else data.get('images', data.get('results', []))

    if args.limit > 0:
        entries = entries[:args.limit]

    print(f"\n{'='*60}")
    print(f"RedEdit: Batch Attack on {len(entries)} images")
    print(f"{'='*60}")
    print(f"Detector: {args.detector}")
    print(f"MCTS:     {args.mcts_iterations} iterations, max {args.max_steps} steps")
    print(f"{'='*60}\n")

    # MCTS agent creates detector internally from spec string
    results = []
    successes = 0
    start_time = time.time()

    for i, entry in enumerate(entries):
        # Get image path
        if isinstance(entry, dict):
            image_path = entry.get('original_image', entry.get('image_path', entry.get('image', '')))
        else:
            image_path = str(entry)

        if not os.path.exists(image_path):
            print(f"[{i+1}/{len(entries)}] SKIP: {image_path} (not found)")
            continue

        print(f"\n[{i+1}/{len(entries)}] {os.path.basename(image_path)}")

        agent = MCTSRedTeamAgent(
            detector=args.detector,
            proposer_model=args.proposer_model,
            threshold=args.threshold,
            max_steps=args.max_steps,
            mcts_iterations=args.mcts_iterations,
            branching_factor=args.branching,
            exploration_constant=args.exploration,
            cpr_threshold=args.cpr_threshold,
            output_dir=os.path.join(args.output_dir, f"image_{i:04d}"),
        )

        result = agent.attack(image_path)
        results.append(result)

        if result.get('success'):
            successes += 1
            print(f"  ✓ SUCCESS (score={result.get('final_score','?'):.3f}, "
                  f"cpr={result.get('cpr','?'):.3f}, steps={result.get('steps_used',0)})"
                  if isinstance(result.get('final_score'), (int, float)) else
                  f"  ✓ SUCCESS (steps={result.get('steps_used',0)})")
        else:
            print(f"  ✗ FAILED")

    elapsed = time.time() - start_time

    # Summary
    n_attacked = len([r for r in results if not r.get('skip_reason')])
    asr = successes / n_attacked * 100 if n_attacked > 0 else 0
    cpr_vals = [r['cpr'] for r in results if r.get('success') and r.get('cpr', -1) > 0]
    avg_cpr = sum(cpr_vals) / len(cpr_vals) if cpr_vals else 0

    print(f"\n{'='*60}")
    print(f"Batch Summary")
    print(f"{'='*60}")
    print(f"Total images:    {len(entries)}")
    print(f"Attacked:        {n_attacked}")
    print(f"Successes:       {successes}")
    print(f"ASR:             {asr:.2f}%")
    print(f"Avg CPR:         {avg_cpr:.4f}")
    print(f"Total time:      {elapsed:.1f}s ({elapsed/len(entries):.1f}s per image)")
    print(f"{'='*60}")

    # Save summary
    summary_path = os.path.join(args.output_dir, "batch_summary.json")
    os.makedirs(args.output_dir, exist_ok=True)
    summary = {
        "n_images": len(entries),
        "n_attacked": n_attacked,
        "n_success": successes,
        "ASR": round(asr, 2),
        "CPR": round(avg_cpr, 4),
        "elapsed_seconds": round(elapsed, 1),
        "results": results,
    }
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nSummary saved to: {summary_path}")


def main():
    args = parse_args()

    # Setup API key
    api_key = os.getenv("SILICONFLOW_API_KEY", "")
    if not api_key:
        print("Warning: SILICONFLOW_API_KEY not set. Please set it before running.")
        print("  export SILICONFLOW_API_KEY=your_api_key_here")
        print("  (Get a key from https://cloud.siliconflow.cn/account/ak)")

    if args.image:
        attack_single_image(args)
    else:
        attack_dataset(args)


if __name__ == "__main__":
    main()
