"""
Demo script for running a single conversational music recommendation query.
This script initializes a Music CRS baseline system and processes a single user query,
returning a recommended track and a natural language response explaining the recommendation.
"""

import os
import json
import torch
import argparse
from mcrs import load_crs_baseline

def main(args):
    """
    Run a single CRS query and display results.

    Args:
        args: Namespace object containing:
            - user_query (str): Natural language music preference query
            - save_path (str): Output directory for results

    Returns:
        None. Results are printed to console and saved to disk.
    """
    music_crs = load_crs_baseline(
        lm_type="meta-llama/Llama-3.2-1B-Instruct",
        retrieval_type="bm25",
        item_db_name="talkpl-ai/TalkPlayData-2-Track-Metadata",
        user_db_name="talkpl-ai/TalkPlayData-2-User-Metadata",
        split_types=["test_warm", "test_cold"], # for test
        corpus_types=["track_name", "artist_name", "album_name"],
        cache_dir="./cache",
        device="cuda",
        dtype=torch.bfloat16
    )
    results = music_crs.chat(user_query=args.user_query)
    os.makedirs(args.save_path, exist_ok=True)
    with open(f"{args.save_path}/results.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    track_id = results['retrieval_items'][0]
    track_url = f"https://open.spotify.com/track/{track_id}"
    print("-"*100)
    print(f"🎵 Music: {track_url}")
    print("🤖 Assistant Response:")
    print(results["response"])
    print("-"*100)
    print(f"More detail results are saved in {args.save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run a single music recommendation query with conversational response generation."
    )
    parser.add_argument(
        "--user_query",
        type=str,
        default="I'm looking for hiphop music.",
        help="Natural language query expressing music preferences (e.g., 'I want upbeat pop songs')"
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="./demo/static",
        help="Directory path to save detailed results JSON file"
    )
    args = parser.parse_args()
    main(args)
