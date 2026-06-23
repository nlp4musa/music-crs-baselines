# Music Conversational Recommendation Challenge — Baselines

Official evaluation framework for the **The RecSys Challenge 2026 Conversational Music Recommendation System Challenge**. Music-CRS focuses on the evolving landscape of music discovery, where static recommendation lists are being replaced by dynamic, conversational interactions. As users increasingly interact with AI through natural language, there is a critical need for systems that can seamlessly integrate Natural Language Understanding (NLU) with high-precision Recommender Systems (RecSys). This challenge aims to push the boundaries of how AI understands nuanced user preferences, explores musical tastes through dialogue, and provides contextually relevant track recommendations.

This repository provides standardized tools to evaluate music recommendation systems on the **TalkPlay Data Challenge** datasets. Participants must follow the strict inference JSON format specified below to ensure their submissions can be properly evaluated.

- **ACM RecSys Website**: [https://www.recsyschallenge.com/](https://www.recsyschallenge.com/)
- **Challenge Website**: [https://nlp4musa.github.io/music-crs-challenge/](https://nlp4musa.github.io/music-crs-challenge/)
- **Challenge datasets**: [talkpl-ai/talkplay-data-challenge](https://huggingface.co/collections/talkpl-ai/talkplay-data-challenge)

## Timeline

| Date | Milestone |
|------|-----------|
| 31 March 2026 | Website online |
| 10 April 2026 | Start RecSys Challenge — Release dataset (Train, Development, Blind A) |
| 15 April 2026 | Submission System Open — Leaderboard live (with Blind A dataset) |
| 15 June 2026 | Blind Dataset B released, Activate submission system for Blind B dataset |
| 30 June 2026 | End RecSys Challenge |
| 6 July 2026 | Final Leaderboard & Winners — EasyChair open for submissions |
| 9 July 2026 | Upload code of the final predictions |
| 20 July 2026 | Paper Submission Due |
| 3 August 2026 | Paper Acceptance Notifications |
| 10 August 2026 | Camera-Ready Papers |
| September 2026 | RecSys Challenge Workshop at ACM RecSys 2026 |

---

## Baseline System

The system operates on a **two-stage pipeline**:
1. **RecSys** — Retrieve candidate tracks matching user preferences
2. **LLM** — Generate a natural language response explaining the recommendations

### Core Components

| Component | Description | Module |
|---|---|---|
| LLM | Generates natural language responses (Llama-3.2-1B-Instruct) | `mcrs/lm_modules/` |
| RecSys | Retrieves relevant tracks via BM25 (sparse) or BERT (dense) | `mcrs/retrieval_modules/` |
| User DB | Stores user profiles (user_id, age, gender, country) | `mcrs/db_user/user_profile.py` |
| Item DB | Contains track metadata (name, artist, album, tags, release date) | `mcrs/db_item/music_catalog.py` |

---

## Challenge Resources

- **Dataset collection**: [TalkPlayData-Challenge](https://huggingface.co/collections/talkpl-ai/talkplay-data-challenge)
- **Conversation Dataset**: [TalkPlayData-Challenge-Dataset](https://huggingface.co/datasets/talkpl-ai/TalkPlayData-Challenge-Dataset)
- **Track Metadata**: [TalkPlayData-Challenge-Track-Metadata](https://huggingface.co/datasets/talkpl-ai/TalkPlayData-Challenge-Track-Metadata)
- **User Profiles**: [TalkPlayData-Challenge-User-Metadata](https://huggingface.co/datasets/talkpl-ai/TalkPlayData-Challenge-User-Metadata)
- **Blind A Dataset**: [TalkPlayData-Challenge-Blind-A](https://huggingface.co/datasets/talkpl-ai/TalkPlayData-Challenge-Blind-A)
- **Blind B Dataset**: [TalkPlayData-Challenge-Blind-B](https://huggingface.co/datasets/talkpl-ai/TalkPlayData-Challenge-Blind-B)

---

## Quick Start

### Installation

```bash
uv venv .venv --python=3.10
source .venv/bin/activate
uv pip install -e .
uv pip install flash-attn --no-build-isolation # for fast llm inference
```

### Run Inference on the Development Set

**⚠️ Note: During inference, the recommender system must always retrieve candidates from the entire track catalog. Do not filter, subset, or restrict tracks using `track_split_types` or any other mechanism!**

For BM25/BERT baselines, your config must include:

```yaml
track_split_types:
  - "all_tracks"
```

If you do not use `all_tracks`, your evaluation may be considered invalid.

- Always use `all_tracks` for every experiment and submission.
- Do **not** preprocess, filter, or use only a subset of tracks during inference.


```bash
# BM25 baseline
python run_inference_devset.py --tid llama1b_bm25_devset --batch_size 16

# BERT baseline
python run_inference_devset.py --tid llama1b_bert_devset --batch_size 16
```

Results are saved to `exp/inference/{tid}.json`.

### Run Inference on Blind Sets (for submission)

```bash
# BM25 baseline
python run_inference_blindset.py --tid llama1b_bm25_blindset_B --batch_size 16

# BERT baseline
python run_inference_blindset.py --tid llama1b_bert_blindset_B --batch_size 16
```

---

## Custom Configuration

Create a config file in `config/`:

```yaml
# config/my_model.yaml
lm_type: "Qwen/Qwen3-4B" # change llama to qwen3
retrieval_type: "bm25"
test_dataset_name: "talkpl-ai/TalkPlayData-Challenge-Dataset"
item_db_name: "talkpl-ai/TalkPlayData-Challenge-Track-Metadata"
user_db_name: "talkpl-ai/TalkPlayData-Challenge-User-Metadata"
track_split_types:
  - "all_tracks"
user_split_types:
  - "all_users"
corpus_types:
  - "track_name"
  - "artist_name"
  - "album_name"
  - "release_date"
cache_dir: "./cache"
device: "cuda"
attn_implementation: "flash_attention_2"
```

Then run with your config:

```bash
python run_inference_devset.py --tid my_model
```

---

## Evaluation

For evaluation, please refer to: https://github.com/nlp4musa/music-crs-evaluator

---

## Tips & Extensions

See `./tips/` for advanced techniques. Some directions to explore:

- **Improve Item Representation** — Add audio features or use stronger embedding models
- **Add a Reranker Module** — Implement two-stage ranking with LLM or embedding-based rerankers
- **Generative Retrieval** — Use semantic IDs for end-to-end track generation

---

Good luck with the challenge!
