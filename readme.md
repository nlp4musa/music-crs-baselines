# Music Conversational Recommendation Challenge Baselines

Welcome to the **Music CRS Challenge**! This repository provides baseline systems for building conversational music recommendation systems using the [TalkPlayData-2](https://huggingface.co/datasets/talkpl-ai/TalkPlayData-2) dataset and [TalkPlayData-Blind](https://huggingface.co/collections/talkpl-ai/talkplay-data-challenge).

## 📰 Timeline

- **Task description release**: October 15, 2025
- **Development dataset release**: October 15, 2025
- **Baseline system release**: October 15, 2025
- **Submission site**: December 11, 2025
- **Blind evaluation dataset release**: December 11, 2025
- **Final submission deadline**: December 30, 2025
- **Results notification**: January 30, 2026


## 📋 Challenge Overview

Build a conversational AI that can:
- Understand user music preferences through natural dialogue
- Recommend relevant tracks from a music catalog
- Generate engaging, personalized responses about music
- Benchmark track: Evaluate on public test sets with known ground truth
- Blind evaluation track: Test on hidden evaluation data for final ranking

## Baseline System

![System Architecture](https://i.imgur.com/vSlXQ7v.png)

The system operates on a **two-stage pipeline**:
1. **Recsys**: Find candidate tracks matching user preferences
2. **LLM**: Create natural language responses explaining recommendations

### Core Components

1. **🤖 LLM (Language Model)**
   - Generates natural language responses
   - Model: Llama-3.2-1B-Instruct
   - Module: `mcrs/lm_modules`

2. **🎯 RecSys (Recommendation System)**
   - Retrieves relevant tracks from catalog
   - Methods: BM25 (sparse) or BERT (dense)
   - Module: `mcrs/retrieval_modules/`

3. **👤 User DB (User Database)**
   - Stores user profiles (user_id, age, gender, country)
   - Module: `mcrs/db_user/user_profile.py`

4. **🎵 Item DB (Music Catalog Database)**
   - Contains track metadata (track_id track name, artist, album, tags, release date)
   - Module: `mcrs/db_item/music_catalog.py`

---

## 📚 Challenge Resources

- **Conversation Dataset**: [TalkPlayData-2](https://huggingface.co/datasets/talkpl-ai/TalkPlayData-2)
- **Track Metadata**: [Track Metadata](https://huggingface.co/datasets/talkpl-ai/TalkPlayData-2-Track-Metadata)
- **Pre-extracted Track Embeddings**: [Track Embeddings](https://huggingface.co/datasets/talkpl-ai/TalkPlayData-2-Track-Embeddings)
- **User Profiles**: [User Metadata](https://huggingface.co/datasets/talkpl-ai/TalkPlayData-2-User-Metadata)
- **Pre-extracted User Embeddings**: [User Embeddings](https://huggingface.co/datasets/talkpl-ai/TalkPlayData-2-User-Embeddings)
- **Blind Dataset**: [TalkPlayData-Blind](https://huggingface.co/datasets/talkpl-ai/TalkPlayData-Blind)

## 🚀 Quick Start

### Installation

```bash
uv venv .venv --python=3.10
source .venv/bin/activate
uv pip install -e .
```


### Run Full Inference

Process the entire test dataset with batch inference:

```bash
# BM25 baseline
python run_inference_test.py --tid llama1b_bm25_testset --batch_size 16
# BERT baseline
python run_inference_test.py --tid llama1b_bert_testset --batch_size 16
```

Results will be saved to `exp/inference/{tid}.json`.

Also you need to submit blind estimation dataset

```bash
python run_inference_blind.py --tid llama1b_bert_blindset --batch_size 16
python run_inference_blind.py --tid llama1b_bm25_blindset --batch_size 16
```

---

### Run a Demo Query
Try the baseline system with a simple query:

```bash
python run_crs.py --user_query "I'm looking for jazz music."
```

**Example Output:**
```
----------------------------------------------------------------------------------------------------
🎵 Music: https://open.spotify.com/track/3auejP8jQXX4soeSvMCtqL
🤖 Assistant Response:
I'm glad you liked the recommended track \"Isabella\" by Gregg Karukas!

Isabella is a smooth jazz track that exudes a soothing and intimate atmosphere.
The song features a gentle piano melody, accompanied by a subtle saxophone solo,
creating a warm and relaxing ambiance. The tempo is moderate, with a steady beat
that encourages you to sway to the rhythm....[omitted]
```


## 🛠️ Custom Configuration

Create your own config file in `config/`:

```yaml
# config/my_model.yaml
lm_type: "meta-llama/Llama-3.2-1B-Instruct"
retrieval_type: "qwen_embedding"  # your custom retriever
item_db_name: "talkpl-ai/TalkPlayData-2-Track-Metadata"
user_db_name: "talkpl-ai/TalkPlayData-2-User-Metadata"
split_types:
  - "test_warm"
  - "test_cold"
corpus_types:
  - "track_name"
  - "artist_name"
  - "album_name"
  - "tag_list"
cache_dir: "./cache"
device: "cuda"
attn_implementation: "flash_attention_2"
```

Then run with your config:
```bash
python run_inference.py --tid my_model
```

## 📊 Evaluation

For evaluation, please refer to:
https://github.com/nlp4musa/music-crs-evaluator

## 🎯 Challenge Tips

1. Start simple: Run baseline, understand the pipeline
2. Iterate quickly: Test changes on a subset before full evaluation
3. Use caching: Precompute embeddings to speed up experiments
4. Monitor metrics: Track both recommendation accuracy and response quality

See `./tips/` for advanced techniques and future directions:
- **Improve Item Representation**: Add audio features, use better embedding models
- **Add Reranker Module**: Implement two-stage ranking with LLM or embedding-based rerankers
- **Generative Retrieval**: Use semantic IDs for end-to-end track generation

---

## 🤝 Contributing

Feel free to:
- Implement new retrieval/reranking modules
- Add evaluation metrics
- Improve prompt engineering
- Share your best-performing configurations

Good luck with the challenge! 🎵
