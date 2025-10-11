## Improve Item Representation

Here are some ways to make your system work better:

---

### 1. Add More Track Information

Give the model more details about each song:

Option A: Add more text fields
- Right now, the system only uses: track name, artist name, album name
- You can add: genre tags, mood labels, release year, popularity scores
- Edit the `corpus_types` in your config file to include `tag_list`

Option B: Use audio features
- Instead of just text, use the actual sound of the music
- Try CLAP (a model that understands both text and audio)
- This helps find songs that sound similar, not just have similar descriptions

How to do it:
- Change the `_stringify_metadata()` function to include more fields
- Add code to extract audio features from tracks
- Combine text and audio information together

---

### 2. Use a Better Retrieval Model

Replace the basic BM25/BERT models with newer, more powerful ones:

Better text models:
- Qwen2.5-Embedding - Works well with multiple languages
- Contriever - Good at finding relevant items without training
- E5 or BGE - Currently the best text embedding models
- ColBERT - Matches words more precisely


--

## Resource

- https://huggingface.co/datasets/talkpl-ai/TalkPlayData-2-Track-Metadata
- https://huggingface.co/datasets/talkpl-ai/TalkPlayData-2-Track-Embeddings
