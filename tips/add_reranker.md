## Add Reranker Module

Refine initial retrieval results with a second-stage ranker:

Option A: Embedding-based reranking
- Use user embeddings for personalization
  - Compute user profile from listening history
  - Score candidates by user-item similarity

- Cross-modal reranking: Combine multiple signals
  - Text relevance + audio similarity + user preference

Option B: LLM-based reranking
- Use LLM to judge relevance of top-k candidates
- Prompt: "Rank these tracks by relevance to: {user_query}"
- Models: Llama-3-8B, Qwen-7B, or specialized rankers

Implementation approach:
```python
# Add to CRS pipeline after retrieval
retrieval_items = self.retrieval.text_to_item_retrieval(query, topk=100)

# Rerank top candidates
if self.reranker:
    retrieval_items = self.reranker.rerank(
        query=query,
        candidates=retrieval_items[:50],
        user_profile=user_profile,
        topk=20
    )
```

---

## Resource

- https://huggingface.co/datasets/talkpl-ai/TalkPlayData-2-User-Embeddings
- https://huggingface.co/datasets/talkpl-ai/TalkPlayData-2-Track-Metadata
- https://huggingface.co/datasets/talkpl-ai/TalkPlayData-2-Track-Embeddings
