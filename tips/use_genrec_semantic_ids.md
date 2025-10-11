## Generative Retrieval (Advanced)

Replace embedding similarity with end-to-end generation:

Concept: Instead of retrieve-then-generate, directly generate track identifiers

Semantic IDs approach:
- Assign hierarchical semantic IDs to tracks (e.g., `jazz/smooth/piano/0042`)
- Train LLM to generate relevant track IDs given user query
- Single model replaces both retrieval and generation stages

Benefits:
- Unified architecture
- Can model complex user intent
- Leverages LLM reasoning capabilities

Implementation steps:
1. Create semantic ID system for tracks
2. Fine-tune LLM to generate track IDs
3. Optional: Use collaborative filtering for ID assignments


## Resource

- https://huggingface.co/datasets/talkpl-ai/TalkPlayData-2-Track-Metadata
- https://huggingface.co/datasets/talkpl-ai/TalkPlayData-2-Track-Embeddings
