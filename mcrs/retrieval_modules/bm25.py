"""BM25-based retrieval utilities for music track metadata.

This module builds and queries a BM25 index over track metadata fields
loaded from a Hugging Face dataset. The index is cached to disk for
subsequent reuse.
"""
import os
import json
import bm25s
from datasets import load_dataset, concatenate_datasets


class BM25_MODEL:
    """BM25 retriever over track metadata.
    Builds an index from specified corpus fields (e.g., `track_name`, `artist_name`,`album_name`) and provides text-to-item retrieval.
    """
    def __init__(self,
        dataset_name: str = "talkpl-ai/TalkPlayData-2-Track-Metadata",
        split_types: list[str] = ["test_warm", "test_cold"],
        corpus_types: list[str] = ["track_name", "artist_name", "album_name"],
        cache_dir: str = "./cache"
    ) -> None:
        """Initialize the BM25 retriever.
        Args:
            dataset_name: Hugging Face dataset name containing track metadata.
            split_types: Dataset splits to load and concatenate.
            corpus_types: Metadata fields to include in the text corpus.
            cache_dir: Directory to cache the BM25 index and artifacts.
        """
        self.dataset_name = dataset_name
        self.split_types = split_types
        self.corpus_types = corpus_types
        self.corpus_name = "_".join(corpus_types)
        self.cache_dir = cache_dir
        self.metadata_dict = self._load_corpus()
        if os.path.exists(f"{self.cache_dir}/bm25/{self.corpus_name}"):
            self.bm25_model, self.track_ids = self._load_bm25(self.corpus_name)
        else:
            self.build_index()
            self.bm25_model, self.track_ids = self._load_bm25(self.corpus_name)

    def _load_bm25(self, corpus_name: str) -> tuple[bm25s.BM25, list[str]]:
        """Load a cached BM25 index and track id list.
        Args:
            corpus_name: Name of the corpus subdirectory under the cache.
        Returns:
            A tuple of (bm25_model, track_ids).
        """
        bm25 = bm25s.BM25.load(f"{self.cache_dir}/bm25/{corpus_name}", load_corpus=True)
        track_ids = json.load(open(f"{self.cache_dir}/bm25/{corpus_name}/track_ids.json", "r"))
        return bm25, track_ids

    def _load_corpus(self) -> dict[str, dict]:
        """Load and combine metadata splits from the configured dataset.
        Returns:
            A mapping from `track_id` to its metadata dictionary.
        """
        metadata_dataset = load_dataset(self.dataset_name)
        metadata_concat_dataset = concatenate_datasets([metadata_dataset[split_type] for split_type in self.split_types])
        metadata_dict = {item["track_id"]: item for item in metadata_concat_dataset}
        return metadata_dict

    def _stringify_metadata(self, metadata: dict[str, object]) -> str:
        """Convert a metadata dict into a multi-line string for indexing.
        Args:
            metadata: Track metadata with fields listed in `self.corpus_types`.
        Returns:
            A newline-separated string with `field: value` per selected field.
        """
        metadata_str = ""
        for corpus_type in self.corpus_types:
            entity = metadata[corpus_type]
            if isinstance(entity, list):
                entity = ", ".join(entity)
            metadata_str += f"{corpus_type}: {entity}\n"
        return metadata_str

    def build_index(self) -> None:
        """Build and persist a BM25 index over the loaded corpus.
        """
        track_ids = list(self.metadata_dict.keys())
        corpus = []
        for track_id in track_ids:
            metadata = self.metadata_dict[track_id]
            metadata_str = self._stringify_metadata(metadata)
            corpus.append(metadata_str)
        corpus_tokens = bm25s.tokenize(corpus)
        retriever = bm25s.BM25()
        retriever.index(corpus_tokens)
        os.makedirs(os.path.join(self.cache_dir, "bm25", self.corpus_name), exist_ok=True)
        retriever.save(f"{self.cache_dir}/bm25/{self.corpus_name}", corpus=corpus)
        with open(os.path.join(self.cache_dir, "bm25", self.corpus_name, "track_ids.json"), "w") as f:
            json.dump(track_ids, f, indent=2)

    def text_to_item_retrieval(self, query: str, topk: int) -> list[str]:
        """Retrieve top-k track IDs for a natural language query.
        Args:
            query: The user text query to match against the metadata corpus.
            k: Number of items to retrieve.
        Returns:
            A list of track IDs ordered by decreasing BM25 score.
        """
        query_tokens = bm25s.tokenize([query.lower()])
        doc_scores = self.bm25_model.retrieve(query_tokens, k=topk, return_as="tuple")
        bm25_results = doc_scores.documents[0]
        return [self.track_ids[item['id']] for item in bm25_results]

    def batch_text_to_item_retrieval(self, queries: list[str], topk: int) -> list[list[str]]:
        """Retrieve top-k track IDs for multiple queries in batch.
        Args:
            queries: List of user text queries to match against the metadata corpus.
            topk: Number of items to retrieve per query.
        Returns:
            A list of lists, where each inner list contains track IDs ordered by decreasing BM25 score.
        """
        query_tokens = bm25s.tokenize([q.lower() for q in queries])
        doc_scores = self.bm25_model.retrieve(query_tokens, k=topk, return_as="tuple")
        results = []
        for i in range(len(queries)):
            bm25_results = doc_scores.documents[i]
            results.append([self.track_ids[item['id']] for item in bm25_results])
        return results
