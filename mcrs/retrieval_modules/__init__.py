from .bm25 import BM25_MODEL
from .bert import BERT_MODEL

def load_retrieval_module(
        retrieval_type: str,
        dataset_name: str = "talkpl-ai/TalkPlayData-2-Track-Metadata",
        split_types: list[str] = ["test_warm", "test_cold"], # for test
        corpus_types: list[str] = ["track_name", "artist_name", "album_name"],
        cache_dir: str = "./cache"
    ):
    if retrieval_type == "bm25":
        return BM25_MODEL(dataset_name, split_types, corpus_types, cache_dir)
    elif retrieval_type == "bert":
        return BERT_MODEL(dataset_name, split_types, corpus_types, cache_dir)
    else:
        raise ValueError(f"Unsupported retrieval type: {retrieval_type}")
