import torch
from .crs_baseline import CRS_BASELINE

def load_crs_baseline(
    lm_type="meta-llama/Llama-3.2-1B-Instruct",
    retrieval_type="bm25",
    item_db_name: str = "talkpl-ai/TalkPlayData-2-Track-Metadata",
    user_db_name: str = "talkpl-ai/TalkPlayData-2-User-Metadata",
    split_types: list[str] = ["test_warm", "test_cold"], # for test
    corpus_types: list[str] = ["track_name", "artist_name", "album_name"],
    cache_dir="./cache",
    device="cuda",
    attn_implementation="eager",
    dtype=torch.bfloat16
):
    return CRS_BASELINE(lm_type, retrieval_type, item_db_name, user_db_name, split_types, corpus_types, cache_dir, device, attn_implementation, dtype)
