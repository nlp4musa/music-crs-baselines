import os
import torch
import json
from datasets import load_dataset, concatenate_datasets

class MusicCatalogDB:
    def __init__(self,
            dataset_name: str = "talkpl-ai/TalkPlayData-2-Track-Metadata",
            split_types: list[str] = ["test_warm", "test_cold"],
            corpus_types: list[str] = ["track_name", "artist_name", "album_name", "release_date"],
        ):
        metadata_dataset = load_dataset(dataset_name)
        metadata_concat_dataset = concatenate_datasets([metadata_dataset[split_type] for split_type in split_types])
        self.corpus_types = corpus_types
        self.metadata_dict = {item["track_id"]: item for item in metadata_concat_dataset}

    def id_to_metadata(self, track_id: str, use_semantic_id: bool = False):
        metadata = self.metadata_dict[track_id]
        track_id = metadata['track_id']
        entity_str = f"track_id: {track_id}"
        for corpus_type in self.corpus_types:
            corpus_type_value = ", ".join(metadata[corpus_type]).lower()
            entity_str += f", {corpus_type}: {corpus_type_value}"
        return entity_str
