from pydantic import BaseModel, FilePath, DirectoryPath, Field
from typing import Union
from pathlib import Path


class SnapshotConfig(BaseModel):
    snapshot_version: int = Field(...)
    snapshot_directory: Union[Path, DirectoryPath] = Field(...)
    tables_to_use: list[str] = Field(...)
    gold_labled_trial_file: FilePath = Field(...)
    pseudo_labled_trial_file: FilePath = Field(...)
    save_model_binaries_to: Union[Path, FilePath] = Field(...)
