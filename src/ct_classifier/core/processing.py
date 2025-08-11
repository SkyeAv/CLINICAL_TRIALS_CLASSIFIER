from torchdr.affinity import NormalizedGaussianAffinity
from transformers import AutoTokenizer, AutoModel
from sklearn.preprocessing import StandardScaler
from functools import lru_cache, partial
from contextlib import contextmanager
from typing import ContextManager
from torchdr import KernelPCA
from pathlib import Path
from typing import Any
from tqdm import tqdm
import polars as pl
import numpy as np
import joblib
import fsspec
import random
import torch


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    return None


def from_zipfiles(zip_path: Path, tablename: str) -> pl.DataFrame:
    with fsspec.open(f"zip://{tablename}.txt::{zip_path.as_posix()}", mode="rt") as f:
        return pl.read_csv(f, has_header=True, separator="|")


def drop_identifiers(df: pl.DataFrame) -> pl.DataFrame:
    avoid: list[str] = ["id", "nct_id"]
    return df.with_columns(df.select([col for col in df.columns if col not in avoid]))


def scale(df: pl.DataFrame, colname: str, pkl_path: Path, mode: str) -> pl.DataFrame:
    if mode == "production" and pkl_path.exists():
        scaler = joblib.load(pkl_path)
        scaled_values = (
            scaler.transform(
                df.select(pl.col(colname).fill_null(pl.col(colname).mean())).to_numpy()
            )
            .reshape(-1, 1)
            .ravel()
        )
    elif mode == "production":
        msg: str = f"CODE:8 | {pkl_path.as_posix()} does not exist"
        raise RuntimeError(msg)
    elif mode == "training":
        scaler = StandardScaler()
        scaled_values = (
            scaler.fit_transform(
                df.select(pl.col(colname).fill_null(pl.col(colname).mean())).to_numpy()
            )
            .reshape(-1, 1)
            .ravel()
        )
        joblib.dump(scaler, pkl_path)
    else:
        msg = f"CODE:9 | {mode} is not a valid mode"
        raise RuntimeError(msg)
    return df.drop(colname).with_columns(pl.Series(colname, scaled_values))


CACHE_PATH: Path = Path("CACHE/").resolve()
CACHE_PATH.mkdir(parents=True, exist_ok=True)


def numeric(
    df: pl.DataFrame, version: str, tablename: str, mode: str
) -> tuple[pl.DataFrame, list[str]]:
    scaler_cache: Path = CACHE_PATH / "SCALER" / version / tablename
    scaler_cache.mkdir(parents=True, exist_ok=True)
    numerics: list[str] = [
        colname for colname, dtype in df.schema.items() if dtype.is_numeric()
    ]
    for colname in numerics:
        pkl_path: Path = scaler_cache / f"{colname}.pkl"
        df = scale(df, colname, pkl_path, mode)
    return (df, numerics)


def is_boolean(s: pl.Series) -> bool:
    if s.null_count() == len(s):
        return False
    mask = s.is_in(["t", "f"])
    return mask.any() and mask.sum() == s.len() - s.null_count()


def boolean(
    df: pl.DataFrame, version: str, tablename: str
) -> tuple[pl.DataFrame, list[str]]:
    booleans: list[str] = []
    for colname, series in df.to_dict(as_series=True).items():
        if is_boolean(series):
            df = df.with_columns(
                pl.col(colname).map_elements(
                    lambda x: 1 if str(x) == "t" else 0, return_dtype=pl.UInt8
                )
            )
    return (df, booleans)


@contextmanager  # type: ignore
def suppress_tqdm() -> ContextManager:  # type: ignore
    original_init = tqdm.__init__

    def patched_init(self, *args, **kwargs) -> None:  # type: ignore
        kwargs["disable"] = True
        original_init(self, *args, **kwargs)
        return None

    tqdm.__init__ = patched_init
    try:
        yield
    finally:
        tqdm.__init__ = original_init


def from_transformers(model_name: str) -> tuple[AutoTokenizer, AutoModel]:
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)  # type: ignore
    with suppress_tqdm():
        model = AutoModel.from_pretrained(
            model_name,
            device_map={"": "cpu"},
            torch_dtype="float32",
            low_cpu_mem_usage=False,
        ).eval()
    return (tokenizer, model)


@lru_cache(maxsize=64)
def biobert_embedding(
    tokenizer: AutoTokenizer, model: AutoModel, x: str
) -> torch.Tensor:
    inputs = tokenizer(x, return_tensors="pt", truncation=True, max_length=512)  # type: ignore
    inputs = {k: v.to("cpu") for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)  # type: ignore
        pooler_output = getattr(outputs, "pooler_output", None)
        if pooler_output is None or pooler_output.numel() == 0:
            raise RuntimeError(f"CODE:5 | No pooler_output from BioBert for input: {x}")

        return (  # type: ignore
            pooler_output.detach().cpu().float().squeeze()
        )  # shape: (768,)


def text_fields(
    df: pl.DataFrame,
    version: str,
    tablename: str,
    tokenizer: AutoTokenizer,
    model: AutoModel,
    seed: int,
    mode: str,
    training_encoded_cols: list[str] = [],
    min_cutoff: int = 1,
    biobert_embedding_cutoff: int = 15,
) -> tuple[pl.DataFrame, list[str], torch.Tensor, list[str]]:
    pca_cache: Path = CACHE_PATH / "PCA" / version / tablename
    pca_cache.mkdir(parents=True, exist_ok=True)
    texts: list[str] = [
        colname
        for colname, dtype in df.schema.items()
        if dtype == pl.String
        and (colname in training_encoded_cols if mode == "production" else True)
    ]
    embeddeds: list[str] = []
    embeddings: list[torch.Tensor] = []
    embedding_fn = partial(biobert_embedding, tokenizer, model)
    for colname, series in df.to_dict(as_series=True).items():
        if colname in texts:
            unique_values: int = series.n_unique()
            if biobert_embedding_cutoff >= unique_values > min_cutoff:
                dummies = df[colname].to_dummies()
                df = df.drop(colname).hstack(dummies)
            elif unique_values > biobert_embedding_cutoff:
                pkl_path: Path = pca_cache / f"{colname}.pkl"
                single_column_embedding: list[torch.Tensor] = [
                    embedding_fn(x) for x in series
                ]
                with torch.no_grad():
                    D: torch.Tensor = torch.cdist(
                        single_column_embedding,
                        single_column_embedding
                    )
                    sigma: torch.Tensor = torch.median(D[D > 0])
                aff = NormalizedGaussianAffinity(
                    sigma=sigma,
                    zero_diag=False,
                    backend="torch",
                    device=DEVICE,
                    _pre_processed=True,
                )
                kpca = KernelPCA(
                    affinity=aff,
                    n_components=64,
                    random_state=seed,
                    backend="torch",
                    device="cpu",
                )
                dr_embedding = kpca.fit_transform(single_column_embedding)
                joblib.dump(kpca, pkl_path)
                embeddings.append(
                    torch.stack(dr_embedding)
                )
                texts.remove(colname)
                embeddeds.append(colname)
                df = df.drop(colname)
            else:
                df = df.drop(colname)
    return (df, texts, torch.stack(embeddings, dim=0), embeddeds)


def features(cfg: dict[str, Any], seed: int, mode: str) -> tuple[list[str], np.ndarray]:
    set_seed(seed)
    version: str = str(cfg["snapshot_version"])
    zip_path: Path = cfg["snapshot_directory"] / f"{version}.zip"
    tokenizer, model = from_transformers("dmis-lab/biobert-base-cased-v1.1")
    all_embeddings: list[torch.Tensor] = []
    encoded_cols: list[str] = []
    col_cache: Path = CACHE_PATH / "COLUMNS" / version / "columns.pkl"
    col_cache.parent.mkdir(parents=True, exist_ok=True)
    if mode == "production" and col_cache.exists():
        training_encoded_cols: list[str] = joblib.load(col_cache)
    elif mode == "production":
        msg: str = f"CODE:6 | {col_cache.as_posix()} does not exist"
        raise RuntimeError(msg)
    elif mode == "training":
        training_encoded_cols = []
    else:
        msg = f"CODE:7 | {mode} is an invalid mode"
        raise RuntimeError(msg)
    for idx, tablename in enumerate(cfg["tables_to_use"]):
        df: pl.DataFrame = from_zipfiles(zip_path, tablename)
        if idx == 0:
            trial_identifiers: list[str] = df["nct_id"].to_list()
        df, encoded_numerics = numeric(df, version, tablename, mode)
        df, encoded_booleans = boolean(df, version, tablename)
        df, encoded_texts, freetext_embeddings, embeddeds = text_fields(
            df, version, tablename, tokenizer, model, seed, mode, training_encoded_cols
        )
        encoded_cols_in_df: list[str] = (
            encoded_numerics + encoded_booleans + encoded_texts
        )
        if mode == "training":
            df = df.with_columns(df.select(encoded_cols_in_df))
        if mode == "production":
            encoded_cols_to_keep: list[str] = [
                col for col in encoded_cols_in_df if col in training_encoded_cols
            ]
            df = df.with_columns(df.select(encoded_cols_to_keep))
        encoded_cols.extend(encoded_cols_in_df + embeddeds)
        table_embeddings: torch.Tensor = torch.cat(
            [
                torch.from_numpy(df.to_numpy()).float(),
                freetext_embeddings.transpose(0, 1).flatten(start_dim=1),
            ],
            dim=1,
        )
        all_embeddings.append(table_embeddings)
    feature_tensor: torch.Tensor = torch.cat(all_embeddings, dim=0)
    if mode == "training":
        joblib.dump(encoded_cols, col_cache)
    return trial_identifiers, feature_tensor.numpy()
