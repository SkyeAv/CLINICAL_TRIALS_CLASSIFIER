from src.ct_classifier.core.utils import load_yaml, load_model, load_labels, xz_backup
from src.ct_classifier.models.snaphot import SnapshotConfig
from src.ct_classifier.core.processing import features
from src.ct_classifier.core.training import new_model
from pathlib import Path
from typing import Any
import numpy as np
import typer

app = typer.Typer()


@app.command()
def train_model(
    snapshot_config: Path = typer.Option(
        ...,
        "-c",
        "--snapshot-config",
        help="path to clinical trials snapshot configuration",
    ),
    seed: int = typer.Option(87, "-s", "--seed", help="seed for reproducability"),
    backup: bool = typer.Option(
        False,
        "-b",
        "--backup-old-db",
        help="creates a timestamped xz compressed backup of the previous chroma_db build",
    ),
) -> None:
    cfg: dict[str, Any] = load_model(load_yaml(snapshot_config), SnapshotConfig)
    model_p: Path = cfg["save_model_binaries_to"]
    if backup and model_p.is_file():
        xz_backup(model_p)
    labels, encoded_features = features(cfg, 87, "training")
    gold_lables: np.ndarray = np.array(load_labels(cfg["gold_labled_trial_file"]))
    pseudo_lables: np.ndarray = np.array(load_labels(cfg["pseudo_labled_trial_file"]))
    new_model(
        model_p,
        seed,
        labels,
        encoded_features,
        gold_lables,
        pseudo_lables,
    )
    return None


# wrapper for entrypoints
def main() -> None:
    app()
    return None
