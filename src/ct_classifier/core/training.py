from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, roc_auc_score
from typing import Any, Self
from loguru import logger
from pathlib import Path
import xgboost as xgb
import numpy as np
import sys


def aggegate_inputs(
    seed: int,
    labels: list[str],
    features: np.ndarray,
    gold_labels: np.ndarray,
    pseudo_labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    labels_np: np.ndarray = np.array(labels)

    in_gold: np.ndarray = np.isin(labels_np, gold_labels)
    in_pseudo: np.ndarray = np.isin(labels_np, pseudo_labels)

    pos_mask: np.ndarray = in_gold | in_pseudo
    neg_mask: np.ndarray = ~pos_mask

    orig_weights: np.ndarray = np.where(in_gold, 1.0, np.where(in_pseudo, 0.2, 0.0))

    pos_indices = np.where(pos_mask)[0]
    X_pos = features[pos_indices]
    y_pos = np.ones(pos_indices.shape[0], dtype=int)
    w_pos = orig_weights[pos_indices]

    number_of_random_trials: int = len(y_pos)

    neg_indices_all = np.where(neg_mask)[0]
    neg_indices = np.random.choice(
        neg_indices_all, size=number_of_random_trials, replace=False
    )
    X_neg = features[neg_indices]
    y_neg = np.zeros(number_of_random_trials, dtype=int)

    unique_weights, counts = np.unique(w_pos, return_counts=True)
    proportions = counts / counts.sum()
    w_neg = np.random.choice(
        unique_weights, size=number_of_random_trials, p=proportions
    )

    X = np.vstack([X_pos, X_neg])
    y = np.concatenate([y_pos, y_neg])
    w = np.concatenate([w_pos, w_neg])

    return (X, y, w)


def dmatrixes(
    seed: int, X: np.ndarray, y: np.ndarray, w: np.ndarray
) -> tuple[xgb.DMatrix, xgb.DMatrix, np.ndarray]:
    X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
        X, y, w, test_size=0.2, random_state=seed, stratify=y
    )
    dtrain = xgb.DMatrix(X_train, label=y_train, weight=w_train)
    dtest = xgb.DMatrix(X_test, label=y_test, weight=w_test)
    return (dtrain, dtest, y_test)


logger.remove()
logger.add(
    sys.stderr,
    level="INFO",
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
)
logger.add(
    "training.log",
    level="INFO",
    rotation="10 MB",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
)


class LoguruCallback(xgb.callback.TrainingCallback):
    def after_iteration(
        self: Self, model: xgb.Booster, epoch: int, evals_log: Any
    ) -> bool:
        for dset, metrics in evals_log.items():
            for mname, history in metrics.items():
                logger.info(f"[round {epoch+1:03d}] {dset}-{mname}={history[-1]:.6f}")
        return False  # continue training


def new_model(
    model_name: Path,
    seed: int,
    labels: list[str],
    features: np.ndarray,
    gold_labels: np.ndarray,
    pseudo_labels: np.ndarray,
) -> None:
    X, y, w = aggegate_inputs(seed, labels, features, gold_labels, pseudo_labels)
    dtrain, dtest, y_test = dmatrixes(seed, X, y, w)
    params: dict[str, Any] = {
        "objective": "binary:logistic",
        "eval_metric": ["auc", "logloss"],
        "learning_rate": 0.05,
        "max_depth": 4,
        "min_child_weight": 10,
        "gamma": 1.0,
        "seed": seed,
        "subsample": 0.8,
        "colsample_bytree": 0.7,
        "reg_alpha": 0.5,
        "reg_lambda": 1.0,
        "tree_method": "hist",
    }
    bst = xgb.train(
        params,
        dtrain,
        num_boost_round=10000,
        evals=[(dtrain, "train"), (dtest, "test")],
        callbacks=[LoguruCallback()],
        early_stopping_rounds=50,
        verbose_eval=False,  # suppresses built-in prints
    )
    y_prob = bst.predict(dtest)
    test_auc: float = roc_auc_score(y_test, y_prob)
    test_logloss: float = log_loss(y_test, y_prob)
    logger.info(f"Final Test AUC: {test_auc:.6f}")
    logger.info(f"Final Test LogLoss: {test_logloss:.6f}")
    bst.save_model(model_name.as_posix())
    return None
