import requests

import matplotlib.pyplot as plt
import seaborn as sns

import umap, numpy as np, pandas as pd

from nltk import FreqDist

import pandas as pd

from sklearn.decomposition import TruncatedSVD
from scipy import sparse
import numpy as np

from collections import Counter
from scipy.sparse import csr_matrix

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

# ------------------------------------------------------------------
# common objects
# ------------------------------------------------------------------

lr = LogisticRegression(
    max_iter=4000,
    solver="lbfgs",
    n_jobs=-1)

rf = RandomForestClassifier(
    n_estimators=200,  # more trees for stability
    max_depth=None,  # let it grow fully
    n_jobs=-1,  # parallelizex
    random_state=42)

hgbc = HistGradientBoostingClassifier(
    random_state=42)

from sklearn.neural_network import MLPClassifier

mlp = MLPClassifier(hidden_layer_sizes=(512, 128),
                    activation='relu',
                    solver='adam',
                    alpha=1e-4,
                    max_iter=500,
                    random_state=42)

clfs = {"lr" : lr, "rf" : rf, "hgbc" : hgbc, "mlp" : mlp}

def load_embedding(name, base_dir="../data/labyrinth_embeddings"):
    return np.load(f"{base_dir}/{name}.npy", mmap_mode="r")

def merge_embeddings(embs, method="mean", weights=None) -> np.ndarray:
    """Merge a list of embedding matrices [n, d] via concat / mean / weighted."""
    mats = [np.asarray(load_embedding(e)) for e in embs]
    # sanity checks
    n_rows = [m.shape[0] for m in mats]
    if len(set(n_rows)) != 1:
        raise ValueError(f"Inconsistent n_samples across embeddings: {n_rows}")
    n, d_list = n_rows[0], [m.shape[1] for m in mats]

    if method == "concat":
        return np.concatenate(mats, axis=1)  # [n, sum(d)]
    elif method == "mean":
        M = np.stack(mats, axis=0)           # [k, n, d]
        return M.mean(axis=0)                # [n, d]
    elif method == "weighted":
        if weights is None:
            raise ValueError("weights must be provided for method='weighted'")
        w = np.asarray(weights, dtype=float)
        if w.shape[0] != len(mats):
            raise ValueError(f"weights length {w.shape[0]} != #embs {len(mats)}")
        w = w / w.sum()
        M = np.stack(mats, axis=0)           # [k, n, d]
        return (w[:, None, None] * M).sum(axis=0)  # [n, d]
    else:
        raise ValueError(f"Unknown method: {method}")

def run_classifier(emb, classifier, y, cv, return_model=False):
    if isinstance(emb, str):
        X = load_embedding(emb)
    else:
        X = merge_embeddings(emb, method="concat")

    # Cross-validated predictions (evaluation)
    y_pred = cross_val_predict(classifier, X, y, cv=cv)
    acc = accuracy_score(y, y_pred)
    f1 = f1_score(y, y_pred, average="macro")

    model = None
    if return_model:
        # train on the full data to get a deployable model
        model = clone(classifier).fit(X, y)

    return y_pred, acc, f1, model

unique_lbls = ["mythological",
               "technical_literal",
               "confusion_metaphorics",
               "scientific_complexity_metaphorics",
               "medical_anatomical",
               "ambiguous_indeterminate"]


palette = {
    "mythological": "#1E40AF", # "#7D3FB2",  # Regal violet — imaginative, mythic depth
    "technical_literal": "#7D3FB2", # "#C68E17",  # Golden ochre — tangible, crafted, architectural
    "confusion_metaphorics":  "#228B68",  # Emerald teal — organic, physiological clarity
    "scientific_complexity_metaphorics": "#C68E17", #"#1E40AF",  # Deep royal blue — intellectual, analytical
    "medical_anatomical": "#C24130",  # Burnt orange-red — emotional, human, entangled
    "ambiguous_indeterminate": "#6B7280",  # Neutral slate gray — uncertain, indeterminate
}