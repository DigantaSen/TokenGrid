"""
Learned token-cost predictor — how an *honest* agent forms its bid.

TokenGrid's auction ranks bids by reputation * urgency / predicted_tokens and
its reputation ledger scores agents on min(pred, actual) / max(pred, actual).
Both only work if an honest agent can actually forecast its own cost. The
hackathon version faked that with a fixed formula (prompt chars / 4 + half
the output cap); this replaces it with a model trained on real traces:

    features  MiniLM embedding of the task text + length/number counts + benchmark
    target    log(total tokens)
    learners  "ridge"       — linear, with built-in CV over the regularization strength
              "lgbm"        — gradient-boosted trees (captures non-linear effects)
              "lgbm_median" — the same trees, trained on absolute log-error (targets
                              the median, which is what the reputation ledger scores)
    others    per-benchmark average, k-nearest-neighbours on embeddings, and
              short/medium/long cost buckets (crossfit_group_mean / _knn / _buckets)
    point     exp(log prediction) * Duan's smearing factor = an estimate of the
              *mean* cost (exp of a log-scale prediction alone underestimates it)
    quantile  exp(log prediction + q-th quantile of the model's log-errors) = a
              cost the task stays under with probability ~q (a "safe" bid).
              The errors come from cross-validation *inside* the training data,
              so they are honest out-of-sample errors, not optimistic fit errors.

Predictions used in experiments are cross-fitted: each task's forecast comes
from a model that never saw that task.
"""
import re
import warnings

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import config
from .tasks import DATASETS

_NUM_RE = re.compile(r"\d+(?:\.\d+)?")
LEARNERS = ("ridge", "lgbm", "lgbm_median")
# LightGBM complains when fitted and queried with plain arrays; the column order never changes here.
warnings.filterwarnings("ignore", message="X does not have valid feature names")


def handcrafted_features(texts: list[str], datasets: list[str]) -> np.ndarray:
    return np.array([
        [np.log1p(len(t)), np.log1p(len(t.split())), len(_NUM_RE.findall(t)), t.count("\n") + 1,
         *[float(ds == d) for d in DATASETS]]
        for t, ds in zip(texts, datasets)
    ], dtype=np.float64)


def embed_texts(texts: list[str]) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(config.EMBEDDING_MODEL_NAME)
    return np.asarray(model.encode(texts, normalize_embeddings=True, batch_size=64,
                                   show_progress_bar=len(texts) > 300), dtype=np.float32)


def _make_model(learner: str):
    if learner in ("lgbm", "lgbm_median"):
        from lightgbm import LGBMRegressor
        # "lgbm_median" minimizes absolute error in log space, i.e. targets the
        # median log-cost — the quantity the ledger's min/max ratio score rewards.
        objective = "l1" if learner == "lgbm_median" else "regression"
        return LGBMRegressor(objective=objective, n_estimators=300, learning_rate=0.03, num_leaves=15,
                             min_child_samples=20, subsample=0.8, subsample_freq=1, colsample_bytree=0.5,
                             random_state=0, verbose=-1)
    return make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 4, 13)))


class TokenPredictor:
    def __init__(self, learner: str = "ridge"):
        if learner not in LEARNERS:
            raise ValueError(f"unknown learner {learner}")
        self.learner = learner
        self.model = _make_model(learner)
        self.smear = 1.0
        self.log_errors = np.zeros(1)

    def fit(self, X: np.ndarray, tokens: np.ndarray, quantiles: bool = False) -> "TokenPredictor":
        y = np.log(np.maximum(tokens, 1))
        self.model.fit(X, y)
        # Mean-targeting learners get Duan's smearing; the median learner must
        # not, or it would be pushed back toward the mean it deliberately avoids.
        if self.learner != "lgbm_median":
            self.smear = float(np.mean(np.exp(y - self.model.predict(X))))
        if quantiles:
            # Out-of-sample errors on the training data itself, via an inner CV.
            inner = cross_val_predict(_make_model(self.learner), X, y, cv=KFold(5, shuffle=True, random_state=0))
            self.log_errors = y - inner
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.maximum(1.0, np.exp(self.model.predict(X)) * self.smear)

    def predict_quantile(self, X: np.ndarray, q: float) -> np.ndarray:
        return np.maximum(1.0, np.exp(self.model.predict(X) + np.quantile(self.log_errors, q)))


def crossfit_predict(X: np.ndarray, tokens: np.ndarray, n_folds: int = 5, seed: int = 0,
                     learner: str = "ridge", quantile: float | None = None) -> np.ndarray:
    """Out-of-fold forecasts: the point (mean) estimate, or the q-quantile."""
    pred = np.zeros(len(tokens))
    for train, test in KFold(n_splits=n_folds, shuffle=True, random_state=seed).split(X):
        m = TokenPredictor(learner).fit(X[train], tokens[train], quantiles=quantile is not None)
        pred[test] = m.predict(X[test]) if quantile is None else m.predict_quantile(X[test], quantile)
    return pred


def crossfit_constant(tokens: np.ndarray, n_folds: int = 5, seed: int = 0) -> np.ndarray:
    """Baseline forecaster: every task costs the training-fold mean."""
    pred = np.zeros(len(tokens))
    for train, test in KFold(n_splits=n_folds, shuffle=True, random_state=seed).split(tokens):
        pred[test] = tokens[train].mean()
    return pred


def _folds(n: int, n_folds: int, seed: int):
    return KFold(n_splits=n_folds, shuffle=True, random_state=seed).split(np.zeros(n))


def crossfit_group_mean(groups: list[str], tokens: np.ndarray, n_folds: int = 5, seed: int = 0) -> np.ndarray:
    """Per-benchmark average: "a math task costs what math tasks cost on average".
    The strong simple baseline any learned model has to beat to justify itself."""
    groups = np.asarray(groups)
    pred = np.zeros(len(tokens))
    for train, test in _folds(len(tokens), n_folds, seed):
        means = {g: tokens[train][groups[train] == g].mean() for g in set(groups[train])}
        pred[test] = [means.get(g, tokens[train].mean()) for g in groups[test]]
    return pred


def crossfit_knn(emb: np.ndarray, tokens: np.ndarray, k: int = 10, n_folds: int = 5, seed: int = 0) -> np.ndarray:
    """Average cost of the k most similar training tasks (cosine similarity of
    normalized sentence embeddings)."""
    pred = np.zeros(len(tokens))
    for train, test in _folds(len(tokens), n_folds, seed):
        sims = emb[test] @ emb[train].T
        nearest = np.argpartition(-sims, kth=min(k, len(train) - 1), axis=1)[:, :k]
        pred[test] = tokens[train][nearest].mean(axis=1)
    return pred


def crossfit_buckets(X: np.ndarray, tokens: np.ndarray, n_buckets: int = 3, n_folds: int = 5, seed: int = 0) -> np.ndarray:
    """Classify each task as short / medium / long (training-fold terciles) and
    forecast that bucket's median cost — coarser, but robust to outliers."""
    from sklearn.linear_model import LogisticRegression

    pred = np.zeros(len(tokens))
    for train, test in _folds(len(tokens), n_folds, seed):
        edges = np.quantile(tokens[train], np.linspace(0, 1, n_buckets + 1)[1:-1])
        labels = np.digitize(tokens[train], edges)
        medians = np.array([np.median(tokens[train][labels == b]) for b in range(n_buckets)])
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=3000)).fit(X[train], labels)
        pred[test] = medians[clf.predict(X[test])]
    return pred


def rep_change_per_call(bonus_rate: float, penalty_rate: float) -> float:
    """Average reputation an honest agent gains per call. Below zero, honest
    agents drift toward removal: the 'cliff' from E7."""
    return config.REPUTATION_BONUS * bonus_rate - config.REPUTATION_PENALTY * penalty_rate


def tokengrid_accuracy(pred: np.ndarray, actual: np.ndarray) -> np.ndarray:
    """The reputation ledger's own per-call score."""
    p, a = np.maximum(pred, 1), np.maximum(actual, 1)
    return np.minimum(p, a) / np.maximum(p, a)


def forecast_quality(pred: np.ndarray, actual: np.ndarray) -> dict:
    acc = tokengrid_accuracy(pred, actual)
    return {
        "mae_tokens": float(np.abs(pred - actual).mean()),
        "spearman": float(spearmanr(pred, actual).statistic),
        "mean_tokengrid_accuracy": float(acc.mean()),
        # Shares of calls that would earn the ledger's bonus / penalty for an
        # agent forecasting with this predictor *honestly*.
        "frac_bonus": float((acc > config.REPUTATION_ACCURACY_HIGH).mean()),
        "frac_penalty": float((acc < config.REPUTATION_ACCURACY_LOW).mean()),
        # Share of tasks whose real cost stayed at or under the forecast —
        # for a q-quantile forecast this should be close to q.
        "coverage": float((actual <= pred).mean()),
        "rep_change_per_call": rep_change_per_call(float((acc > config.REPUTATION_ACCURACY_HIGH).mean()),
                                                   float((acc < config.REPUTATION_ACCURACY_LOW).mean())),
    }
