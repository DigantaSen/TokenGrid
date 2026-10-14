import numpy as np

from app.token_predictor import crossfit_constant, crossfit_predict, forecast_quality, tokengrid_accuracy


def test_ledger_accuracy_is_symmetric():
    assert tokengrid_accuracy(np.array([100.0]), np.array([300.0]))[0] == tokengrid_accuracy(np.array([300.0]), np.array([100.0]))[0]


def test_learned_beats_constant_when_signal_exists():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(600, 5))
    tokens = np.exp(5 + 0.8 * X[:, 0] + 0.2 * rng.normal(size=600))
    learned = forecast_quality(crossfit_predict(X, tokens), tokens)
    const = forecast_quality(crossfit_constant(tokens), tokens)
    assert learned["spearman"] > 0.8
    assert learned["mae_tokens"] < const["mae_tokens"]
    assert learned["mean_tokengrid_accuracy"] > const["mean_tokengrid_accuracy"]


def test_percentile_forecast_is_calibrated_and_ordered():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(800, 4))
    tokens = np.exp(5 + 0.6 * X[:, 0] + 0.4 * rng.normal(size=800))
    p50, p90 = (crossfit_predict(X, tokens, quantile=q) for q in (0.5, 0.9))
    assert abs((tokens <= p50).mean() - 0.5) < 0.06
    assert abs((tokens <= p90).mean() - 0.9) < 0.05
    assert np.all(p90 >= p50)


def test_lgbm_learner_runs_and_finds_signal():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(500, 5))
    tokens = np.exp(5 + 0.8 * X[:, 0] + 0.2 * rng.normal(size=500))
    assert forecast_quality(crossfit_predict(X, tokens, learner="lgbm"), tokens)["spearman"] > 0.7


def test_group_mean_knn_and_buckets_are_cross_fitted_and_sensible():
    from app.token_predictor import crossfit_buckets, crossfit_group_mean, crossfit_knn
    rng = np.random.default_rng(3)
    groups = ["a"] * 300 + ["b"] * 300
    tokens = np.r_[rng.normal(200, 20, 300), rng.normal(800, 50, 300)].clip(1)
    gm = crossfit_group_mean(groups, tokens)
    assert abs(gm[:300].mean() - 200) < 10 and abs(gm[300:].mean() - 800) < 15

    emb = np.r_[np.tile([1.0, 0.0], (300, 1)), np.tile([0.0, 1.0], (300, 1))] + rng.normal(0, 0.05, (600, 2))
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    knn = crossfit_knn(emb, tokens)
    assert np.abs(knn - np.r_[[200] * 300, [800] * 300]).mean() < 40

    buckets = crossfit_buckets(emb, tokens)
    assert len(set(np.round(buckets))) <= 15          # a few bucket medians, not a continuum
    assert forecast_quality(buckets, tokens)["spearman"] > 0.6


def test_median_learner_skips_mean_correction():
    from app.token_predictor import TokenPredictor
    rng = np.random.default_rng(4)
    X = rng.normal(size=(400, 3))
    tokens = np.exp(5 + 0.5 * X[:, 0] + 0.8 * rng.normal(size=400))
    assert TokenPredictor("lgbm_median").fit(X, tokens).smear == 1.0
    assert TokenPredictor("lgbm").fit(X, tokens).smear > 1.0   # mean > median for skewed costs
