"""Fixed chronological train/calibration/test separation; no shuffled validation."""
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder

from .features import CATEGORICAL, FEATURES, NUMERIC


def estimator(quantile=None):
    preprocessing = ColumnTransformer([
        ("numeric", SimpleImputer(strategy="constant", fill_value=-1, keep_empty_features=True), NUMERIC),
        ("position", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
    ])
    return make_pipeline(preprocessing, HistGradientBoostingRegressor(
        loss="squared_error" if quantile is None else "quantile", quantile=quantile,
        max_iter=100, max_leaf_nodes=15, l2_regularization=10,
        learning_rate=.07, early_stopping=False, random_state=42))


def fit(train, calibration):
    if train.empty or calibration.empty or train.season.max() >= calibration.season.min():
        raise ValueError("Training seasons must strictly precede nonempty calibration seasons")
    models = [estimator(q).fit(train[FEATURES], train.ppr) for q in (None, .1, .9)]
    lower, upper = sorted_bounds(models, calibration)
    scores = np.maximum(lower - calibration.ppr.to_numpy(), calibration.ppr.to_numpy() - upper)
    # Finite-sample split-conformal correction, expand only. No coverage guarantee under temporal drift.
    level = min(1., np.ceil((len(scores) + 1) * .8) / len(scores))
    correction = max(0., float(np.quantile(scores, level, method="higher")))
    return {"models": models, "correction": correction,
            "position_means": train.groupby("position").ppr.mean().to_dict(),
            "trained_through_season": int(train.season.max()),
            "calibrated_through_season": int(calibration.season.max()),
            "scoring": "PPR: pass yards .04, pass TD 4, INT -2; rush/rec yards .1, TD 6; reception 1; lost fumble -2; 2pt 2"}


def sorted_bounds(models, rows):
    a, b = models[1].predict(rows[FEATURES]), models[2].predict(rows[FEATURES])
    return np.minimum(a, b), np.maximum(a, b)


def predict(bundle, rows):
    lower, upper = sorted_bounds(bundle["models"], rows)
    result = rows[["player_id", "season", "week", "position", "team", "opponent", "history_games"]].copy()
    result["projected_ppr"] = bundle["models"][0].predict(rows[FEATURES])
    result["lower_80"] = lower - bundle["correction"]
    result["upper_80"] = upper + bundle["correction"]
    result["recent_average"] = rows.ppr_last4.fillna(rows.position.map(bundle["position_means"]))
    result["position_average"] = rows.position.map(bundle["position_means"])
    return result


def evaluate(bundle, rows):
    predictions = predict(bundle, rows)
    predictions["actual_ppr"] = rows.ppr
    def metrics(subset):
        y = subset.actual_ppr
        return {"n": len(subset), **{name: {
            "mae": float(mean_absolute_error(y, subset[col])),
            "rmse": float(np.sqrt(mean_squared_error(y, subset[col]))),
        } for name, col in [("model", "projected_ppr"), ("last4", "recent_average"), ("position", "position_average")]},
                "interval_coverage": float(((y >= subset.lower_80) & (y <= subset.upper_80)).mean()),
                "mean_interval_width": float((subset.upper_80 - subset.lower_80).mean())}
    relevant = predictions.loc[predictions.recent_average >= np.where(predictions.position == "QB", 8, 3)]
    return {"overall": metrics(predictions),
            "by_position": {p: metrics(s) for p, s in predictions.groupby("position")},
            "prior_usage_cohort": metrics(relevant) if len(relevant) else None}, predictions
