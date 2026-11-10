import argparse
import json
import pickle
import hashlib
import platform
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import sklearn

from .data import download, fetch_seasons, load_stats
from .features import build_features
from .model import evaluate, fit, predict


def main():
    parser = argparse.ArgumentParser(description="Weekly PPR research pipeline")
    sub = parser.add_subparsers(dest="command", required=True)
    fetch = sub.add_parser("fetch")
    fetch.add_argument("--seasons", nargs="+", type=int, required=True)
    train = sub.add_parser("train")
    train.add_argument("--calibration-season", type=int, default=2024)
    train.add_argument("--test-season", type=int, default=2025)
    train.add_argument("--stats", nargs="+", required=True)
    train.add_argument("--scoring", default="config/scoring.json")
    forecast = sub.add_parser("predict")
    forecast.add_argument("--stats", nargs="+", required=True)
    forecast.add_argument("--candidates", required=True)
    forecast.add_argument("--model", default="models/weekly.pkl")
    snapshot = sub.add_parser("snapshot")
    snapshot.add_argument("--season", type=int, required=True)
    snapshot.add_argument("--week", type=int, required=True)
    args = parser.parse_args()
    if args.command == "fetch":
        print(*fetch_seasons(args.seasons), sep="\n")
    elif args.command == "snapshot":
        if not 1 <= args.week <= 18:
            parser.error("Week must be between 1 and 18")
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        path = download(f"https://api.sleeper.app/projections/nfl/{args.season}/{args.week}?season_type=regular",
                        f"data/snapshots/{args.season}/{args.week}/{stamp}.json")
        print(f"Saved {path}; retrieval time alone does not establish pregame status.")
    elif args.command == "train":
        if args.calibration_season >= args.test_season:
            parser.error("Calibration must precede test season")
        weights = json.loads(Path(args.scoring).read_text())
        rows = build_features(load_stats(args.stats, weights))
        train_rows = rows.loc[rows.season < args.calibration_season]
        calibration = rows.loc[rows.season == args.calibration_season]
        test = rows.loc[rows.season == args.test_season]
        if test.empty:
            parser.error("No rows for test season")
        bundle = fit(train_rows, calibration)
        bundle["scoring"] = weights
        metrics, predictions = evaluate(bundle, test)
        report = {"train_seasons": sorted(map(int, train_rows.season.unique())),
                  "source_files": [{"path": p, "sha256": hashlib.sha256(Path(p).read_bytes()).hexdigest()} for p in args.stats],
                  "versions": {"python": platform.python_version(), "pandas": pd.__version__, "sklearn": sklearn.__version__},
                  "generated_at": datetime.now(timezone.utc).isoformat(),
                  "interval_expansion": bundle["correction"],
                  "calibration_season": args.calibration_season, "test_season": args.test_season,
                  "scoring": bundle["scoring"], "metrics": metrics,
                  "limitations": ["Conditional on a player-week existing in the stats feed; not an availability model.",
                                  "Historical source data may contain later corrections; not a point-in-time archive.",
                                  "No comparison to archived Sleeper forecasts yet.",
                                  "Intervals target 80% marginal coverage; temporal drift can reduce coverage."]}
        Path("models").mkdir(exist_ok=True)
        Path("reports").mkdir(exist_ok=True)
        with open("models/weekly.pkl", "wb") as handle:
            pickle.dump(bundle, handle)
        Path("reports/evaluation.json").write_text(json.dumps(report, indent=2))
        predictions.to_csv("reports/heldout_predictions.csv", index=False)
        print(json.dumps(report, indent=2))
    else:
        # Pickle is for locally trained, trusted artifacts only.
        with open(args.model, "rb") as handle:
            bundle = pickle.load(handle)
        candidates = pd.read_csv(args.candidates, dtype={"player_id": str})
        if candidates.empty or candidates.season.min() <= bundle["calibrated_through_season"]:
            parser.error("Forecast season must follow calibration season")
        rows = build_features(load_stats(args.stats, bundle["scoring"]), candidates)
        output = predict(bundle, rows).sort_values("projected_ppr", ascending=False)
        Path("reports").mkdir(exist_ok=True)
        output.to_csv("reports/weekly_projections.csv", index=False)
        print("Saved reports/weekly_projections.csv (conditional on playing; verify availability separately)")


if __name__ == "__main__":
    main()
