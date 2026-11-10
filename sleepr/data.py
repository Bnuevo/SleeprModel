"""Cached public data with retrieval provenance. Never infer pregame availability."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

import pandas as pd

STATS_URL = "https://github.com/nflverse/nflverse-data/releases/download/stats_player/stats_player_week_{season}.csv"
POSITIONS = ("QB", "RB", "WR", "TE")
KEY = ["player_id", "season", "week"]
STATS = ["passing_yards", "passing_tds", "passing_interceptions", "rushing_yards",
         "rushing_tds", "receptions", "receiving_yards", "receiving_tds",
         "sack_fumbles_lost", "rushing_fumbles_lost", "receiving_fumbles_lost",
         "passing_2pt_conversions", "rushing_2pt_conversions", "receiving_2pt_conversions",
         "attempts", "carries", "targets", "special_teams_tds", "fumble_recovery_tds", "fumbles_lost_total"]
WEIGHTS = {"passing_yards": .04, "passing_tds": 4, "passing_interceptions": -2,
           "rushing_yards": .1, "rushing_tds": 6, "receptions": 1,
           "receiving_yards": .1, "receiving_tds": 6, "fumbles_lost_total": -2,
           "special_teams_tds": 6, "fumble_recovery_tds": 6,
           "passing_2pt_conversions": 2, "rushing_2pt_conversions": 2,
           "receiving_2pt_conversions": 2}


def download(url, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return path
    with urlopen(Request(url, headers={"User-Agent": "SleeprModel/0.1"}), timeout=90) as response:
        payload = response.read()
    # Exclusive creation preserves existing snapshots even on repeated runs.
    with path.open("xb") as handle:
        handle.write(payload)
    path.with_suffix(path.suffix + ".meta.json").write_text(json.dumps({
        "url": url, "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "historically_available_at": None,
    }, indent=2))
    return path


def fetch_seasons(seasons, root="data/raw"):
    return [download(STATS_URL.format(season=s), Path(root) / f"weekly_{s}.csv") for s in seasons]


def load_stats(paths, weights=None):
    weights = WEIGHTS if weights is None else weights
    frame = pd.concat([pd.read_csv(p, dtype={"player_id": str}, low_memory=False) for p in paths], ignore_index=True)
    frame = frame.rename(columns={"recent_team": "team", "opponent_team": "opponent"})
    required = set(KEY + ["season_type", "position", "team", "opponent"] + STATS)
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required NFL columns: {sorted(missing)}")
    frame = frame.loc[(frame.season_type == "REG") & frame.position.isin(POSITIONS)].copy()
    if frame[KEY].isna().any().any() or frame.duplicated(KEY).any():
        raise ValueError("Missing or duplicate player-season-week identifiers")
    for col in STATS:
        frame[col] = pd.to_numeric(frame[col], errors="raise").fillna(0)
    if not weights or set(weights) - set(frame.columns):
        raise ValueError("Scoring references unsupported stat columns")
    frame["ppr"] = sum(frame[col] * weight for col, weight in weights.items())
    return frame.sort_values(["season", "week", "player_id"]).reset_index(drop=True)
