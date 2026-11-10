"""Every feature uses strictly earlier weeks within the same season."""
import numpy as np
import pandas as pd

USAGE = ["ppr", "attempts", "carries", "targets", "receptions", "passing_yards",
         "rushing_yards", "receiving_yards"]
NUMERIC = [f"{s}_last{n}" for s in USAGE for n in (1, 4)] + ["history_games", "weeks_since_game", "opponent_allowed_last4"]
CATEGORICAL = ["position"]
FEATURES = NUMERIC + CATEGORICAL


def build_features(stats, candidates=None):
    history = stats.copy()
    history["is_candidate"] = False
    if candidates is not None:
        candidates = candidates.copy()
        required = ["player_id", "season", "week", "position", "team", "opponent"]
        if set(required) - set(candidates):
            raise ValueError(f"Candidates require {required}")
        if candidates[required].isna().any().any() or candidates.duplicated(["player_id", "season", "week"]).any():
            raise ValueError("Candidates have missing fields or duplicate identifiers")
        if not candidates.position.isin(["QB", "RB", "WR", "TE"]).all():
            raise ValueError("Only QB, RB, WR and TE candidates are supported")
        # One forecast week per invocation; strip outcomes supplied by callers.
        if len(candidates[["season", "week"]].drop_duplicates()) != 1:
            raise ValueError("Candidates must have a single season and week")
        season, week = candidates.iloc[0][["season", "week"]]
        history = history.loc[(history.season < season) | ((history.season == season) & (history.week < week))]
        candidates = candidates[required].assign(is_candidate=True)
        history = pd.concat([history, candidates], ignore_index=True)
    history = history.sort_values(["player_id", "season", "week"]).reset_index(drop=True)
    group = history.groupby(["player_id", "season"], sort=False)
    for stat in USAGE:
        for window in (1, 4):
            history[f"{stat}_last{window}"] = group[stat].transform(
                lambda s: s.shift(1).rolling(window, min_periods=1).mean())
    history["history_games"] = group.cumcount()
    history["weeks_since_game"] = history.week - group.week.shift(1)
    # Position-total points conceded per opponent/week, then lag before joining.
    allowed = stats.groupby(["season", "opponent", "position", "week"], as_index=False).ppr.sum()
    lookup = {}
    for key, rows in allowed.groupby(["season", "opponent", "position"]):
        lookup[key] = rows.sort_values("week")
    def prior_opponent(row):
        rows = lookup.get((row.season, row.opponent, row.position))
        if rows is None:
            return np.nan
        return rows.loc[rows.week < row.week, "ppr"].tail(4).mean()
    history["opponent_allowed_last4"] = [prior_opponent(r) for r in history.itertuples()]
    return history.loc[history.is_candidate].copy() if candidates is not None else history
