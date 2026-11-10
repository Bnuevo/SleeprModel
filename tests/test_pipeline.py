import tempfile
import unittest
from pathlib import Path

import pandas as pd
from pandas.testing import assert_frame_equal

from sleepr.data import STATS, load_stats
from sleepr.features import FEATURES, build_features
from sleepr.model import evaluate, fit


def fixture():
    rows = []
    for season in (2021, 2022, 2023):
        for week in range(1, 7):
            for player in range(4):
                row = {s: 0 for s in STATS}
                row.update(player_id=str(player), season=season, week=week, season_type="REG",
                           position="WR", team="A", opponent="B", receptions=player + week,
                           targets=player + week + 2, receiving_yards=10 * (player + week))
                rows.append(row)
    return pd.DataFrame(rows)


class PipelineTests(unittest.TestCase):
    def stats(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stats.csv"
            fixture().to_csv(path, index=False)
            return load_stats([path])

    def test_scoring_and_duplicate_rejection(self):
        frame = fixture()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stats.csv"
            frame.to_csv(path, index=False)
            self.assertEqual(load_stats([path]).iloc[0].ppr, 2.)
            frame.loc[0, "passing_interceptions"] = 2
            frame.to_csv(path, index=False)
            self.assertEqual(load_stats([path], {"passing_interceptions": -1}).iloc[0].ppr, -2.)
            frame.loc[0, ["special_teams_tds", "fumble_recovery_tds", "fumbles_lost_total"]] = [1, 1, 1]
            frame.to_csv(path, index=False)
            self.assertEqual(load_stats([path]).iloc[0].ppr, 8.)
            pd.concat([frame, frame.iloc[:1]]).to_csv(path, index=False)
            with self.assertRaises(ValueError):
                load_stats([path])

    def test_current_and_future_outcomes_cannot_change_features(self):
        stats = self.stats()
        before = build_features(stats)
        poisoned = stats.copy()
        poisoned.loc[(poisoned.season == 2023) & (poisoned.week >= 4), ["ppr", "targets"]] = 9999
        after = build_features(poisoned)
        mask = (before.season == 2023) & (before.week == 4)
        assert_frame_equal(before.loc[mask, FEATURES], after.loc[mask, FEATURES])

    def test_candidate_matches_historical_and_ignores_supplied_outcomes(self):
        stats = self.stats()
        candidate = stats.loc[(stats.season == 2023) & (stats.week == 4)].copy()
        candidate["ppr"] = 99999
        forecast = build_features(stats, candidate)
        historical = build_features(stats)
        expected = historical.loc[(historical.season == 2023) & (historical.week == 4)]
        assert_frame_equal(forecast[FEATURES].reset_index(drop=True), expected[FEATURES].reset_index(drop=True))
        self.assertTrue(forecast.ppr.isna().all())

    def test_new_season_resets_history(self):
        rows = build_features(self.stats())
        self.assertTrue((rows.loc[rows.week == 1, "history_games"] == 0).all())
        self.assertTrue(rows.loc[rows.week == 1, "ppr_last4"].isna().all())

    def test_model_and_interval_smoke(self):
        rows = build_features(self.stats())
        bundle = fit(rows.loc[rows.season == 2021], rows.loc[rows.season == 2022])
        report, predictions = evaluate(bundle, rows.loc[rows.season == 2023])
        self.assertEqual(report["overall"]["n"], 24)
        self.assertTrue((predictions.lower_80 <= predictions.upper_80).all())
        self.assertTrue(predictions.projected_ppr.notna().all())
        with self.assertRaises(ValueError):
            fit(rows.loc[rows.season == 2023], rows.loc[rows.season == 2022])


if __name__ == "__main__":
    unittest.main()
