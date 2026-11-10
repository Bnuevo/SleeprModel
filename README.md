# SleeprModel

Weekly PPR projection research for start/sit decisions in The Highlanders. The first version trains a gradient-boosted scoring model, estimates an 80% prediction range, and evaluates against recent-average and position-average baselines.

Experimental: no advantage over Sleeper or the existing Highlanders model has been established. Availability, lineup optimization, calibrated boom/bust probabilities and dashboard integration remain future work.

## Setup

Python 3.11 or later:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m unittest discover -s tests -v
```

On this machine the system Python launcher is broken. The working interpreter is `/Library/Frameworks/Python.framework/Versions/3.13/bin/python3`.

## Download and evaluate

```sh
.venv/bin/python -m sleepr fetch --seasons 2021 2022 2023 2024 2025
.venv/bin/python -m sleepr train --stats data/raw/weekly_2021.csv data/raw/weekly_2022.csv data/raw/weekly_2023.csv data/raw/weekly_2024.csv data/raw/weekly_2025.csv
```

The default split fits model parameters on 2021–2023, adjusts intervals on 2024, and evaluates on 2025. Test-week features use earlier observed results, as they would during a season. Test outcomes never fit model parameters or interval adjustments. This is one chronological holdout, not yet a multi-fold rolling backtest. Random internal early stopping is disabled.

Outputs, ignored by Git:

- `reports/evaluation.json`: overall, position-level and prior-usage-cohort errors and interval coverage.
- `reports/heldout_predictions.csv`: individual held-out forecasts and actuals.
- `models/weekly.pkl`: model, scoring weights and training/calibration cutoffs. Load only trusted local pickle files.

The prior-usage cohort includes prior-four-observed-game averages of at least 8 points for QB or 3 for other positions. Cold starts fall back to training-only position averages. This is a rough cohort, not a validated starter population.

## Forecast candidates

Supply a CSV with `player_id,season,week,position,team,opponent`. IDs must be nflverse GSIS IDs, not Sleeper IDs. All candidates must be for one week in a season after calibration. Select candidates using a pregame roster/schedule source, never actual participation that week. Only QB/RB/WR/TE are supported.

```sh
.venv/bin/python -m sleepr predict --stats data/raw/weekly_2025.csv --candidates candidates.csv
```

This writes `reports/weekly_projections.csv`, ranked by mean points. Include current-season stats for current-season forecasts. Same-week and later stats are discarded. Features reset each season, so rookies and week-one players rely heavily on position priors. Predictions are conditional on playing: verify byes, injuries and lineup eligibility separately. Negative lower bounds are permitted because fantasy scoring can be negative.

## Archive Sleeper projections

```sh
.venv/bin/python -m sleepr snapshot --season 2026 --week 11
```

Each call creates a raw snapshot with UTC retrieval time, source URL and SHA-256 hash. No recurring schedule is installed. Retrieval time alone does not establish pregame status: kickoff times and player-ID joins are still needed before benchmarking. Historical projections returned now must not be treated as archived pregame forecasts.

## Model and scoring

Features use strictly prior weeks: player points, attempts, carries, targets, catches, yards, observed-game count, gap since last observed game, and opponent position-total points allowed. One- and four-game averages capture recent form. Identity and team labels are not learned as shortcuts. Training-only preprocessing handles missing history.

[Scoring configuration](config/scoring.json) matches offensive rules in the existing Highlanders dashboard snapshot, including full PPR, four-point passing TDs and **-1 per interception**. Refresh settings before production use. Scoring includes special-teams TDs, recovery TDs, total lost fumbles and two-point conversions. Kicker and team-defense projections are outside this version. Scoring is embedded in the model and reused at inference. [League configuration](config/league.json) holds league/owner identifiers; no live roster import exists yet.

Mean and 10th/90th percentile estimates use scikit-learn histogram gradient boosting. Quantile crossings are reordered; a separate calibration season supplies a split-conformal expansion. Ranges target 80% marginal coverage, not coverage guarantees for each player. Seasonal drift and dependent player-games can invalidate nominal coverage: inspect held-out coverage. No head-to-head probabilities are inferred from a Gaussian or independence assumption.

## Data and limitations

Source: [nflverse weekly stats](https://github.com/nflverse/nflverse-data/releases/tag/stats_player), produced by `nflfastR::calculate_stats()`. See the [upstream repository](https://github.com/nflverse/nflverse-data) and [availability schedule](https://nflreadr.nflverse.com/articles/nflverse_data_schedule.html). Respect source-specific attribution and redistribution terms; this code license does not relicense upstream data. Raw data is not committed or redistributed.

Cached downloads preserve source URL, retrieval time and content hash. Later corrections remain possible: a lagged historical value is not proof it was published before kickoff. Missing required columns and duplicate identifiers fail explicitly. The evaluation population is regular-season player-weeks represented in the stats feed; inactive and zero-stat player-weeks may be absent. Errors therefore do not establish full-roster start/sit performance.

## Next milestones

1. Add kickoff times and Sleeper↔GSIS mappings with unmatched-ID audits.
2. Join pregame rosters and availability; retain inactive and zero-opportunity outcomes explicitly.
3. Add snaps, role changes, prior-season priors and injury scenarios with publication cutoffs.
4. Run rolling season folds and compare against archived Sleeper forecasts and the [existing model](https://github.com/Bnuevo/highlanders-advanced-metrics/blob/main/site/dist/matchup-model.js).
5. Validate boom/bust and start/sit probabilities; enforce flex/superflex eligibility and evaluate lineup decisions.
6. Export versioned forecasts to the Highlanders dashboard after validation.

The earlier dynasty target-share idea is deferred while weekly PPR projections are the priority.
