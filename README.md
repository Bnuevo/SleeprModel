# SleeprModel

Predict future NFL player opportunities for dynasty fantasy football, with Sleeper league context.

## Status

Project scaffold and modeling plan. No model has been trained and no player recommendations have been generated.

## League

Sleeper league ID: `1312319914337402880`

League settings and rosters must be fetched and verified before tailoring recommendations. The owner's team has not yet been identified.

## First model

Predict next-season wide receiver target share and probability of gaining at least five percentage points of target share. Report uncertainty alongside predictions.

Separate usage when active from season totals. Define target share consistently using team pass attempts in the games a player is active. Handle players with no following-season NFL appearances explicitly rather than silently dropping them.

## Roadmap

1. Import Sleeper league settings, users, and rosters through its read-only API; cache raw responses locally.
2. Join historical NFL usage data with stable player identifiers; audit unmatched players.
3. Build historical player-season features: usage trajectory, target earning, age, experience, draft capital, and competition.
4. Benchmark against unchanged prior-season usage and simple regression.
5. Train gradient-boosted models and evaluate on later seasons with rolling time splits.
6. Produce roster outlooks, trade research candidates, and available stashes.
7. Extend to running backs and tight ends with position-specific targets.

## Evaluation

Use only information available at the prediction date. Keep offseason roster changes in explicit scenarios until known. Measure target-share error, classification calibration, and performance versus baselines. Do not treat model scores as validated probabilities until calibrated on held-out seasons.

## Data

- Sleeper API: https://docs.sleeper.com/
- NFL historical data: https://github.com/nflverse/nflverse-data

Check source licensing, availability, and publication dates before ingestion. Keep raw league/member data out of version control. Scoring and roster format determine the value of predicted usage, not the underlying NFL usage prediction itself.
