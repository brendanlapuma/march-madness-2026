# Copilot Instructions For march-madness-2026

## Purpose
This repository builds March Madness game win probabilities from historical NCAA data.
The main workflow is implemented in `winner-2025.ipynb`, which produced `predictions.csv` for a prior competition run.

When assisting in this repo, prioritize improving and validating the modeling pipeline for the current season while keeping outputs compatible with Kaggle submission format (`ID`, `Pred`).

## Environment And Data
- Data lives in `data/` and comes from Kaggle March Machine Learning Mania 2026.
- Core notebook uses:
  - `numpy`, `pandas`, `matplotlib`, `seaborn`
  - `statsmodels`, `tqdm`
  - `scikit-learn`, `xgboost`, `scipy`
- Project uses Poetry (`pyproject.toml`).
- If dependency resolution fails, check Python version compatibility first (`pyproject.toml` currently has `requires-python = ">=3.14"`, which may need adjustment depending on local environment/tooling support).

## Notebook Pipeline (`winner-2025.ipynb`)
1. Load and combine data:
- Reads men and women regular-season detailed results, tournament detailed results, and seeds.
- Concatenates men + women into unified frames.
- Filters seasons with cutoff `season = 2003`.

2. Prepare doubled game rows:
- `prepare_data(df)` normalizes counting stats for overtime using `(40 + 5 * NumOT) / 40`.
- Creates two rows per game by swapping winner/loser into symmetric `T1_*` vs `T2_*` views.
- Adds targets/features:
  - `PointDiff = T1_Score - T2_Score`
  - `win = 1 if PointDiff > 0 else 0`
  - `men_women = 1` for men IDs starting with `1`, else `0` for women

3. Easy features:
- Extract numeric seed from `Seed` string.
- Merge `T1_seed` and `T2_seed`.
- Compute `Seed_diff = T2_seed - T1_seed`.

4. Medium features:
- Build season average box-score features from regular season grouped by `Season` and `T1_TeamID`.
- For each side (T1 and T2), include team averages and opponent-against-team averages.

5. Hard features (Elo):
- Compute season-level Elo ratings from regular-season wins only.
- Initialize at 1000 each season.
- Parameters: `elo_width=400`, `k_factor=100`.
- Merge `T1_elo`, `T2_elo`, and `elo_diff`.

6. Hardest features (GLM quality):
- Fit Gaussian GLM per season and gender group with formula:
  - `PointDiff ~ -1 + T1_TeamID + T2_TeamID`
- Restricts modeled team set to tourney teams plus selected non-tourney upset-capable teams.
- Produces `T1_quality`, `T2_quality`, and `diff_quality`.

7. Model training:
- Uses XGBoost regression (`reg:squarederror`) to predict point differential.
- Trains leave-one-season-out models (one model per held-out season).
- Tracks MAE and collects out-of-fold predictions.

8. Probability calibration:
- Converts predicted margins to win probabilities via `UnivariateSpline`.
- Clips probabilities to `[0.01, 0.99]`.
- Evaluates with Brier score.

9. Submission generation:
- Loads `SampleSubmissionStage2.csv`.
- Recreates all features for every matchup ID.
- Predicts with each season model, averages probabilities.
- Applies manual confidence adjustment:
  - increase prediction by 10% for values below 0.85
- Writes `predictions.csv`.

## Important Conventions And Caveats
- Gender encoding is inconsistent in comments vs values in some cells. Keep behavior consistent with code unless intentionally refactoring.
- Several transformations mutate dataframes in place. Prefer explicit `.copy()` when refactoring to avoid `SettingWithCopy` issues.
- Notebook currently mixes exploration, feature engineering, training, and inference in one linear flow. Favor extracting reusable functions/modules before major model changes.

## Guidance For Future Copilot Sessions
When asked to improve results for the new season:
1. Preserve current notebook behavior first, then introduce one controlled change at a time.
2. Add measurable validation for each change (MAE, Brier, and if possible log loss on out-of-fold data).
3. Validate feature availability at submission time before adding new features.
4. If large refactors are requested, move logic into Python modules and keep notebook as an orchestrator/analysis layer.

## High-Value Improvement Areas (2026)
- Hyperparameter search with season-aware CV and reproducible seeds.
- Better calibration and calibration-by-gender checks.
- Explicit handling for missing seeds/ratings at inference.
- Replace deprecated seaborn calls (for example `distplot`) with current alternatives.

## Output Contract
- Final submission file must be `predictions.csv` with columns:
  - `ID`
  - `Pred`
- `Pred` must remain clipped to valid probability range `(0, 1)`.
