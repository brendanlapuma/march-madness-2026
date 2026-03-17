import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import tqdm
import xgboost as xgb
from scipy.interpolate import UnivariateSpline
from sklearn.metrics import brier_score_loss, mean_absolute_error


BASELINE_FEATURES = [
    "men_women",
    "T1_seed",
    "T2_seed",
    "Seed_diff",
    "T1_Rk",
    "T2_Rk",
    "T1_avg_Score",
    "T1_avg_FGA",
    "T1_avg_OR",
    "T1_avg_DR",
    "T1_avg_Blk",
    "T1_avg_PF",
    "T1_avg_opponent_FGA",
    "T1_avg_opponent_Blk",
    "T1_avg_opponent_PF",
    "T1_avg_PointDiff",
    "T2_avg_Score",
    "T2_avg_FGA",
    "T2_avg_OR",
    "T2_avg_DR",
    "T2_avg_Blk",
    "T2_avg_PF",
    "T2_avg_opponent_FGA",
    "T2_avg_opponent_Blk",
    "T2_avg_opponent_PF",
    "T2_avg_PointDiff",
    "T1_elo",
    "T2_elo",
    "elo_diff",
    "T1_quality",
    "T2_quality",
]

BASELINE_PARAM = {
    "objective": "reg:squarederror",
    "booster": "gbtree",
    "eta": 0.0093,
    "subsample": 0.6,
    "colsample_bynode": 0.8,
    "num_parallel_tree": 2,
    "min_child_weight": 4,
    "max_depth": 4,
    "tree_method": "hist",
    "grow_policy": "lossguide",
    "max_bin": 38,
}
BASELINE_NUM_ROUNDS = 704


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tune feature subsets and XGBoost params for March Madness OOF Brier score."
    )
    parser.add_argument("--data-dir", default="data", help="Path to Kaggle data directory.")
    parser.add_argument("--season-cutoff", type=int, default=2003)
    parser.add_argument("--n-trials", type=int, default=40, help="Random search trials after baseline.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducible search. Omit for a different sequence each run.")
    parser.add_argument("--use-gpu", action="store_true", help="Request CUDA in XGBoost.")
    parser.add_argument(
        "--require-gpu",
        action="store_true",
        help="Fail immediately if CUDA training is not available instead of falling back to CPU.",
    )
    parser.add_argument("--results-csv", default="tuning_results.csv")
    parser.add_argument("--top-k", type=int, default=10, help="How many best trials to print.")
    return parser.parse_args()


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    # Keep only fields needed for feature engineering and model targets.
    df = df[
        [
            "Season",
            "DayNum",
            "LTeamID",
            "LScore",
            "WTeamID",
            "WScore",
            "NumOT",
            "LFGM",
            "LFGA",
            "LFGM3",
            "LFGA3",
            "LFTM",
            "LFTA",
            "LOR",
            "LDR",
            "LAst",
            "LTO",
            "LStl",
            "LBlk",
            "LPF",
            "WFGM",
            "WFGA",
            "WFGM3",
            "WFGA3",
            "WFTM",
            "WFTA",
            "WOR",
            "WDR",
            "WAst",
            "WTO",
            "WStl",
            "WBlk",
            "WPF",
        ]
    ].copy()

    # Normalize counting stats to regulation-equivalent pace.
    adjot = (40 + 5 * df["NumOT"]) / 40
    adjcols = [
        "LScore",
        "WScore",
        "LFGM",
        "LFGA",
        "LFGM3",
        "LFGA3",
        "LFTM",
        "LFTA",
        "LOR",
        "LDR",
        "LAst",
        "LTO",
        "LStl",
        "LBlk",
        "LPF",
        "WFGM",
        "WFGA",
        "WFGM3",
        "WFGA3",
        "WFTM",
        "WFTA",
        "WOR",
        "WDR",
        "WAst",
        "WTO",
        "WStl",
        "WBlk",
        "WPF",
    ]
    for col in adjcols:
        df[col] = df[col] / adjot

    # Double each game so both team orderings exist (T1 vs T2 and swapped).
    dfswap = df.copy()
    df.columns = [x.replace("W", "T1_").replace("L", "T2_") for x in list(df.columns)]
    dfswap.columns = [x.replace("L", "T1_").replace("W", "T2_") for x in list(dfswap.columns)]
    output = pd.concat([df, dfswap], ignore_index=True)
    output["PointDiff"] = output["T1_Score"] - output["T2_Score"]
    output["win"] = (output["PointDiff"] > 0).astype(int)
    # Keep the same encoding used in the notebook: 1 for men IDs that start with "1".
    output["men_women"] = output["T1_TeamID"].apply(lambda t: str(t).startswith("1")).astype(int)
    return output


def load_kenpom_features(data_dir: str, m_teams: pd.DataFrame, w_teams: pd.DataFrame) -> pd.DataFrame:
    """Load kenpom.csv and join with team IDs for both men and women."""
    kenpom_path = Path(data_dir) / "kenpom.csv"
    if not kenpom_path.exists():
        return None

    kenpom = pd.read_csv(kenpom_path)
    # Ignore Seed/Conf and tournament flag columns.
    drop_cols = [c for c in ["Seed", "Conf", "Tourney"] if c in kenpom.columns]
    kenpom = kenpom.drop(columns=drop_cols, errors="ignore")

    # If TeamID is already in kenpom, use it directly; otherwise join by team name.
    if "TeamID" not in kenpom.columns:
        # Create a normalized team name for joining.
        m_teams_join = m_teams[["TeamID", "TeamName"]].copy()
        m_teams_join["TeamNameNorm"] = m_teams_join["TeamName"].str.lower().str.replace(r"\s+", "", regex=True)
        w_teams_join = w_teams[["TeamID", "TeamName"]].copy()
        w_teams_join["TeamNameNorm"] = w_teams_join["TeamName"].str.lower().str.replace(r"\s+", "", regex=True)

        kenpom["TeamNorm"] = kenpom["Team"].str.lower().str.replace(r"\s+", "", regex=True)
        kenpom_m = pd.merge(kenpom, m_teams_join[["TeamID", "TeamNameNorm"]], left_on="TeamNorm", right_on="TeamNameNorm", how="left")
        kenpom_w = pd.merge(kenpom, w_teams_join[["TeamID", "TeamNameNorm"]], left_on="TeamNorm", right_on="TeamNameNorm", how="left")
        kenpom_m = kenpom_m.dropna(subset=["TeamID"])
        kenpom_w = kenpom_w.dropna(subset=["TeamID"])
        kenpom = pd.concat([kenpom_m, kenpom_w], ignore_index=True)

    kenpom = kenpom.dropna(subset=["Season", "TeamID"])
    kenpom["TeamID"] = kenpom["TeamID"].astype(int)
    kenpom["Season"] = kenpom["Season"].astype(int)

    # Select numerical features only (exclude Team, Tourney identifiers, and key columns)
    feature_cols = [c for c in kenpom.columns if c not in ["Season", "TeamID"] and kenpom[c].dtype in ["int64", "float64"]]
    kenpom = kenpom[["Season", "TeamID"] + feature_cols].copy()

    return kenpom


def expected_result(elo_a: float, elo_b: float, elo_width: float) -> float:
    return 1.0 / (1 + 10 ** ((elo_b - elo_a) / elo_width))


def update_elo(winner_elo: float, loser_elo: float, k_factor: float, elo_width: float) -> tuple[float, float]:
    expected_win = expected_result(winner_elo, loser_elo, elo_width)
    change_in_elo = k_factor * (1 - expected_win)
    winner_elo += change_in_elo
    loser_elo -= change_in_elo
    return winner_elo, loser_elo


def build_modeling_frame(data_dir: str, season_cutoff: int) -> pd.DataFrame:
    data_path = Path(data_dir)

    # Read men and women sources separately, then stack to match notebook behavior.
    m_regular = pd.read_csv(data_path / "MRegularSeasonDetailedResults.csv")
    m_tourney = pd.read_csv(data_path / "MNCAATourneyDetailedResults.csv")
    m_seeds = pd.read_csv(data_path / "MNCAATourneySeeds.csv")

    w_regular = pd.read_csv(data_path / "WRegularSeasonDetailedResults.csv")
    w_tourney = pd.read_csv(data_path / "WNCAATourneyDetailedResults.csv")
    w_seeds = pd.read_csv(data_path / "WNCAATourneySeeds.csv")

    regular_results = pd.concat([m_regular, w_regular], ignore_index=True)
    tourney_results = pd.concat([m_tourney, w_tourney], ignore_index=True)
    seeds = pd.concat([m_seeds, w_seeds], ignore_index=True)

    regular_results = regular_results.loc[regular_results["Season"] >= season_cutoff].copy()
    tourney_results = tourney_results.loc[tourney_results["Season"] >= season_cutoff].copy()
    seeds = seeds.loc[seeds["Season"] >= season_cutoff].copy()

    # Build symmetric game-level training frames.
    regular_data = prepare_data(regular_results)
    tourney_data = prepare_data(tourney_results)

    seeds["seed"] = seeds["Seed"].str[1:3].astype(int)

    seeds_t1 = seeds[["Season", "TeamID", "seed"]].copy()
    seeds_t2 = seeds[["Season", "TeamID", "seed"]].copy()
    seeds_t1.columns = ["Season", "T1_TeamID", "T1_seed"]
    seeds_t2.columns = ["Season", "T2_TeamID", "T2_seed"]

    tourney_data = tourney_data[["Season", "T1_TeamID", "T2_TeamID", "PointDiff", "win", "men_women"]].copy()
    tourney_data = pd.merge(tourney_data, seeds_t1, on=["Season", "T1_TeamID"], how="left")
    tourney_data = pd.merge(tourney_data, seeds_t2, on=["Season", "T2_TeamID"], how="left")
    tourney_data["Seed_diff"] = tourney_data["T2_seed"] - tourney_data["T1_seed"]

    # Season-average team and opponent stats from regular season.
    boxcols = [
        "T1_Score",
        "T1_FGM",
        "T1_FGA",
        "T1_FGM3",
        "T1_FGA3",
        "T1_FTM",
        "T1_FTA",
        "T1_OR",
        "T1_DR",
        "T1_Ast",
        "T1_TO",
        "T1_Stl",
        "T1_Blk",
        "T1_PF",
        "T2_Score",
        "T2_FGM",
        "T2_FGA",
        "T2_FGM3",
        "T2_FGA3",
        "T2_FTM",
        "T2_FTA",
        "T2_OR",
        "T2_DR",
        "T2_Ast",
        "T2_TO",
        "T2_Stl",
        "T2_Blk",
        "T2_PF",
        "PointDiff",
    ]

    ss = regular_data.groupby(["Season", "T1_TeamID"])[boxcols].mean().reset_index()

    ss_t1 = ss.copy()
    ss_t1.columns = ["T1_avg_" + x.replace("T1_", "").replace("T2_", "opponent_") for x in ss_t1.columns]
    ss_t1 = ss_t1.rename(columns={"T1_avg_Season": "Season", "T1_avg_TeamID": "T1_TeamID"})

    ss_t2 = ss.copy()
    ss_t2.columns = ["T2_avg_" + x.replace("T1_", "").replace("T2_", "opponent_") for x in ss_t2.columns]
    ss_t2 = ss_t2.rename(columns={"T2_avg_Season": "Season", "T2_avg_TeamID": "T2_TeamID"})

    tourney_data = pd.merge(tourney_data, ss_t1, on=["Season", "T1_TeamID"], how="left")
    tourney_data = pd.merge(tourney_data, ss_t2, on=["Season", "T2_TeamID"], how="left")

    # Elo ratings are computed from regular-season winners only, season by season.
    base_elo = 1000
    elo_width = 400
    k_factor = 100

    elos = []
    for season in sorted(set(seeds["Season"])):
        season_games = regular_data.loc[regular_data["Season"] == season].copy()
        season_games = season_games.loc[season_games["win"] == 1].reset_index(drop=True)
        teams = set(season_games["T1_TeamID"]) | set(season_games["T2_TeamID"])
        elo = dict(zip(teams, [base_elo] * len(teams)))

        for i in range(season_games.shape[0]):
            w_team = season_games.loc[i, "T1_TeamID"]
            l_team = season_games.loc[i, "T2_TeamID"]
            w_elo, l_elo = update_elo(elo[w_team], elo[l_team], k_factor, elo_width)
            elo[w_team] = w_elo
            elo[l_team] = l_elo

        elo_df = pd.DataFrame.from_dict(elo, orient="index").reset_index()
        elo_df = elo_df.rename(columns={"index": "TeamID", 0: "elo"})
        elo_df["Season"] = season
        elos.append(elo_df)

    elos = pd.concat(elos, ignore_index=True)
    elos_t1 = elos.rename(columns={"TeamID": "T1_TeamID", "elo": "T1_elo"})
    elos_t2 = elos.rename(columns={"TeamID": "T2_TeamID", "elo": "T2_elo"})

    tourney_data = pd.merge(tourney_data, elos_t1, on=["Season", "T1_TeamID"], how="left")
    tourney_data = pd.merge(tourney_data, elos_t2, on=["Season", "T2_TeamID"], how="left")
    tourney_data["elo_diff"] = tourney_data["T1_elo"] - tourney_data["T2_elo"]

    regular_data = regular_data.copy()
    seeds_t1 = seeds_t1.copy()
    seeds_t2 = seeds_t2.copy()

    regular_data["ST1"] = regular_data.apply(lambda t: f"{int(t['Season'])}/{int(t['T1_TeamID'])}", axis=1)
    regular_data["ST2"] = regular_data.apply(lambda t: f"{int(t['Season'])}/{int(t['T2_TeamID'])}", axis=1)
    seeds_t1["ST1"] = seeds_t1.apply(lambda t: f"{int(t['Season'])}/{int(t['T1_TeamID'])}", axis=1)
    seeds_t2["ST2"] = seeds_t2.apply(lambda t: f"{int(t['Season'])}/{int(t['T2_TeamID'])}", axis=1)

    # Restrict GLM pool to tournament teams + non-tourney teams that upset a tourney team.
    st = set(seeds_t1["ST1"]) | set(seeds_t2["ST2"])
    st = st | set(
        regular_data.loc[
            (regular_data["T1_Score"] > regular_data["T2_Score"]) & (regular_data["ST2"].isin(st)),
            "ST1",
        ]
    )

    dt = regular_data.loc[regular_data["ST1"].isin(st) | regular_data["ST2"].isin(st)].copy()
    dt["T1_TeamID"] = dt["T1_TeamID"].astype(str)
    dt["T2_TeamID"] = dt["T2_TeamID"].astype(str)
    dt.loc[~dt["ST1"].isin(st), "T1_TeamID"] = "0000"
    dt.loc[~dt["ST2"].isin(st), "T2_TeamID"] = "0000"

    def team_quality(season: int, men_women: int) -> pd.DataFrame:
        # Same per-season GLM quality formulation used in the notebook.
        formula = "PointDiff~-1+T1_TeamID+T2_TeamID"
        glm = sm.GLM.from_formula(
            formula=formula,
            data=dt.loc[(dt["Season"] == season) & (dt["men_women"] == men_women), :],
            family=sm.families.Gaussian(),
        ).fit()

        quality = pd.DataFrame(glm.params).reset_index()
        quality.columns = ["TeamID", "quality"]
        quality["Season"] = season
        quality = quality.loc[quality["TeamID"].str.contains("T1_")].reset_index(drop=True)
        quality["TeamID"] = quality["TeamID"].str[10:14].astype(int)
        return quality

    glm_quality = []
    seasons = sorted(set(seeds["Season"]))
    for s in tqdm.tqdm(seasons, unit="season", desc="GLM quality"):
        # Preserve the notebook's season thresholds for each league.
        if s >= 2010:
            glm_quality.append(team_quality(s, 0))
        if s >= 2003:
            glm_quality.append(team_quality(s, 1))

    glm_quality = pd.concat(glm_quality, ignore_index=True)
    glm_quality_t1 = glm_quality.rename(columns={"TeamID": "T1_TeamID", "quality": "T1_quality"})
    glm_quality_t2 = glm_quality.rename(columns={"TeamID": "T2_TeamID", "quality": "T2_quality"})

    tourney_data = pd.merge(tourney_data, glm_quality_t1, on=["Season", "T1_TeamID"], how="left")
    tourney_data = pd.merge(tourney_data, glm_quality_t2, on=["Season", "T2_TeamID"], how="left")
    tourney_data["diff_quality"] = tourney_data["T1_quality"] - tourney_data["T2_quality"]

    # Load and merge kenpom features.
    m_teams = pd.read_csv(data_path / "MTeams.csv")
    w_teams = pd.read_csv(data_path / "WTeams.csv")
    kenpom = load_kenpom_features(data_dir, m_teams, w_teams)
    if kenpom is not None:
        # Build rename dicts that preserve Season column
        kenpom_feature_cols = [c for c in kenpom.columns if c not in ["Season", "TeamID"]]
        rename_dict_t1 = {c: f"T1_{c}" for c in kenpom_feature_cols}
        rename_dict_t1["TeamID"] = "T1_TeamID"
        kenpom_t1 = kenpom.rename(columns=rename_dict_t1)
        
        rename_dict_t2 = {c: f"T2_{c}" for c in kenpom_feature_cols}
        rename_dict_t2["TeamID"] = "T2_TeamID"
        kenpom_t2 = kenpom.rename(columns=rename_dict_t2)
        
        tourney_data = pd.merge(tourney_data, kenpom_t1, on=["Season", "T1_TeamID"], how="left")
        tourney_data = pd.merge(tourney_data, kenpom_t2, on=["Season", "T2_TeamID"], how="left")

    return tourney_data


def get_candidate_groups(tourney_data: pd.DataFrame) -> dict[str, list[str]]:
    """Map group name -> list of columns. T1_/T2_ paired features are always toggled together."""
    protected = {"Season", "T1_TeamID", "T2_TeamID", "PointDiff", "win"}
    # These have no T1_/T2_ counterpart and are sampled individually.
    singles = {"men_women", "Seed_diff", "elo_diff", "diff_quality"}

    all_cols = {c for c in tourney_data.columns if c not in protected}
    groups: dict[str, list[str]] = {}

    for col in sorted(all_cols):
        if col in singles:
            groups[col] = [col]
        elif col.startswith("T1_"):
            base = col[3:]
            t2_col = "T2_" + base
            if t2_col in all_cols:
                # Group the pair under the shared base name.
                if base not in groups:
                    groups[base] = [col, t2_col]
            else:
                groups[col] = [col]
        elif col.startswith("T2_"):
            base = col[3:]
            # Only add as standalone if there is no matching T1_ column.
            if "T1_" + base not in all_cols:
                groups[col] = [col]

    return groups


def sample_feature_subset(
    rng: np.random.Generator,
    candidate_groups: dict[str, list[str]],
    baseline_features: list[str],
    trial_idx: int,
) -> list[str]:
    # Trial 0 is the exact baseline for an apples-to-apples comparison.
    if trial_idx == 0:
        return baseline_features.copy()

    baseline_set = set(baseline_features)
    chosen: list[str] = []

    for group_name, group_cols in candidate_groups.items():
        # A group is "in baseline" only if every column in it appears there.
        in_baseline = all(c in baseline_set for c in group_cols)
        # Bias search around baseline: keep baseline groups often, add new ones occasionally.
        if in_baseline:
            if rng.random() < 0.80:
                chosen.extend(group_cols)
        else:
            if rng.random() < 0.15:
                chosen.extend(group_cols)

    # Ensure a stable minimum signal even in sparse random draws.
    core_groups = ["seed", "elo", "quality", "men_women", "Seed_diff", "elo_diff"]
    for group_name in core_groups:
        if group_name in candidate_groups:
            for col in candidate_groups[group_name]:
                if col not in chosen:
                    chosen.append(col)

    return sorted(set(chosen))


def sample_params(
    rng: np.random.Generator,
    baseline_param: dict,
    baseline_rounds: int,
    use_gpu: bool,
    trial_idx: int,
) -> tuple[dict, int]:
    # Baseline trial uses the exact known-good notebook values.
    if trial_idx == 0:
        params = copy.deepcopy(baseline_param)
    else:
        # Sample near typical strong ranges instead of searching unbounded space.
        params = {
            "objective": "reg:squarederror",
            "booster": "gbtree",
            "eta": float(np.exp(rng.uniform(np.log(0.005), np.log(0.03)))),
            "subsample": float(rng.uniform(0.50, 0.90)),
            "colsample_bynode": float(rng.uniform(0.55, 1.0)),
            "num_parallel_tree": int(rng.integers(1, 5)),
            "min_child_weight": int(rng.integers(1, 9)),
            "max_depth": int(rng.integers(3, 8)),
            "tree_method": "hist",
            "grow_policy": rng.choice(["lossguide", "depthwise"], p=[0.8, 0.2]).item(),
            "max_bin": int(rng.integers(24, 65)),
            "reg_alpha": float(np.exp(rng.uniform(np.log(1e-4), np.log(1.0)))),
            "reg_lambda": float(np.exp(rng.uniform(np.log(0.3), np.log(3.0)))),
        }

    params["device"] = "cuda" if use_gpu else "cpu"

    if trial_idx == 0:
        num_rounds = baseline_rounds
    else:
        # Perturb boosting rounds around baseline depth of training.
        num_rounds = int(rng.integers(max(300, baseline_rounds - 300), baseline_rounds + 301))

    return params, num_rounds


def evaluate_config(
    tourney_data: pd.DataFrame,
    features: list[str],
    params: dict,
    num_rounds: int,
    require_gpu: bool,
) -> dict:
    # Leave-one-season-out cross validation to mimic notebook evaluation.
    seasons = sorted(set(tourney_data["Season"]))
    oof_preds = []
    oof_targets = []
    oof_seasons = []
    oof_mae = []
    models = {}

    local_params = copy.deepcopy(params)
    requested_device = local_params.get("device", "cpu")
    fallback_happened = False

    for oof_season in seasons:
        train_mask = tourney_data["Season"] != oof_season
        val_mask = tourney_data["Season"] == oof_season

        # Use float32 to cut host memory traffic and GPU transfer overhead.
        x_train = tourney_data.loc[train_mask, features].to_numpy(dtype=np.float32)
        y_train = tourney_data.loc[train_mask, "PointDiff"].values
        x_val = tourney_data.loc[val_mask, features].to_numpy(dtype=np.float32)
        y_val = tourney_data.loc[val_mask, "PointDiff"].values

        if local_params.get("device") == "cuda":
            # QuantileDMatrix is typically faster with hist on GPU.
            qdm_max_bin = int(local_params.get("max_bin", 256))
            dtrain = xgb.QuantileDMatrix(x_train, label=y_train, max_bin=qdm_max_bin)
            dval = xgb.QuantileDMatrix(x_val, label=y_val, max_bin=qdm_max_bin)
        else:
            dtrain = xgb.DMatrix(x_train, label=y_train)
            dval = xgb.DMatrix(x_val, label=y_val)

        try:
            model = xgb.train(
                params=local_params,
                dtrain=dtrain,
                num_boost_round=num_rounds,
            )
        except xgb.core.XGBoostError as ex:
            # Graceful runtime fallback only for actual CUDA availability/runtime errors.
            if local_params.get("device") == "cuda":
                ex_text = str(ex).lower()
                cuda_unavailable = (
                    "cuda" in ex_text
                    or "gpu" in ex_text
                    or "no visible gpu" in ex_text
                    or "invalid device ordinal" in ex_text
                    or "not compiled with cuda" in ex_text
                )
                if not cuda_unavailable:
                    raise RuntimeError(f"XGBoost training failed on GPU config: {ex}") from ex
                if require_gpu:
                    raise RuntimeError(
                        "CUDA requested but unavailable in this run. "
                        "Re-run without --require-gpu to allow CPU fallback. "
                        f"Original error: {ex}"
                    ) from ex
                print(f"CUDA unavailable for season {oof_season}; falling back to CPU for this trial. Details: {ex}")
                local_params["device"] = "cpu"
                fallback_happened = True
                dtrain = xgb.DMatrix(x_train, label=y_train)
                dval = xgb.DMatrix(x_val, label=y_val)
                model = xgb.train(
                    params=local_params,
                    dtrain=dtrain,
                    num_boost_round=num_rounds,
                )
            else:
                raise RuntimeError(f"XGBoost training failed: {ex}") from ex

        preds = model.predict(dval)
        models[oof_season] = model
        oof_mae.append(mean_absolute_error(y_val, preds))
        oof_preds.extend(preds.tolist())
        oof_targets.extend(y_val.tolist())
        oof_seasons.extend([oof_season] * len(y_val))

    # Convert predicted point margins into probabilities via spline calibration.
    t = 25
    labels = (np.array(oof_targets) > 0).astype(int)
    dat = sorted(zip(oof_preds, labels), key=lambda x: x[0])
    pred_sorted, label_sorted = zip(*dat)
    spline_model = UnivariateSpline(np.clip(pred_sorted, -t, t), label_sorted, k=5)
    spline_fit = np.clip(spline_model(np.clip(oof_preds, -t, t)), 0.01, 0.99)

    brier = brier_score_loss(labels, spline_fit)

    by_season = {}
    eval_df = pd.DataFrame({"Season": oof_seasons, "label": labels, "pred": spline_fit})
    for season in seasons:
        s = eval_df.loc[eval_df["Season"] == season]
        by_season[int(season)] = float(brier_score_loss(s["label"], s["pred"]))

    return {
        "brier": float(brier),
        "mae": float(np.mean(oof_mae)),
        "params_used": local_params,
        "num_rounds": int(num_rounds),
        "features": features,
        "brier_by_season": by_season,
        "models": models,
        "requested_device": requested_device,
        "actual_device": local_params.get("device", "cpu"),
        "gpu_fallback": fallback_happened,
    }


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    build_start = time.perf_counter()
    print("Building modeling frame from raw data...")
    tourney_data = build_modeling_frame(args.data_dir, args.season_cutoff)
    print(f"Feature engineering/load time: {time.perf_counter() - build_start:.2f}s")

    candidate_groups = get_candidate_groups(tourney_data)
    all_candidate_cols = [c for cols in candidate_groups.values() for c in cols]
    missing_baseline = [c for c in BASELINE_FEATURES if c not in all_candidate_cols]
    if missing_baseline:
        raise ValueError(f"Baseline features missing from engineered frame: {missing_baseline}")

    print(f"Rows in tourney frame: {len(tourney_data)}")
    print(f"Candidate feature groups: {len(candidate_groups)} groups covering {len(all_candidate_cols)} columns")

    results = []
    best = None
    search_start = time.perf_counter()

    # Truncate output file at run start so each run has a clean per-trial log.
    results_path = Path(args.results_csv)
    if results_path.exists():
        results_path.unlink()

    # Trial loop includes baseline (trial 0) + user-requested random trials.
    for trial in range(args.n_trials + 1):
        features = sample_feature_subset(rng, candidate_groups, BASELINE_FEATURES, trial)
        params, num_rounds = sample_params(rng, BASELINE_PARAM, BASELINE_NUM_ROUNDS, args.use_gpu, trial)

        metrics = evaluate_config(tourney_data, features, params, num_rounds, args.require_gpu)

        row = {
            "trial": trial,
            "is_baseline": trial == 0,
            "brier": metrics["brier"],
            "mae": metrics["mae"],
            "num_features": len(features),
            "num_rounds": num_rounds,
            "features_json": json.dumps(features),
            "params_json": json.dumps(metrics["params_used"], sort_keys=True),
            "brier_by_season_json": json.dumps(metrics["brier_by_season"], sort_keys=True),
            "requested_device": metrics["requested_device"],
            "actual_device": metrics["actual_device"],
            "gpu_fallback": metrics["gpu_fallback"],
        }
        results.append(row)
        # Persist one row immediately so interrupted runs still leave usable results.
        pd.DataFrame([row]).to_csv(results_path, mode="a", header=(trial == 0), index=False)

        # Track best by global OOF Brier, which is the optimization target.
        if best is None or metrics["brier"] < best["brier"]:
            best = {
                "trial": trial,
                "brier": metrics["brier"],
                "mae": metrics["mae"],
                "num_rounds": num_rounds,
                "params": metrics["params_used"],
                "features": features,
                "brier_by_season": metrics["brier_by_season"],
                "requested_device": metrics["requested_device"],
                "actual_device": metrics["actual_device"],
                "gpu_fallback": metrics["gpu_fallback"],
            }

        print(
            f"trial={trial:03d} baseline={trial==0} "
            f"brier={metrics['brier']:.6f} mae={metrics['mae']:.6f} "
            f"n_features={len(features)} rounds={num_rounds} "
            f"device={metrics['actual_device']} fallback={metrics['gpu_fallback']}"
        )

    # Persist full trial history for later analysis/filtering.
    result_df = pd.DataFrame(results).sort_values("brier").reset_index(drop=True)
    result_df.to_csv(args.results_csv, index=False)

    print("\nTop trials:")
    print(result_df.head(args.top_k)[["trial", "is_baseline", "brier", "mae", "num_features", "num_rounds"]].to_string(index=False))

    print("\nBest config to copy into notebook:")
    print(f"best_trial = {best['trial']}")
    print(f"best_brier = {best['brier']:.6f}")
    print(f"best_mae = {best['mae']:.6f}")
    print(f"best_num_rounds = {best['num_rounds']}")
    print(f"best_features = {best['features']}")
    print(f"best_params = {best['params']}")
    print(f"best_brier_by_season = {best['brier_by_season']}")
    print(f"\nSaved full trial log: {args.results_csv}")
    print(f"Total search time: {time.perf_counter() - search_start:.2f}s")


if __name__ == "__main__":
    main()
