# march-madness-2026

1. download the [data](https://www.kaggle.com/competitions/march-machine-learning-mania-2026/data), put in `/data` directory
2. Create virtual environment (use poetry)
- I recommend running `poetry config virtualenvs.in-project true` to keep the venv local
- Then, run `poetry install --no-root`
3. Activate virtual environment
- Mac: `source .venv/bin/activate`
- PC: `.venv/scripts/activate`
4. Add the jupyter notebook kernel to your virtual environment

(mac-specific) if you get a weird python 32-bit version error, in a terminal OUTSIDE THE VIRTUAL ENVIRONMENT, run the following (and restart your vscode terminal):
```
brew install libomp
```

Now that the development environment is set up, you can get going with the data ingestion.

**kaggle_scraper_noleak.py**

Web-scrapes the kenpom website for additional features, saves to a csv within the data directory. Specifically retrieves pre-tournament kenpom stats to avoid training leakage. 

**tune_xgb_features.py**

Runs a random search over XGBoost hyperparameters and feature subsets, optimizing out-of-fold Brier score. Trial 0 is always the current baseline so you have a direct comparison. Results are saved to `tuning_results.csv`.

Basic usage (CPU):
```
python tune_xgb_features.py --n-trials 40
```

With GPU acceleration:
```
python tune_xgb_features.py --n-trials 40 --use-gpu
```

Add `--require-gpu` to fail fast if CUDA is unavailable rather than silently falling back to CPU:
```
python tune_xgb_features.py --n-trials 40 --use-gpu --require-gpu
```

Fix the random seed to reproduce a specific search run:
```
python tune_xgb_features.py --n-trials 40 --seed 2026
```

Other options:
| Flag | Default | Description |
|---|---|---|
| `--data-dir` | `data` | Path to the Kaggle data folder |
| `--season-cutoff` | `2003` | Earliest season included in training |
| `--n-trials` | `40` | Number of random trials to run after the baseline |
| `--results-csv` | `tuning_results.csv` | Where to save the full trial log |
| `--top-k` | `10` | How many top trials to print at the end |

At the end of the run, the script prints the best feature list, hyperparameters, and `num_rounds` ready to paste directly into cells 38 and 39 of `2026_notebook.ipynb`.

**2026_notebook.ipynb**

- 2026_notebook contains the current strategy, heavily based on the 2023 winner's gradient-boosted random forest
- Make sure to skim the notebook and update years where necessary (find the 2025s and change to the updated year, etc).
- If data formatting gets changed by Kaggle, this will break.
- Just running this file is sufficient for creating a kaggle-style submission for the march machine learning mania competition (as of 2025)

***The following is for post-training fun***

**simulate_n_brackets.ipynb**

- simulate-n-brackets contains the logic to create simulated brackets by using the probabilities generated in submission.csv as weighted coin flips
- It also generates and saves a (currently slightly odd-looking) visual representation of one bracket

**simulated-bracket-playground**

- Check it out! Contains some fun stuff that can be done with the simulated brackets stored in bracket_simulations.csv


Code to simulate n brackets using previous years' submission format: https://www.kaggle.com/code/lennarthaupts/simulate-n-brackets/notebook