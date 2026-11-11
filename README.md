# High-Engagement User Study

This repository accompanies Chapter 5, the account study, in Cara Lynch's unpublished PhD thesis, *The Predictability of Engagement Allocation in Online Communities*.

The study compares simple ranking baselines with logistic regression and LightGBM. It uses prior-month activity across 20 Reddit communities to predict which users will rank in the top 10% for received engagement per contribution the next month.

## Try it without Reddit data

With Conda installed, open a terminal in the folder containing this README and run:

```text
conda env create -f environment.yml
conda activate high-engagement-study
python -B examples/smoke_models.py
```

The example makes a table of 330 rows, one for each of 30 invented accounts in each of 11 months. It fits both learned models and checks their outputs across six rounds of training and testing. Each run keeps the invented input table, metrics, validation reports and LightGBM model files in a new folder under `runs/synthetic/`. The script prints the folder path. The example was tested on Windows with Python 3.11.14 and the package versions in `environment.yml`.

## Run with compatible inputs

Use the commands from the repository root. If you already have a prepared model panel and its feature schema, start at step 3.

**1. Prepare the monthly files.** Stage `01` converts compressed JSON Lines to Parquet. Put files named like `RC_2021-01.zst` under `inputs/raw/comments/` and `RS_2021-01.zst` under `inputs/raw/submissions/`. The commands then clean the files, replace account identifiers using a private salt file, and filter to the [reviewed communities](outputs/selected_subreddits_final.csv). Substitute your own input and salt paths.

```text
python -m Preprocessing.01_convert_to_parquet --input-root inputs/raw --out-dir runs/converted --year 2021 --jobs 1
python -m Preprocessing.03_0_cleaning --parquet-root runs/converted --out-root runs/clean --year 2021 --jobs 1 --duckdb-threads 1
python -m Preprocessing.04_0_pseudonymise_authors --in-root runs/clean --out-root runs/pseudonymised --year 2021 --salt-file inputs/private-salt.bin
python -m Preprocessing.07_2_make_selected_subreddit_parquets --in-root runs/pseudonymised --out-root runs/selected --year 2021 --selection-mode explicit_csv --explicit-subreddit-csv outputs/selected_subreddits_final.csv --subset-summary-csv outputs/selected_subreddits_final.csv --duckdb-threads 1
```

The gaps in the numbering cover checks, text processing, descriptive analysis or community review. The commands above use the supplied community list and the core files needed for modelling. This release fixes the converter's handling of the final partial batch of comments.

**2. Build and check the panel.** A panel has one row per account and prediction month. The builder writes it and its feature schema to `runs/panel/`.

```text
python -m Modelling.0_0_make_modelling_panel --in-root runs/selected --out-root runs/panel --selected-subreddits-csv outputs/selected_subreddits_final.csv --author-col author_pid --duckdb-threads 1
python -m Modelling.0_1_validate_modelling_panel --panel-path runs/panel/modelling_panel.parquet --feature-schema-csv runs/panel/feature_schema.csv --out-root runs/panel-check --duckdb-threads 1
```

**3. Run the baselines and fit the models.** Replace the panel paths if you are starting from a prepared panel. The baselines rank accounts by prior activity, prior engagement or chance. The wrapper trains and validates the two learned models.

```text
python -m Modelling.1_0_run_baselines --panel-path runs/panel/modelling_panel.parquet --feature-schema-csv runs/panel/feature_schema.csv --out-root runs/baselines --random-seed 20260529 --duckdb-threads 1
python -B examples/run_models.py --panel-path runs/panel/modelling_panel.parquet --feature-schema-csv runs/panel/feature_schema.csv --out-root runs/account-study
```

The learned models select candidates by average precision on the validation month, refit on training plus validation, then score the held-out month. All methods are evaluated on July to December 2021. Compare `average_precision_macro_mean` in the `*_metrics_summary.csv` files under `runs/baselines/`, `runs/account-study/logistic/` and `runs/account-study/lightgbm/`. The [split calendar](config/rolling_origin_splits.csv), [nine predictors](config/selected_features.csv) and [selected settings](config/study2_appendix_fold_hyperparameters.csv) document the original model runs.

## Data source and citation

The study used the 2021 comments and submissions in [*Reddit comments/submissions 2005-06 to 2022-12*](https://web.archive.org/web/20241213161954/https://academictorrents.com/details/7c0645c94321311bb05bd879ddee4d0eba08aaee), a Pushshift-derived archive credited to stuck_in_the_matrix and Watchful1 (2023) and distributed through Academic Torrents. The original Reddit data are excluded. The [MIT licence](LICENSE) covers the software and documentation. See [CITATION.cff](CITATION.cff) for the software citation.
