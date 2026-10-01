# Topic list — Predicting vehicle prices with regularisation

This folder is the study guide for `Regularisation_Assg_Car_Price_Prediction_Starter.ipynb`.

The starter notebook is the official submission file. These notes do not replace it, and they must not be pasted into it as a finished answer. Read a topic, then do the matching notebook cells yourself.

The dataset used in every example below is `Car_Price_data.csv`: **15,915 cars**, **23 columns**, no blank cells in the file as stored. Listed price runs from **4,950** to **74,600**, with a median of **16,900** and a mean of **18,024**. The mean sits above the median because a few expensive cars pull the average up. That single fact is why the target-transform topic matters.

## How to read these notes

Each file has the same shape:

1. What the idea is.
2. Why this assignment asks for it, including the marks.
3. The formula, then a small example you can recompute by hand.
4. What that formula means on this car data.
5. What to produce in the notebook.
6. What usually goes wrong.

Read them in the order below. Later topics assume the earlier ones.

## Topics

| Order | File | Notebook section | Marks |
| --- | --- | --- | --- |
| 1 | [01_the_prediction_problem.md](01_the_prediction_problem.md) | Objectives | — |
| 2 | [02_reading_this_dataset.md](02_reading_this_dataset.md) | 1 Data understanding, 1.1 Load | — |
| 3 | [03_missing_values.md](03_missing_values.md) | 2.1.1 | 1 |
| 4 | [04_distributions_and_frequency.md](04_distributions_and_frequency.md) | 2.1.2 and 2.1.3 | 6 |
| 5 | [05_rare_categories_and_imbalance.md](05_rare_categories_and_imbalance.md) | 2.1.4, including `Type` | 3 |
| 6 | [06_multi_value_feature_lists.md](06_multi_value_feature_lists.md) | 2.1.3 note and 2.4.2 | part of 11 |
| 7 | [07_skewed_target_and_log_transform.md](07_skewed_target_and_log_transform.md) | 2.1.5 | 3 |
| 8 | [08_correlation_numbers_and_categories.md](08_correlation_numbers_and_categories.md) | 2.2.1 and 2.2.2 | 6 |
| 9 | [09_outliers.md](09_outliers.md) | 2.3.1 and 2.3.2 | 5 |
| 10 | [10_feature_engineering_and_encoding.md](10_feature_engineering_and_encoding.md) | 2.4.1 and 2.4.3 | 4 plus engineering |
| 11 | [11_train_test_split_leakage_and_scaling.md](11_train_test_split_leakage_and_scaling.md) | 2.4.4 and 2.4.5 | 4 |
| 12 | [12_linear_regression_ols.md](12_linear_regression_ols.md) | 3.1.1 fit | part of 10 |
| 13 | [13_evaluation_metrics.md](13_evaluation_metrics.md) | 3.1.1 metrics, later “negative MAE” | part of 10 and 5 |
| 14 | [14_residuals_assumptions_and_vif.md](14_residuals_assumptions_and_vif.md) | 3.1.2 | 5 |
| 15 | [15_overfitting_bias_variance.md](15_overfitting_bias_variance.md) | Why section 3 exists | — |
| 16 | [16_ridge_regression.md](16_ridge_regression.md) | 3.2 | 10 |
| 17 | [17_lasso_regression.md](17_lasso_regression.md) | 3.3 | 10 |
| 18 | [18_tuning_alpha.md](18_tuning_alpha.md) | 3.2 and 3.3 alpha search | inside 10 + 10 |
| 19 | [19_comparing_models_and_writing_the_conclusion.md](19_comparing_models_and_writing_the_conclusion.md) | 3.4 and 4 | 5 + 10 |

Section 2 is worth 35 marks, section 3 is worth 35, and the conclusion is worth 10. The notebook’s own heading numbers the last modelling step as “3” again; that is a typo in the brief. The real last modelling block is regularisation inside section 3, then the conclusion in section 4.

## What “done” looks like

You are ready to write the conclusion when you can answer these in your own words, with a number from your notebook next to each one:

- Which column is the target, and which transform you applied, and why.
- Which predictors are numeric, which are categorical, and what you did with the four comma-separated feature lists.
- Whether a linear model’s errors look random, and whether any feature’s VIF forced you to drop or combine columns.
- The test MAE (or RMSE, or R²) for ordinary linear regression, Ridge, and Lasso, on the same scale.
- Whether Lasso set any coefficients to zero, and whether Ridge only shrank them.
- Whether train error and test error were far apart. That is the overfitting check.
- Whether regularisation improved prediction, or only made the coefficients more stable. Both outcomes are acceptable if you explain them.

## Python environment for this workspace

The Mac default `python3` is **3.14**. Scientific packages are not a safe bet on that version yet. This workspace uses a virtual environment built on **Python 3.12**, which is also installed via Homebrew.

From the workspace root `My Masters`:

```bash
source .venv/bin/activate
python -c "import sklearn, pandas; print(sklearn.__version__)"
```

In Jupyter or Cursor, choose the kernel **Python (Masters ML)**. That kernel points at `.venv`. The package list is `requirements.txt` in the workspace root.

When you add the next assignment folder, create a `theory/` directory inside that folder the same way. Reuse this one environment unless a later course pins different package versions.
