# 11. Train/test split, leakage, and scaling

Notebook: **2.4.4** (2 marks) and **2.4.5** (2 marks).

## Why you hide part of the data

If you fit the line and then score it on the same cars, the score answers “how well did we memorise this file?” The reseller’s question is “how well will we price a car that was not in the file?”

Hold out a test set, typically 20%:

n(test) = 0.2 × 15,915 ≈ 3,183, and n(train) ≈ 12,732

Fit every transformer and every model on the training rows only. Touch the test rows twice: once to transform them with the already-fitted scaler and encoder, and once at the end to compute the metric. If you use the test metric to choose the Ridge penalty, the test set has secretly become part of training. Topic 18 explains how the notebook asks you to plot test scores anyway, and what that costs.

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

`random_state=42` makes the split repeatable so Ridge and Lasso are compared on the same cars. `y` here is ln(price), not raw price, if you followed topic 7.

## What leakage looks like

Leakage is any path by which information from the test cars, or from the target, enters the features before prediction.

| Leak | Why the score becomes a lie |
| --- | --- |
| `price` left inside `X` | The model reads the answer. |
| Mean, percentile fence, or one-hot categories computed on all rows before the split | Test cars change the fill value, the cap, or the column set. |
| Scaler `fit` on `X_train` and `X_test` stacked together | The test mean shifts the training z-scores. |
| Choosing α because it had the best **test** MAE, then reporting that same MAE as the unbiased performance | You picked the winner after seeing the exam. |

On this file the first row of that table is the one that silently ruins the assignment. Drop `price` and `log_price` from `X` and check `X.columns` before the split.

## Standardisation, the formula you will actually use

Ridge and Lasso penalise the size of coefficients. A coefficient on `km` (tens of thousands) and a coefficient on `age` (a handful of years) are not comparable until the columns are in the same units. Standardisation gives each numeric column mean 0 and standard deviation 1, **using the training column**:

zᵢ = (xᵢ − μ(train)) / s(train)

## A hand example

Training ages: 1, 3, 5. Then μ = 3 and the population-style spread used in a hand calculation can be shown with the sample standard deviation. Deviations: −2, 0, +2. Sample variance with denominator n-1 = 2:

s² = ((-2)² + 0² + 2²)/2 = 4, s = 2

The age-5 car becomes (5 - 3) / 2 = 1. It is one training standard deviation above the training mean.

A test car with age 7 must use the **same** μ and s, not its own:

z = (7 - 3)/2 = 2

If you refit μ and s on the test car alone, a single test car would have an undefined or meaningless scale, and the coefficient learned on training z-scores would be applied to a different unit.

sklearn’s `StandardScaler` uses the training mean and the training standard deviation (divisor n, not n-1). The difference versus the hand example is small at n = 12,732. Do not mix a hand-rolled n-1 scaler with a story about sklearn.

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train[numeric_cols])
X_test_scaled = scaler.transform(X_test[numeric_cols])
```

`fit_transform` is for training. `transform` alone is for test. Calling `fit_transform` on the test block is the leak.

## What to scale

Scale the continuous measurements: `km`, `age`, `hp_kW`, `Displacement_cc`, `Weight_kg`, `cons_comb`, and any ratio you created.

One-hot columns are already on a 0/1 scale. Scaling them is optional. If you scale them, “one standard deviation of the Diesel dummy” is an awkward unit in the conclusion. A clear setup is: scale numeric columns, leave 0/1 columns as 0/1. Then a dummy coefficient is still a log-price shift for “this level versus the reference”, which is the reading in topic 7.

Do not scale `y` unless you have a reason. The log transform already changed its shape. Scaling `y` as well makes the intercept harder to explain and does not help Ridge, because the penalty is on the coefficients of `X`.

## What usually goes wrong

- Splitting after scaling.
- Fitting the scaler on a dataframe that still contains strings.
- Forgetting that `X_train_scaled` is a NumPy array and no longer has column names. Keep a list of names, because section 3.4 asks you to compare coefficients by feature.
