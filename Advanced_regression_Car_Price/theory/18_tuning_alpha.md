# 18. Choosing alpha

Notebook: **3.2.1–3.2.3** and **3.3.1–3.3.3**. The brief tells you to try a list of alphas, plot train and test scores against alpha, pick the best of that list, then repeat on a tighter range.

## What you are searching for

α controls the strength of the penalty. You want the value that predicts new cars best, not the value that reconstructs the training cars best.

On a plot with α on the horizontal axis:

- Train score is best (least negative, if the score is negative MAE) at the smallest α, because a weak penalty lets the model fit the training rows more closely.
- Test score usually improves as α grows from zero, then worsens once the penalty starts erasing real signal.
- The α at the top of the **test** curve (or the bottom of the test-error curve) is the brief’s “best alpha”.

If both curves are flat, the model is not sensitive to the penalty. Say that. A flat curve means regularisation is not hurting and not helping much. That is a finding.

## Negative MAE on the plot

The brief’s “best score (negative MAE)” is -MAE. Plot that if you want the marker to see the same quantity the cell comment names. The best alpha is the highest point on the test curve. If you prefer to plot MAE itself, the best alpha is the lowest point, and you should relabel the axis so you do not call an error a score.

Whatever you plot, convert the winning score back to a positive MAE in the sentence under the chart: “best negative MAE is −0.142, so MAE is 0.142 on the log-price scale.”

## The two-pass search the brief describes

Pass 1 uses a coarse grid so you find the neighbourhood. Space the grid on a log scale, because the interesting change is often from 0.001 to 0.01, not from 10 to 11.

Ridge’s penalty in scikit-learn is added to the raw sum of squares, and it is not divided by n. After `StandardScaler`, a single coefficient shrinks by the factor n / (n + α). With about 12,700 training rows, α = 10 barely moves anything. The coarse grid has to reach tens of thousands.

```python
ridge_alphas = [0.1, 1, 10, 100, 1_000, 3_000, 10_000, 30_000, 100_000]
```

Lasso is on a different scale. scikit-learn divides the squared error by 2n before it adds α Σ |βⱼ|, so a useful Lasso alpha is often much smaller than a useful Ridge alpha. A starting grid:

```python
lasso_alphas = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1, 1]
```

These lists are search ranges, not answers. If the best point is at the end of the grid (the largest or the smallest alpha you tried), the true optimum is probably outside the grid. Extend the grid. Do not fine-tune around an endpoint.

Pass 2 builds a tighter grid around the winner. If the coarse Ridge winner is 10,000, and the neighbours are 3,000 and 30,000, search between those neighbours. `np.logspace` keeps the steps even on a log plot:

```python
ridge_alphas_fine = np.logspace(np.log10(3_000), np.log10(30_000), 20)
```

For Lasso, if the coarse winner is 0.001, search from 0.0003 to 0.003. Prefer `np.logspace` when the interval spans powers of ten, so the small values are not all crushed into the left of the plot.

```python
lasso_alphas_fine = np.logspace(np.log10(3e-4), np.log10(3e-3), 20)
```

Then fit one model per alpha, record train score and test score, plot both, and read off the best test score. Refit once on the training data with that alpha and store the coefficients. That stored model is the one section 3.4 compares.

A loop is enough and is easy to explain:

```python
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error

train_scores = []
test_scores = []
for a in ridge_alphas_fine:
    model = Ridge(alpha=a)
    model.fit(X_train_prepared, y_train)
    train_scores.append(-mean_absolute_error(y_train, model.predict(X_train_prepared)))
    test_scores.append(-mean_absolute_error(y_test, model.predict(X_test_prepared)))
```

The same loop with `Lasso(alpha=a, max_iter=10000)` is the Lasso search. Plot alpha on a log axis (`plt.xscale("log")`) or the coarse grid will look like two points.

## The honest limitation of this procedure

The brief tells you to look at the test score while you choose α. That is a teaching design, and you should follow it so the notebook matches the questions. It does use the test set as a tuning set. The MAE you then report for that same test set is a little optimistic, because you kept the α that flattered it.

In a production pricing model you would choose α by cross-validation **inside the training set** only:

CV score(α) = average validation MAE across folds of the training data

```python
from sklearn.linear_model import RidgeCV

ridge_cv = RidgeCV(alphas=ridge_alphas_fine, scoring="neg_mean_absolute_error")
ridge_cv.fit(X_train_prepared, y_train)
ridge_cv.alpha_
```

`RidgeCV` picks α without the test rows. You would then compute test MAE **once**, with that frozen α.

For this assignment, do the plot the brief asks for. In the conclusion, one sentence is enough: alpha was chosen by the test curve, as the notebook specifies, so the reported test MAE is slightly optimistic compared with a locked test set. If you want to be stricter, choose alpha with `RidgeCV` / `LassoCV` on the training data, still show the train-versus-alpha curve, and quote a single test evaluation. Either path is defensible if the write-up matches the code. Do not silently do one and describe the other.

`LassoCV` uses its own path algorithm and a default number of alphas. If you use it, set `alphas=lasso_alphas_fine` and `max_iter` high so it searches the range you intended.

## What usually goes wrong

- Fine-tuning with a grid that does not contain the coarse winner.
- Scoring Ridge with R² and Lasso with MAE, then comparing the two “best alphas” as if the scores were the same objective.
- Taking the alpha with the best **training** score. That alpha collapses toward 0, and you have refit ordinary least squares under another name.
