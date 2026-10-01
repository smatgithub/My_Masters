# 16. Ridge regression

Notebook: **3.2**, 10 marks. Choosing the number \(\alpha\) is topic 18. This file is the model itself.

## What changes relative to least squares

Least squares minimises RSS only. Ridge minimises RSS plus a penalty on the squared coefficients:

\[
\mathrm{Loss}_{\text{Ridge}} = \sum_{i=1}^{n} (y_i - \widehat{y}_i)^2 + \alpha \sum_{j=1}^{p} \beta_j^2
\]

\(\alpha\) (alpha) is a single non-negative number you choose. It is not estimated by the normal equation. The intercept \(\beta_0\) is **not** inside the sum. sklearn’s `Ridge` leaves the intercept unpenalised when `fit_intercept=True`, which is the default. Penalising the intercept would drag every prediction toward zero even when the average log price is near 9.7, which is nonsense.

The closed form, with the intercept handled separately and the columns standardised, is:

\[
\hat\beta_{\text{Ridge}} = (X^\top X + \alpha I)^{-1} X^\top y
\]

\(I\) is the identity matrix. Adding \(\alpha\) to each diagonal entry is what keeps the inverse alive when columns are collinear. That is the algebraic reason Ridge is the remedy you name after a high VIF.

## How the penalty moves a coefficient

Look at one coefficient and ignore the rest. The penalty \(\alpha \beta_j^2\) charges more as \(\beta_j\) moves away from zero, in either direction, because the square kills the sign. The optimiser accepts a slightly worse RSS in exchange for a smaller \(\beta_j\).

| \(\alpha\) | What the fit does |
| --- | --- |
| 0 | The penalty vanishes. Ridge is ordinary least squares. |
| Small relative to the training row count. On this file, values like 1 or 10. | Almost no shrinkage. The fit still looks like least squares. |
| Large enough to matter. On this file, often thousands to tens of thousands, because sklearn does not divide the penalty by \(n\). | Coefficients shrink toward 0. None of them is forced to be exactly 0. |
| Very large, far above the training row count | Every penalised coefficient goes to 0. The model predicts the training mean of \(y\) (the intercept). |

Shrink toward zero is not the same as delete. A weak equipment flag keeps a small coefficient. It does not disappear. That is the difference you will **see** when you compare Ridge coefficients with Lasso coefficients in section 3.4.

## A one-number illustration, in the units sklearn actually uses

`StandardScaler` makes a column have variance 1. With the intercept removed, that column’s sum of squares is about \(n\), the number of training rows, not 1. sklearn’s `Ridge` then minimises

\[
\sum_{i=1}^{n} (y_i - \widehat{y}_i)^2 + \alpha \beta^2
\]

For a single scaled column and no intercept, the shrinkage is:

\[
\beta_{\text{Ridge}} = \beta_{\text{OLS}} \cdot \frac{n}{n + \alpha}
\]

On this assignment \(n\) is about 12,700 training rows. Alpha has to be large compared with \(n\) before the fraction moves.

| \(\alpha\) | Factor \(n/(n+\alpha)\) with \(n = 12{,}700\) | If the OLS coefficient is 2 |
| ---: | ---: | ---: |
| 1 | 0.9999 | 2.00 |
| 100 | 0.992 | 1.98 |
| 1,000 | 0.927 | 1.85 |
| 10,000 | 0.560 | 1.12 |
| 100,000 | 0.113 | 0.23 |

An alpha of 1 is not “mild regularisation” here. It is almost ordinary least squares. An alpha of 10,000 is the neighbourhood where a lone coefficient actually halves. Correlated groups shrink sooner than this one-column table, because they share one signal, but if your first Ridge plot is a flat line, the grid is probably stuck far below \(n\). Topic 18 starts the search wide for that reason.

The sign stays positive. The effect is damped, not removed and not flipped. With many correlated features the same damping is shared across the group: if `hp_kW` and `Displacement_cc` carry one signal, Ridge gives both a modest coefficient instead of a huge positive on one and a huge negative on the other. Nothing in this table is forced to exactly zero.

## What you must have done before `Ridge.fit`

The penalty treats every \(\beta_j^2\) in the same units. If `km` is in the tens of thousands and `age` is in years, \(\beta_{\text{km}}\) is tiny and \(\beta_{\text{age}}\) is large only because the units differ, and the penalty then crushes `age` and ignores `km`. Standardisation (topic 11) is part of Ridge, not a cosmetic step. Fit the scaler on the training data, then:

```python
from sklearn.linear_model import Ridge

ridge = Ridge(alpha=10.0, random_state=42)
ridge.fit(X_train_prepared, y_train)
```

`Ridge` in current scikit-learn does not use `random_state` for the closed-form solver. Passing it is harmless if the installed version accepts it; if it raises `TypeError`, remove that argument. The randomness in this section is the train/test split and, if you use it, cross-validation folds, not the Ridge fit itself.

## What usually goes wrong

- Comparing Ridge coefficients with ordinary-regression coefficients when only one of the two models saw scaled data.
- Setting \(\alpha = 0\) and calling it regularisation. That run is the baseline, which you already fit in section 3.1.
- Expecting some coefficients to print as exact zeros. If they do, you have fit Lasso, or you are looking at a rounded display. Print with enough decimals, or count `np.sum(np.abs(coef) < 1e-8)`.
