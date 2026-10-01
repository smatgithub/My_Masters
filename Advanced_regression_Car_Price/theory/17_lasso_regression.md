# 17. Lasso regression

Notebook: **3.3**, 10 marks. The notebook text in 3.3.2 says “Apply Ridge Regularisation” above a Lasso cell. That is a copy-paste slip in the brief. The heading of 3.3 is Lasso, and the code comments say Lasso. Fit Lasso.

## The penalty that can delete a feature

Lasso minimises RSS plus a penalty on the **absolute** coefficients:

\[
\mathrm{Loss}_{\text{Lasso}} = \sum_{i=1}^{n} (y_i - \widehat{y}_i)^2 + \alpha \sum_{j=1}^{p} \lvert \beta_j \rvert
\]

The intercept is again not penalised.

The absolute value is why zeros appear. A square is flat at zero: the slope of \(\beta^2\) at \(\beta = 0\) is 0, so a tiny coefficient is almost free under Ridge, and the fit keeps it. The absolute value has a sharp corner at zero. The penalty’s pull toward zero stays at full strength \(\alpha\) even for a tiny coefficient. If the feature does not reduce RSS by enough to pay that pull, the optimum sits exactly at the corner: \(\beta_j = 0\).

There is no \((X^\top X + \alpha I)^{-1}\) formula for Lasso. The absolute value is not differentiable at zero, so the solution is computed by coordinate descent. sklearn’s `Lasso` does that. You do not implement the loop.

## A picture in one feature

Take the same teaching case as Ridge: one standardised feature, OLS coefficient 2. Lasso’s one-feature soft-threshold rule is:

\[
\beta_{\text{Lasso}} =
\begin{cases}
2 - \alpha & \text{if } 2 > \alpha \\
0 & \text{if } \lvert 2 \rvert \le \alpha \\
-2 + \alpha & \text{if the OLS coefficient is negative and large}
\end{cases}
\]

For a positive OLS coefficient of 2:

| \(\alpha\) | Lasso coefficient | Ridge coefficient from topic 16 |
| ---: | ---: | ---: |
| 0 | 2 | 2 |
| 0.5 | 1.5 | \(2 / 1.5 \approx 1.33\) |
| 1 | 1 | 1 |
| 2 | 0 | \(2 / 3 \approx 0.67\) |
| 3 | 0 | 0.5 |

At \(\alpha = 2\) Lasso has removed the feature. Ridge still keeps more than half of it. That single table is the comparison section 3.4 wants, once you replace the toy numbers with your real coefficients.

The one-feature rule is not how a multi-feature Lasso shares credit among correlated equipment flags. In the real fit, Lasso tends to keep one flag from a correlated group and set the neighbours to zero. Which one it keeps can change if the training sample changes. Do not describe a zeroed flag as “proven irrelevant”. Describe it as “not needed once the features Lasso kept are in the model”.

## What to fit, and what to show

```python
from sklearn.linear_model import Lasso

lasso = Lasso(alpha=0.001, max_iter=10000)
lasso.fit(X_train_prepared, y_train)

zeroed = np.sum(np.abs(lasso.coef_) < 1e-8)
```

`max_iter=10000` matters. Lasso’s solver sometimes hits the default iteration limit when there are many equipment flags. If you see a `ConvergenceWarning`, increase `max_iter` before you trust the zeros. A coefficient that is zero only because the solver stopped early is not a selected feature.

Show three things the brief asks for:

- The chosen \(\alpha\), and the negative MAE at that \(\alpha\).
- The coefficient next to each feature name.
- A count of exact zeros, and the names of those features. Section 3.4.2 asks you to visualise coefficients Lasso dropped. A bar chart of the largest surviving coefficients, plus a printed list of the zeroed names, answers that cell. Do not try to bar-chart 200 zeros; the list is the chart’s companion.

Scale the predictors the same way you did for Ridge. Do not reuse the Ridge alpha. scikit-learn minimises \(\mathrm{RSS} + \alpha \sum \beta_j^2\) for Ridge, but for Lasso it minimises \(\mathrm{RSS}/(2n) + \alpha \sum \lvert\beta_j\rvert\). The \(1/(2n)\) factor, with \(n\) around 12,700, means a Lasso alpha of 1 is a heavy penalty, while a Ridge alpha of 1 barely shrinks anything. Tune them on separate grids. Topic 18 gives starting ranges. A Lasso alpha that still leaves some coefficients alive, on standardised log-price data, is often somewhere between \(10^{-5}\) and \(10^{-1}\). That is a search hint, not a result.

## What usually goes wrong

- Using `Lasso(alpha=1)` on standardised log-price data, watching every coefficient collapse to zero, and concluding that no feature matters. \(\alpha = 1\) is often far too large when \(y\) is a log and \(X\) is standardised. Look at the path of scores in topic 18 before you interpret zeros.
- Calling a coefficient of `1e-16` a real effect. Treat values below `1e-8` as zero.
- Fitting Lasso on unscaled `km` and then announcing that kilometres were “selected out”. The penalty deleted the small coefficient that the kilometre unit created, not the business effect.
