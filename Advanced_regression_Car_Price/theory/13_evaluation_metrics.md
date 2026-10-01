# 13. Evaluation metrics, including negative MAE

Notebook: **3.1.1** asks for suitable metrics. Later Ridge and Lasso cells ask for the **best score (negative MAE)**. Section **3.4.1** (2 marks) compares the three models. Use the same metrics for all three.

## Three numbers, from the same three errors

Take three cars. The model predicts log price, you exponentiate back to currency, and you see:

| Car | Actual price | Predicted price | Error \(y - \widehat{y}\) | Absolute error |
| --- | ---: | ---: | ---: | ---: |
| A | 15,000 | 16,000 | −1,000 | 1,000 |
| B | 18,000 | 17,000 | +1,000 | 1,000 |
| C | 22,000 | 20,000 | +2,000 | 2,000 |

### Mean absolute error

\[
\mathrm{MAE} = \frac{1}{n} \sum_{i=1}^{n} \lvert y_i - \widehat{y}_i \rvert
= \frac{1{,}000 + 1{,}000 + 2{,}000}{3} = 1{,}333.\overline{3}
\]

The average car is missed by about 1,333 currency units. Every error counts in proportion to its size. MAE is the metric the brief points at when it says negative MAE.

### Root mean squared error

\[
\mathrm{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \widehat{y}_i)^2}
= \sqrt{\frac{1{,}000{,}000 + 1{,}000{,}000 + 4{,}000{,}000}{3}}
= \sqrt{2{,}000{,}000} \approx 1{,}414
\]

RMSE is larger than MAE because car C’s error is squared. One badly priced car moves RMSE more than it moves MAE. If your RMSE is far above your MAE, a few cars are much worse than the typical miss.

### \(R^2\)

\(R^2\) compares the model with a model that ignores the features and always predicts the training mean of \(y\).

\[
R^2 = 1 - \frac{\sum (y_i - \widehat{y}_i)^2}{\sum (y_i - \bar{y})^2}
= 1 - \frac{\mathrm{SS}_{\text{res}}}{\mathrm{SS}_{\text{tot}}}
\]

On the three prices, \(\bar{y} = 18{,}333.\overline{3}\).

\[
\mathrm{SS}_{\text{tot}} = (15{,}000 - 18{,}333.\overline{3})^2 + (18{,}000 - 18{,}333.\overline{3})^2 + (22{,}000 - 18{,}333.\overline{3})^2
\]

\[
= (-3{,}333.\overline{3})^2 + (-333.\overline{3})^2 + (3{,}666.\overline{7})^2
= 11{,}111{,}111 + 111{,}111 + 13{,}444{,}444
= 24{,}666{,}666
\]

\[
\mathrm{SS}_{\text{res}} = 6{,}000{,}000
\]

\[
R^2 = 1 - \frac{6{,}000{,}000}{24{,}666{,}666} \approx 0.757
\]

About 76% of the price variation around the mean is captured by the predictions, in this toy. The remaining 24% is still error. \(R^2 = 1\) is a perfect fit. \(R^2 = 0\) is no better than predicting the mean. A negative \(R^2\) on the **test** set means the model is worse than predicting the training mean; that is a failed model, not a rounding issue.

sklearn’s `r2_score` on the test set uses the test labels in \(\mathrm{SS}_{\text{tot}}\). Quote it as test \(R^2\) and do not compare it with a training \(R^2\) as if the denominators were the same kind of number. Still report both: a training \(R^2\) of 0.90 and a test \(R^2\) of 0.60 is the overfitting signature from topic 15.

## Why the notebook says “negative MAE”

scikit-learn’s model-selection tools **maximise** a score. MAE is an error, so smaller is better. The library therefore uses the negative:

\[
\text{score} = -\mathrm{MAE}
\]

A “best score” of −0.18 means the MAE is 0.18. The least negative score is the winner: −0.18 is better than −0.25 because \(0.18 < 0.25\).

If `y` is \(\ln(\text{price})\), that 0.18 is a log-error, not currency. Say so in the markdown cell, or also print MAE after `np.exp`.

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

pred_log = linreg.predict(X_test_prepared)
mae_log = mean_absolute_error(y_test, pred_log)
rmse_log = mean_squared_error(y_test, pred_log) ** 0.5
r2 = r2_score(y_test, pred_log)

pred_price = np.exp(pred_log)
mae_price = mean_absolute_error(np.exp(y_test), pred_price)
```

`mean_squared_error(..., squared=False)` was removed in recent scikit-learn. Taking the square root, as above, works on the version installed in this workspace.

## A comparison table worth  the marks

One row per model, columns at least: test MAE on log price, test RMSE on log price, test \(R^2\), and test MAE in currency. Training metrics sit beside them so you can see the gap. Topic 19 is how to interpret that table. Do not fill it with numbers you have not computed.

## What usually goes wrong

- Comparing Ridge’s negative-MAE score with linear regression’s positive MAE and concluding Ridge is worse because of the minus sign.
- Computing currency MAE for one model and log MAE for another.
- Using accuracy, precision, or a confusion matrix. Those are classification metrics. Price is not a class.
