# 12. Linear regression by least squares

Notebook: **3.1.1**, the fit half of the 10 marks. Metrics are topic 13. Residual checks are topic 14.

## The model

For car \(i\), with \(p\) predictors:

\[
y_i = \beta_0 + \beta_1 x_{i1} + \beta_2 x_{i2} + \cdots + \beta_p x_{ip} + \varepsilon_i
\]

\(y_i\) is \(\ln(\text{price}_i)\) if you transformed the target. \(x_{ij}\) is feature \(j\) after encoding and scaling. \(\varepsilon_i\) is the part the features do not explain. The prediction drops the error term:

\[
\widehat{y}_i = \beta_0 + \beta_1 x_{i1} + \cdots + \beta_p x_{ip}
\]

In matrix form, with a column of ones in \(X\) for the intercept:

\[
\widehat{\mathbf{y}} = X\beta
\]

## What “best” means

Ordinary least squares picks \(\beta\) to minimise the residual sum of squares:

\[
\mathrm{RSS} = \sum_{i=1}^{n} (y_i - \widehat{y}_i)^2
\]

Squaring does two things. It makes a miss of −3 and a miss of +3 equally bad, and it makes a miss of 6 four times as costly as a miss of 3, not twice. That is why one 74,600 car, on the raw price scale, can steer the whole line. You already reduced that effect by modelling \(\ln(\text{price})\).

## A two-point line, so the formula is visible

Two cars, one feature, no need for a matrix. Age 1 has log-price 10.0. Age 5 has log-price 9.2. The unique straight line through both points has slope

\[
\beta_1 = \frac{9.2 - 10.0}{5 - 1} = \frac{-0.8}{4} = -0.2
\]

and intercept

\[
\beta_0 = 10.0 - (-0.2)(1) = 10.2
\]

\[
\widehat{y} = 10.2 - 0.2 \cdot \text{age}
\]

At age 1: \(10.2 - 0.2 = 10.0\). At age 5: \(10.2 - 1.0 = 9.2\). RSS is 0 because two points always determine a line. The moment you add a third car that is not on the line, RSS is positive and the slope is a compromise. That compromise, for many features, is the normal equation

\[
\hat\beta = (X^\top X)^{-1} X^\top y
\]

You do not invert this by hand. scikit-learn does. The formula tells you when the fit will fail: if \(X^\top X\) has no inverse. That happens when one column is an exact copy or exact sum of others (all \(K\) dummies plus an intercept). Topic 10’s `drop="first"` is what keeps this inverse alive.

With thousands of rows and standardised columns, sklearn’s `LinearRegression` uses a numerically stable solver rather than a raw inverse. The \(\beta\) it returns is still the RSS minimiser.

## What each coefficient means after your preprocessing

Read a coefficient only in the units you actually fed the model.

| If the column is… | Then \(\beta_j = 0.1\) means… |
| --- | --- |
| A standardised numeric feature, and \(y = \ln(\text{price})\) | One training standard deviation higher on that feature is associated with \(e^{0.1} - 1 \approx 10.5\%\) higher price, other columns held fixed. |
| A 0/1 dummy that was **not** standardised, same \(y\) | Being in that level, versus the dropped reference level, is associated with about 10.5% higher price. |
| A raw column in currency, and \(y\) is raw price | One extra unit of the column adds 0.1 currency units. You should not be in this case if you logged and scaled. |

“Held fixed” is the phrase that keeps you honest. The age coefficient is not “the effect of age” in the real market. It is the association of age with log price among cars that have the same kilometres, power, model, and options in this file. Older cars also have more kilometres. Because both columns are in the model, each coefficient is the partial association. That is also why correlated columns make coefficients jumpy: the data cannot see age move while kilometres stay still. Topic 14 measures that.

## Fit it

```python
from sklearn.linear_model import LinearRegression

linreg = LinearRegression()
linreg.fit(X_train_prepared, y_train)
print(linreg.intercept_, linreg.coef_)
```

Store `linreg.coef_` with the feature names. Section 3.4 compares these numbers with the Ridge and Lasso coefficients. If the names are lost, that comparison becomes a list of anonymous floats.

## What usually goes wrong

- Fitting on raw, unscaled, unencoded strings and reading a Python exception as a modelling result. Encoding and scaling come first.
- Reporting the training RSS as the quality of the model. RSS always falls when you add a column, even a useless one. The test metric in topic 13 is the number that can rise when you add junk.
