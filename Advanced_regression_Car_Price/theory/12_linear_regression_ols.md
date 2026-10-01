# 12. Linear regression by least squares

Notebook: **3.1.1**, the fit half of the 10 marks. Metrics are topic 13. Residual checks are topic 14.

## The model

For car i, with p predictors:

yᵢ = β₀ + β₁ xᵢ₁ + β₂ xᵢ₂ + … + βₚ xᵢₚ + εᵢ

yᵢ is ln(priceᵢ) if you transformed the target. xᵢⱼ is feature j after encoding and scaling. εᵢ is the part the features do not explain. The prediction drops the error term:

ŷᵢ = β₀ + β₁ xᵢ₁ + … + βₚ xᵢₚ

In matrix form, with a column of ones in X for the intercept:

ŷ = Xβ

## What “best” means

Ordinary least squares picks β to minimise the residual sum of squares:

RSS = Σᵢ₌₁ⁿ (yᵢ - ŷᵢ)²

Squaring does two things. It makes a miss of −3 and a miss of +3 equally bad, and it makes a miss of 6 four times as costly as a miss of 3, not twice. That is why one 74,600 car, on the raw price scale, can steer the whole line. You already reduced that effect by modelling ln(price).

## A two-point line, so the formula is visible

Two cars, one feature, no need for a matrix. Age 1 has log-price 10.0. Age 5 has log-price 9.2. The unique straight line through both points has slope

β₁ = (9.2 - 10.0)/(5 - 1) = (-0.8)/4 = -0.2

and intercept

β₀ = 10.0 - (-0.2)(1) = 10.2

ŷ = 10.2 - 0.2 · age

At age 1: 10.2 - 0.2 = 10.0. At age 5: 10.2 - 1.0 = 9.2. RSS is 0 because two points always determine a line. The moment you add a third car that is not on the line, RSS is positive and the slope is a compromise. That compromise, for many features, is the normal equation

β̂ = (Xᵀ X)⁻¹ Xᵀ y

You do not invert this by hand. scikit-learn does. The formula tells you when the fit will fail: if Xᵀ X has no inverse. That happens when one column is an exact copy or exact sum of others (all K dummies plus an intercept). Topic 10’s `drop="first"` is what keeps this inverse alive.

With thousands of rows and standardised columns, sklearn’s `LinearRegression` uses a numerically stable solver rather than a raw inverse. The β it returns is still the RSS minimiser.

## What each coefficient means after your preprocessing

Read a coefficient only in the units you actually fed the model.

| If the column is… | Then βⱼ = 0.1 means… |
| --- | --- |
| A standardised numeric feature, and y = ln(price) | One training standard deviation higher on that feature is associated with e^(0.1) - 1 ≈ 10.5% higher price, other columns held fixed. |
| A 0/1 dummy that was **not** standardised, same y | Being in that level, versus the dropped reference level, is associated with about 10.5% higher price. |
| A raw column in currency, and y is raw price | One extra unit of the column adds 0.1 currency units. You should not be in this case if you logged and scaled. |

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
