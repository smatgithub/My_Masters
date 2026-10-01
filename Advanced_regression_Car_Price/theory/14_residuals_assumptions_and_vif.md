# 14. Residuals, the linear-model assumptions, and VIF

Notebook: **3.1.2**, 5 marks. Three checks: residuals against fitted values, the distribution of residuals, and VIF.

## What a residual is

eᵢ = yᵢ - ŷᵢ

It is the miss on car i, on the same scale as y. If y is log price, the residual is a log-price miss. A residual of +0.2 means the car sold for about e^(0.2) - 1 ≈ 22% more than the model predicted. It is not a currency residual unless y is currency.

The assumptions below are about these residuals. They are not extra paperwork. Each one is a way the formula Xβ can be the wrong summary of the data, which is exactly what the conclusion asks: “is a linear model sufficient?”

## 1. Linearity — residual versus fitted plot

Plot eᵢ on the vertical axis and ŷᵢ on the horizontal axis. Add a horizontal line at 0.

If the linear formula is enough, the cloud is a formless band around 0. The running average of the residuals does not curve.

A smile or a frown means the miss depends on the predicted price in a curved way. Typical story: the model overprices mid-range cars and underprices the extremes. A straight line cannot fix that. You would need a transform, a squared term, or a different model family. For this assignment, **show the plot and describe the shape**. You are not required to switch to a tree model. You are required to notice.

```python
residuals = y_train - linreg.predict(X_train_prepared)
fitted = linreg.predict(X_train_prepared)
# scatter fitted vs residuals, axhline at 0
```

Use the training residuals to judge the assumptions of the fitted model. Then look at the test residuals as a confirmation. If only the test plot curves, you may be looking at a small-sample wobble; if both curve, the linear form is strained.

## 2. Constant spread — read it on the same plot

Homoscedasticity means the vertical spread of residuals is roughly stable from left to right. If the cloud is a trumpet, wide at high predicted prices and tight at low ones, the model is less precise on expensive cars than on cheap ones. Squared-error fitting then spends too much attention on the wide end.

Logging the target is the usual cure, and it is why the brief pushed you to transform `price` before this section. If you plot residuals of a raw-price model, you will often see the trumpet. If you plot residuals of the log model, the trumpet should be milder. That comparison is a strong sentence in the conclusion, if you actually make both plots. One plot of the model you kept is enough for the mark.

## 3. Normality — the histogram of residuals

The brief asks for the distribution of residuals. Draw a histogram, and optionally a Q-Q plot.

Normality matters most when you want confidence intervals and p-values for coefficients. For prediction quality, MAE and RMSE do not require normal residuals. A mild skew in the histogram is not a failed assignment. A histogram with two distinct peaks often means a group you forgot to encode (for example all Electric cars sitting in one lump of residuals). Investigate a second peak. Do not delete residuals until the histogram looks bell-shaped.

## 4. Independence

The rows should not be copies of each other or a time series in disguise. This file is a cross-section of listings, so independence is plausible. Do not invent a Durbin-Watson test unless you have an ordered time column. There is no date column in the dictionary.

## 5. Multicollinearity and VIF

Multicollinearity means some predictors are near-copies of each other. The model can still predict well, because either copy carries the signal. The **coefficients** become unstable: a small change in the data swings β from large positive to large negative, and the two correlated features split one effect arbitrarily. Feature-importance comments based on those coefficients are then not trustworthy. Ridge exists largely because of this.

VIF makes the idea a number. For feature j, regress feature j on all the other features and take that regression’s R²ⱼ:

VIFⱼ = 1/(1 - R²ⱼ)

## A hand example

Suppose age is regressed on the other columns and that regression has R² = 0.80. The other columns already explain 80% of the variation in age. Then

VIF(age) = 1/(1 - 0.80) = 5

The variance of the age coefficient is about 5 times what it would be if age were uncorrelated with the other columns. A common coursework rule is:

| VIF | Reading |
| --- | --- |
| 1 | No linear overlap with the other columns |
| 5 to 10 | Worth inspecting. Often tolerable for prediction. |
| Above 10 | The coefficient is poorly identified. Drop, combine, or let Ridge handle it. |

The cut is a convention, not a law. State the cut you use.

```python
from statsmodels.stats.outliers_influence import variance_inflation_factor

# X_vif is the training design matrix as a DataFrame, including a constant
# if you want VIFs that match a model with an intercept.
vifs = [
    variance_inflation_factor(X_vif.values, i)
    for i in range(X_vif.shape[1])
]
```

If a VIF is infinite, two columns are exact linear copies. The usual cause in this assignment is one-hot columns that still include every level plus the intercept, or a column you built as a sum of others (equipment count together with every equipment flag, in a way that reconstructs the count). Drop one column from the exact relationship and recompute.

Practical handling for high but finite VIF, in the order you should try:

1. Drop the engineered column that restates two parents (`km per year` if `km` and `age` stay).
2. Drop one of a tightly correlated pair after checking the heatmap. Keep the one with the clearer business meaning (`hp_kW` rather than a near-duplicate of it).
3. If many equipment flags overlap and prediction is the goal, leave them and let Ridge shrink the group. Say that you did this on purpose. Lasso will likely keep one of the group and zero the rest.

Re-fit the linear model after any drop, and show the new VIF table. The mark is the before-and-after, plus a sentence on whether prediction or interpretation was your reason.

## What usually goes wrong

- Computing VIF on the test set. Use the training matrix.
- Computing VIF before scaling and panicking at huge numbers that are only unit effects. VIF is invariant to scaling of a single column in exact arithmetic, but a badly scaled matrix can make the computation explode. Use the same matrix you fit.
- Deleting every feature with VIF above 5 until almost nothing remains. You can destroy the signal. Prediction can be fine at VIF 8. Instability of the coefficient is the thing you are protecting.
