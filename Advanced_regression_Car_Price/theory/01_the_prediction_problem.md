# 1. The prediction problem

## What this assignment is

A reseller lists used cars. For each car they know the model, mileage, age, power, fuel, gearbox, and a list of options. They want a number: the price they should expect.

That is **supervised regression**.

- **Supervised** means every training row already has the answer. Here the answer is `price`.
- **Regression** means the answer is a continuous number, not a class such as “cheap” or “expensive”.
- **The model** is a formula that turns the other columns (the **features**, or predictors) into a predicted price.

Write one car as a row. If the features are age, kilometres, and power, the model looks like:

\[
\widehat{\text{price}} = \beta_0 + \beta_1 \cdot \text{age} + \beta_2 \cdot \text{km} + \beta_3 \cdot \text{hp}
\]

\(\beta_0\) is the intercept: the predicted price when every feature is zero. Each other \(\beta\) is the change in predicted price when that feature increases by one unit and the other features stay fixed.

The hat on price means “prediction”, not the true listed price.

## A three-car example you can finish by hand

Ignore every column except age. Suppose the reseller has only these three cars:

| Age (years) | Listed price |
| --- | ---: |
| 1 | 22,000 |
| 3 | 18,000 |
| 5 | 14,000 |

Price falls by 2,000 for every extra year of age. The line that passes through all three points is:

\[
\widehat{\text{price}} = 24{,}000 - 2{,}000 \cdot \text{age}
\]

Check the middle car: \(24{,}000 - 2{,}000 \cdot 3 = 18{,}000\). The error is zero. Real data will not be this clean, because two cars of the same age still differ in kilometres, brand, and options. The model’s job is to find the \(\beta\) values that make the errors as small as possible **on average**, not to hit every car exactly.

## Why the brief also talks about feature importance and regularisation

A coefficient is a kind of importance, but only after the features are on a comparable scale. A coefficient of \(-0.02\) on kilometres looks tiny next to \(-2{,}000\) on age, because one extra kilometre is a much smaller change than one extra year. Scaling, covered later, is what makes those coefficients comparable.

**Regularisation** (Ridge and Lasso) is what you do when the model can fit the training cars too closely, especially after you turn every option and every model name into its own column. The penalty pulls coefficients toward zero so the formula stays stable on cars the model has not seen.

## Business reading of the same model

For the reseller, three different questions fall out of one fitted model:

- **Price prediction.** Given a new listing’s features, what price should we expect?
- **Market analysis.** After you compare average price by fuel, body, and condition, you can see which slice of the market is expensive.
- **Feature importance.** After scaling, the largest coefficients are the levers: age, mileage, and power usually dominate paint colour.

The model does not know the reseller’s margin, the local demand shock, or a car that was damaged and repaired. Those are missing variables. A low error on this file does not mean the price is “fair” in the market. It means the price is consistent with the other listings in this file.

## What you should be able to say before you code

The target is `price`. Everything else is a candidate predictor. You will later decide that some columns are not usable as they stand (`Comfort_Convenience` and the other comma-separated lists), and that `price` itself should be transformed because it is skewed. The prediction problem stays the same after that transform: learn a linear formula, measure how far it misses, then see whether Ridge or Lasso makes those misses smaller or the coefficients more trustworthy.
