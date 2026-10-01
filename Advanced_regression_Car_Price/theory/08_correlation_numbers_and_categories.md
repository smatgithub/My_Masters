# 8. Correlation: numbers and categories

Notebook: **2.2.1** (3 marks) and **2.2.2** (3 marks).

## Pearson correlation, for numeric columns

Correlation measures a straight-line association between two numeric columns. It does not mean one causes the other.

r = (Σᵢ₌₁ⁿ (xᵢ - x̄)(yᵢ - ȳ))/(√(Σᵢ₌₁ⁿ (xᵢ - x̄)² Σᵢ₌₁ⁿ (yᵢ - ȳ)²))

r is always between −1 and +1. Zero means no straight-line association. It can still hide a curve.

## A three-car example that comes out exact

| Car | km | Price |
| --- | ---: | ---: |
| A | 10,000 | 25,000 |
| B | 50,000 | 18,000 |
| C | 90,000 | 11,000 |

Means: x̄ = 50,000, ȳ = 18,000.

Deviations of km: −40,000, 0, +40,000.  
Deviations of price: +7,000, 0, −7,000.

Numerator:

(-40,000)(7,000) + 0 + (40,000)(-7,000) = -560,000,000

Denominator:

√((2 · 40,000²)(2 · 7,000²)) = 40,000 · 7,000 · 2 = 560,000,000

r = (-560,000,000)/560,000,000 = -1

Price falls in a perfect straight line as kilometres rise. On the real file, r between `km` and `price` will be negative but not −1, because age, power, and model also move the price. Compute it; do not assume the toy result.

## What the heatmap is for

Section 2.2.1 asks for a correlation map **between features and the target**. Include the numeric predictors and the target you are actually modelling. If the model uses ln(price), correlate with ln(price), and say so. Correlating with raw price and then modelling log price answers a slightly different question.

```python
cols = numeric_predictors + ["log_price"]
corr = df[cols].corr(method="pearson")
# seaborn heatmap of corr, annotated
```

Read the heatmap in two passes:

- **Target column.** Large |r| means the feature moves with price. Expect `age` and `km` to be negative. Expect `hp_kW` to be positive. A near-zero correlation does not always mean “drop it”; a feature can matter only after you hold another feature constant. It does mean it is a weak solo signal.
- **Pairs of predictors.** Two predictors with |r| near 1, such as engine size and power if that is what you find, will fight over the same credit. That fight is multicollinearity. Topic 14 measures it with VIF. The heatmap is the early warning.

## Categories do not have a Pearson r

You cannot compute a meaningful correlation between `Fuel` and price, because `"Diesel"` minus `"Benzine"` is not a number.

Section 2.2.2 asks you to **compare the average target across categories**. Group by the label and summarise y:

ȳ(g) = (1 / n(g)) × Σᵢ∈g yᵢ

| Fuel, imaginary extract | Cars | Mean ln(price) | Mean price |
| --- | ---: | ---: | ---: |
| Diesel | 3 | 9.70 | 16,300 |
| Benzine | 3 | 9.55 | 14,000 |
| Electric | 1 | 10.13 | 25,000 |

Diesel’s log-price average is 0.15 above Benzine’s. Because e^(0.15) ≈ 1.16, that is about 16% higher on the price scale, in this toy table. Electric looks higher still, but it is one car. A bar of means without the counts will over-read Electric. Always print the count next to the mean.

```python
summary = (
    df.groupby("Fuel")["log_price"]
    .agg(mean="mean", count="size")
    .sort_values("mean", descending=False)
)
```

Do this for `make_model`, `body_type`, `Type`, `Fuel`, `Gearing_Type`, `Upholstery_type`, `Drive_chain`, `Paint_Type`, and `vat`. A box plot of log price by category shows the spread, not only the mean. The mean can be similar while the boxes barely overlap, or the means can differ while the boxes overlap so much that the category is a weak predictor.

## What usually goes wrong

- Calling a heatmap of one-hot columns “the correlation analysis” and skipping the group-mean table. The brief separates the two.
- Dropping every feature with |r| < 0.1 before the model. Some of those features matter conditionally. Drop only features you have a reason to drop: near-constant, duplicate, or a VIF problem later.
- Comparing mean **raw** price when the model target is log price, then being surprised that the ranking of categories shifts. Use the modelling target in this section.
