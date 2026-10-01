# 7. The skewed target and the log transform

Notebook: **2.1.5**, 3 marks. The brief says the target looks skewed and tells you to transform it.

## What skew means on price

Sort the 15,915 prices. The middle one, the **median**, is 16,900. The **mean** is 18,024. The cheapest car is 4,950 and the most expensive is 74,600. Half the cars sit at or below 16,900, but the right tail stretches to 74,600, so the average is pulled above the median. That is right skew.

A histogram of `price` will show a peak in the teens of thousands and a long tail to the right. Linear regression fits a line by minimising squared errors. A car priced at 74,600 produces a huge square if the line misses it, so that one car drags the line toward itself. The log transform is how you stop the tail from dominating, without deleting the expensive cars.

## The formula

Use the natural logarithm:

y' = ln(y)

provided every price is positive. Every price in this file is at least 4,950, so the log is defined. You do not need ln(1+y) here. That form is for data that contains zeros.

| Price | ln(price) | Distance from ln(16,900) |
| ---: | ---: | ---: |
| 4,950 | 8.507 | −0.919 |
| 16,900 | 9.735 | 0 |
| 74,600 | 11.220 | +1.485 |

On the raw scale the expensive car is 74,600 / 16,900 ≈ 4.4 times the median, a gap of almost 58,000 currency units. On the log scale the gap from the median is 1.485, while the cheap car is 0.919 on the other side. The tail is still there, but it no longer overwhelms squared error.

Check a multiplication you will use later. Adding a constant on the log scale is a **percentage** on the price scale:

ln(y) + c = ln(y · e^(c))

| Add this to ln(price) | Multiply price by | Percentage change |
| ---: | ---: | --- |
| 0.095 | e^(0.095) ≈ 1.10 | about +10% |
| 0.693 | e^(0.693) ≈ 2.00 | about +100% |
| −0.105 | e^(-0.105) ≈ 0.90 | about −10% |

So if a fitted model on ln(price) gives leather upholstery a coefficient of +0.095, and that column is a raw 0/1 (not standardised), leather is associated with about 10% higher price, holding the other features fixed. It is not associated with 0.095 currency units.

## Undo the transform when you talk about money

The model predicts y'. The reseller wants a price:

predicted price = e^(ŷ′)

If you report MAE on y', the unit is “log currency”, which nobody can put on a sticker. Compute predictions, exponentiate, then compute MAE in currency units if the conclusion needs a business number. Keep one metrics table on the log scale too, because that is the scale the model was trained on. Topic 13 shows both calculations. Be consistent across ordinary regression, Ridge, and Lasso: same y, same inverse, same metric.

sklearn will not undo the log for you. If you pass ln(price) as `y`, every score is on that scale.

## What to produce

1. Histogram of raw `price`, with the mean and median marked or stated.
2. One sentence: right skew, mean above median, tail to 74,600.
3. `y = np.log(df["price"])`, then a second histogram that is closer to symmetric.
4. Keep raw `price` in the dataframe so you can inverse-transform predictions later. Model against `y`, not against a column you have overwritten and forgotten.

## What usually goes wrong

- Using log₁₀ in the plot and ln in the model, then mixing the coefficients. Either base works for symmetry. Natural log is the one used in the percentage table above. Stick to one.
- Interpreting a coefficient as currency after a log transform.
- Transforming `price`, and also transforming `km` with a log, and then describing every coefficient as a percentage. Only the target transform changes the unit of y. A log on a predictor has a different reading (elasticity). Do not add it unless you mean to explain it.
