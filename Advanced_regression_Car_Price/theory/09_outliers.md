# 9. Outliers

Notebook: **2.3.1** (2 marks, identify) and **2.3.2** (3 marks, handle).

## What an outlier is

An outlier is a value that sits far from the bulk of the same column. It is not automatically an error. A car at 74,600 can be a genuine high-spec listing. A car with `km` equal to 0 can be a new car, which is also genuine. The task is to find the extremes, decide which mechanism produced them, and then either keep them, cap them, or remove them. Write the reason.

## The IQR fence, computed by hand

Sort a column and pick the 25th percentile Q₁ and the 75th percentile Q₃. The interquartile range is the width of the middle half:

IQR = Q₃ - Q₁

Tukey’s fences are:

lower = Q₁ - 1.5 · IQR, upper = Q₃ + 1.5 · IQR

Toy prices, already sorted: 10, 12, 13, 14, 15, 16, 18, 40. Eight numbers. Q₁ is the value at the 25% position. A simple teaching split takes the median of the lower half and the upper half: lower half 10, 12, 13, 14 so Q₁ = 12.5; upper half 15, 16, 18, 40 so Q₃ = 17. Then

IQR = 17 - 12.5 = 4.5

upper = 17 + 1.5 · 4.5 = 23.75

40 sits above 23.75, so it is an outlier under this rule. 10 sits above the lower fence 12.5 - 6.75 = 5.75, so it is not.

Pandas uses a precise percentile, so your notebook’s fence will not match this hand split exactly. The idea is the same. Apply it to each numeric column and report how many rows fall outside.

```python
def iqr_bounds(series):
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    return q1 - 1.5 * iqr, q3 + 1.5 * iqr
```

A box plot draws the same fence: the whiskers stop at the last point inside the fence, and points beyond are drawn as dots. One box plot per numeric column is the identification the first mark expects.

## z-scores, and why they are a poor first tool here

A z-score measures distance from the mean in standard deviations:

zᵢ = (xᵢ - x̄)/s

A common cut is |z| > 3. This rule assumes the column is roughly symmetric. Price is skewed, so the mean and s are themselves inflated by the tail, and the rule both misses some tail points and flags ordinary ones. Prefer the IQR fence on the raw columns. If you already took ln(price), the log-price z-score is more reasonable because the distribution is closer to symmetric. Do not stack three different rules and delete every row that fails any of them.

## How to handle them on this dataset

Use a different decision for different columns. Deleting every flagged row is the weakest default, because IQR flags on a skewed column are often the real tail.

| Situation | What it usually is here | A defensible handling |
| --- | --- | --- |
| Very high `price` | Real expensive cars. You already log-transform the target, which reduces their pull. | Keep them. Say that the log is the handling. |
| Very high `km` | Real high-mileage cars, unless the value is absurd (millions of km on a 2-year-old car). | Keep, or cap at a high percentile such as 99% if a few points are physically implausible. |
| `km` near 0 | Consistent with New, Pre-registered, Demonstration. | Keep. Check that they line up with `Type` before calling them errors. |
| `age` of 0 | A new or current-year car. | Keep. Do not divide by age later without a guard. |
| A numeric value that is impossible | Negative price, negative km, weight of 0. | This would be an error. Drop or correct the row, and say which. |

Capping, also called winsorising, replaces values above the upper fence with the fence itself. The row stays, but the extreme becomes equal to the fence. That reduces leverage. It also hides the true value, so only cap a column when you believe the extreme is a recording problem or when you have shown that a few points dominate the fit.

If you delete rows, delete from the modelling table **before** the train/test split, using a rule fixed in advance, and report how many rows you removed and what fraction of 15,915 that is. Removing 2% is a different decision from removing 15%.

## What usually goes wrong

- Flagging outliers on one-hot columns. A 0/1 column will always “fail” a fence. Run the check on the original numeric measurements only.
- Fitting the fence on all data, deleting rows, then splitting. For a simple global rule this is common in coursework. The stricter habit is to compute the fence on the training set and apply it to both. Either way, state what you did.
- Removing price outliers and also log-transforming, then claiming both were necessary, without showing the histogram after each step. Show the plot after the log. If the tail is tamed, you do not also need to delete those cars.
