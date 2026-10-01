# 4. Distributions and frequency

Notebook: **2.1.2** (numeric, 3 marks) and **2.1.3** (categorical, 3 marks).

## What a distribution is

A distribution is the answer to “which values are common, and which are rare?”

For a **numeric** column the picture is a **histogram**. You cut the number line into bins and count the rows in each bin. The shape tells you skew, spikes, and impossible values.

For a **categorical** column the picture is a **bar chart of counts**, which is `value_counts()`. There are no bins, because `"Diesel"` and `"Benzine"` are labels, not positions on a ruler.

## A histogram by hand

Five cars have ages 1, 2, 2, 5, 9. Use bins \([0, 3)\), \([3, 6)\), \([6, 10)\).

| Bin | Ages inside | Count |
| --- | --- | ---: |
| 0 to 3 | 1, 2, 2 | 3 |
| 3 to 6 | 5 | 1 |
| 6 to 10 | 9 | 1 |

Most cars are young. One car at age 9 is a long right tail. That is the same shape you should look for in `price` and `km`.

The mean of those ages is \((1+2+2+5+9)/5 = 3.8\). The median is the middle value after sorting, which is 2. When the mean is well above the median, the right tail is pulling the average. Price in this file does that: median 16,900, mean 18,024, maximum 74,600.

## What to plot

Numeric histograms, one column each:

`km`, `Gears`, `age`, `Previous_Owners`, `hp_kW`, `Displacement_cc`, `Weight_kg`, `cons_comb`, and `Inspection_new` if you are treating it as numeric.

```python
numeric_predictors = [
    "km", "Gears", "age", "Previous_Owners", "hp_kW",
    "Displacement_cc", "Weight_kg", "cons_comb",
]
df[numeric_predictors].hist(bins=30, figsize=(12, 8))
```

Look for three things and write them down:

- **Skew.** `km` and engine size often have a long right tail. A tail is a reason to think about a transform or about outliers, not an automatic reason to delete rows.
- **Spikes.** `Gears`, `age`, and `Previous_Owners` take few distinct values. A histogram of gears will look like a handful of bars. That is expected. They are still numeric.
- **Impossible values.** Negative kilometres, age below 0, or consumption of 0 would be data errors. Confirm the axis instead of assuming the plot is fine.

Categorical bars:

`make_model`, `body_type`, `vat`, `Type`, `Fuel`, `Paint_Type`, `Upholstery_type`, `Gearing_Type`, `Drive_chain`.

```python
for col in categorical_predictors:
    df[col].value_counts().plot(kind="bar", title=col)
```

The mark is not the chart alone. Under each group of charts, state which levels are so rare that a later one-hot column would be estimated from a handful of cars. Topic 5 is the repair. Topic 6 is why the four feature-list columns do **not** belong in this bar-chart loop as raw strings.

## What usually goes wrong

- Drawing one histogram of a categorical column after label-encoding it as 0, 1, 2. The gaps between those integers are meaningless, so the histogram lies.
- Using 100 bins on `Previous_Owners`, which has only a few distinct values. The plot looks empty. Use fewer bins, or a count plot, for low-cardinality numbers.
- Forgetting to exclude `price` from the predictor list. You do plot `price`, but in section 2.1.5 as the **target**, not as a predictor.
