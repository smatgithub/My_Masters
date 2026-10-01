# 3. Missing values

Notebook: **2.1.1**, 1 mark.

## What a missing value is

A missing value is a cell where the measurement was not recorded. In pandas it appears as `NaN`. An empty string in the CSV also becomes missing if you treat it as such. The number you report is the **proportion** in each column:

\[
\text{missing proportion}_j = \frac{\text{number of missing rows in column } j}{\text{number of rows}}
\]

On this file that proportion is **0 for every column**. You still have to show the calculation. A marker who only sees an empty code cell cannot know that you checked.

## A hand example of the two repairs

Suppose a 5-car extract had missing age:

| Car | Age | Price |
| --- | ---: | ---: |
| A | 2 | 20,000 |
| B | 4 | 16,000 |
| C | missing | 18,000 |
| D | 6 | 12,000 |
| E | 4 | 17,000 |

Three of five ages are present. The proportion missing is \(1/5 = 0.20\).

**Drop the row** if the hole is rare and the row cannot be used. You would lose car C and fit the model on A, B, D, and E. You also lose that car’s price, kilometres, and options, which is wasteful when only one field is missing.

**Impute** when you want to keep the row. The mean of the known ages is \((2+4+6+4)/4 = 4\). Filling C with 4 lets the row stay. The cost is that you invented a value. If many rows are filled with the same mean, that column’s variance shrinks and the model becomes too confident.

For a category, the equivalent fill is the **mode** (the most common label), or an explicit level called `"Missing"`. `"Missing"` is better when the fact that the value was absent might itself be informative. Mean-imputation is for numbers. Mode-imputation is for labels. Do not average the strings `"Diesel"` and `"Benzine"`.

## What to do on this file

```python
missing_proportion = df.isna().mean().sort_values(ascending=False)
missing_proportion
```

Write one sentence under the output: every column has proportion 0, so no row was dropped and no value was imputed. That sentence is the mark. If you later create new columns (for example kilometres per year) and divide by an age of 0, you can manufacture missing values that were not in the raw file. Check again after feature engineering.

## What usually goes wrong

- Filling missing values with a mean computed on the **whole** dataset before the train/test split. The test rows then influence the training fill. On this file the issue does not arise, because nothing is missing, but the habit matters on the next assignment.
- Treating the string `"None"` or `"0"` as a real measurement without looking. Here `Inspection_new` equal to 0 means “no new inspection”, which is a real answer, not a missing one.
- Dropping a column because it has missing values, when the column is the target. You cannot impute the answer and then claim to have predicted it.
