# 6. Multi-value feature lists

Notebook: the note under **2.1.3**, and **2.4.2** (part of the 11 feature-engineering marks).

## Why these four columns are not categories

`Comfort_Convenience`, `Entertainment_Media`, `Extras`, and `Safety_Security` store **lists** of equipment, separated by commas. One cell might be:

`Bluetooth, Hands-free equipment, On-board computer, Radio`

Another car might have those four plus `Sound system`. If you treat the whole string as one category, those two cars fall into unrelated levels even though they share almost everything. The number of distinct strings is large, and most strings appear once. That is the rare-level problem from topic 5, multiplied.

The repair is **multi-hot encoding**. Split the string, and give every equipment item its own 0/1 column.

## A hand example

Three cars:

| Car | Entertainment_Media |
| --- | --- |
| A | Radio, Bluetooth |
| B | Radio |
| C | Bluetooth, Sound system |

Split on the comma, strip spaces, and mark presence:

| Car | Radio | Bluetooth | Sound system |
| --- | ---: | ---: | ---: |
| A | 1 | 1 | 0 |
| B | 1 | 0 | 0 |
| C | 0 | 1 | 1 |

Car A is no longer a unique category. It is “has Radio” and “has Bluetooth”. The linear model can learn a price for Bluetooth that is shared by A and C.

The same split applies independently to the other three list columns. Prefix the column names (`comfort__`, `media__`, `extras__`, `safety__`) so `Radio` cannot collide with a comfort item of the same name.

## The brief’s dimensionality warning

A 0/1 column that is almost always 1 does not explain differences in price. If 15,800 of 15,915 cars have ABS, the ABS column is nearly constant. Its coefficient is estimated from the 115 cars without ABS, and it adds a column to every later VIF and Ridge calculation.

A column that is almost always 0 has the same flaw in the other direction.

The notebook tells you to find items present in most cars or absent from most cars, and to drop them. Pick a rule and write it down before you look at model scores. Two workable rules:

- Drop an item if it is present in more than 95% of cars, or in fewer than 5%.
- Or keep only items in a middle band, for example 10% to 90%.

95% of 15,915 is about 15,119 cars. An item above that count is “almost everyone has it”. 5% is about 796 cars. An item below that is “almost nobody has it”.

```python
items = (
    df["Entertainment_Media"]
    .fillna("")
    .str.split(",")
    .explode()
    .str.strip()
)
items = items[items != ""]
prevalence = items.value_counts() / len(df)
keep = prevalence[(prevalence > 0.05) & (prevalence < 0.95)].index
```

Then build the 0/1 columns only for `keep`, and drop the original string column. Repeat for the other three lists. Show a small table of dropped items with their prevalence so the marker can see the rule, not only the resulting width of the dataframe.

## A count feature is a useful companion, not a substitute

You can also add `n_comfort`, `n_media`, `n_extras`, and `n_safety`: how many items were listed. A car with 14 comfort items is equipped differently from a car with 2, even after rare items are dropped. The count is one number and is easy to interpret. It does **not** replace the individual flags, because “has leather steering wheel” and “has cruise control” need not have the same price effect. Use the count if you want a compact signal, and use the filtered flags if you want item-level coefficients. Say which choice you made. Doing both is allowed; they will be correlated, and topic 14 (VIF) is where you check whether that correlation is too strong.

## What usually goes wrong

- Splitting on comma but leaving a leading space, so `" Radio"` and `"Radio"` become two columns.
- Fitting the keep/drop list on all rows and then splitting train and test. Compute prevalence on the **training** rows only, and apply that item list to the test rows. An item that exists only in test data is ignored.
- Leaving the original string column in `X` beside the new flags. The model then sees the same information twice, once in a form it cannot use well.
