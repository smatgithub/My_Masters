# 10. Feature engineering and encoding

Notebook: **2.4.1** (redundant columns and new columns), **2.4.3** (encoding, 4 marks). The list columns are topic 6. Do that work before encoding.

## What feature engineering is

The model can only use columns you give it. Feature engineering is the decision to drop a column that repeats another column, or to build a column that states a business fact the raw columns only imply.

A redundant column is one that restates information you already have. After you split `make_model` you do not also need a hand-built `brand` column **and** the full model dummy, unless you drop `make_model` and keep only the brand. Keeping both lets the model split one fact across two coefficients.

## New columns that have a meaning on this file

Build a column only if you can say what one extra unit means.

**Usage intensity.** Kilometres per year of age:

\[
\text{km per year} = \frac{\text{km}}{\text{age}}
\]

This separates a 3-year-old car with 90,000 km from a 3-year-old car with 20,000 km more clearly than `km` and `age` alone. The trap is age 0. A new car makes the ratio undefined. Guard it: if age is 0, set the ratio to missing and then fill it, or define the denominator as \(\max(\text{age}, 1)\), and write down which guard you used. If you keep both `km` and `km per year`, VIF will rise because each is a rearrangement of the other. Prefer one story: either the pair `(km, age)`, or `(km per year, age)`, not all three without a VIF check.

**Power density.**

\[
\text{kW per litre} = \frac{\text{hp\_kW}}{\text{Displacement\_cc} / 1000}
\]

A 66 kW engine in 1.4 litres is a different machine from 66 kW in 2.0 litres. Again, do not keep this **and** both parents unless VIF says the overlap is tolerable. A clean choice is to keep `hp_kW` and `Displacement_cc` as they are, and add a density column only if the residual plots still show structure. You do not have to add columns to score the marks. A short comment that says “I checked brand, age, and km and did not add a ratio because the two raw columns already enter the linear model” is a real decision.

**Equipment counts.** Topic 6. `n_comfort` and the other counts are engineered features.

## Encoding: turn labels into numbers the formula can multiply

A linear model computes \(\beta \cdot x\). It cannot multiply \(\beta\) by the string `"Diesel"`.

**One-hot (dummy) encoding** is the right default for these categories, because the levels are not in a numeric order. Metallic is not “more paint” than Uni/basic.

\[
x_{\text{Diesel}} =
\begin{cases}
1 & \text{if Fuel is Diesel} \\
0 & \text{otherwise}
\end{cases}
\]

For a column with \(K\) levels you keep \(K-1\) dummies. The missing level is the **reference**. Its effect sits inside the intercept. If Fuel has Benzine, Diesel, and LPG/CNG, and you drop Benzine, then:

\[
\widehat{y} = \beta_0 + \beta_D \cdot \text{Diesel} + \beta_L \cdot \text{LPG} + \cdots
\]

A Benzine car gets only \(\beta_0\) from this group. A Diesel car gets \(\beta_0 + \beta_D\). So \(\beta_D\) is the Diesel-versus-Benzine difference, not the absolute Diesel price.

If you keep all \(K\) dummies **and** an intercept, the columns sum to 1 and ordinary least squares cannot invert \(X^\top X\). sklearn’s `OneHotEncoder(drop="first")` drops one level for you. Ridge and Lasso can tolerate the full set because the penalty breaks the tie, but dropping one level keeps the three models comparable. Do that.

**Do not** integer-encode `make_model` as Audi A1 = 0, Audi A3 = 1, Opel Astra = 2. That tells the model that Astra is twice A3, which is false.

**Ordinal encoding** is only for a column whose levels have a true order you believe is evenly spaced. `Type` is tempting and wrong: the gap from Used to Demonstration is not the same kind of step as Demonstration to New. One-hot it, after the grouping you chose in topic 5.

`Inspection_new` is already 0/1. Leave it. `Gears` can stay numeric if you are willing to say “one extra gear has a constant effect”. If the histogram shows two or three isolated gear counts, one-hot encoding gears is also defensible. Pick one and write the sentence.

```python
from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")
```

`handle_unknown="ignore"` matters after the split: a test-set level you never saw becomes all zeros, which means “looks like the reference class”, rather than crashing. Combined with the `"Other"` bucket from topic 5, unknown levels should be rare.

## What usually goes wrong

- Fitting the encoder on the full dataset before the split, so the test-set levels help decide the column list. Fit the encoder on the training rows only, then `transform` the test rows.
- One-hot encoding the four equipment strings as if each full string were one level. Split them first (topic 6).
- Standardising the dataframe before encoding, which does nothing useful to strings and will fail or be skipped. Encode, then scale the numeric columns. Topic 11 fixes the order.
