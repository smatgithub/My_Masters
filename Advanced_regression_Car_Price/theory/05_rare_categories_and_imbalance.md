# 5. Rare categories and class imbalance

Notebook: **2.1.4**, 3 marks. The brief gives you the meaning of every `Type` level. It tells you to decide the other columns yourself.

## Why a rare level hurts a linear model

One-hot encoding turns a label into a 0/1 column. The coefficient on that column is estimated from the rows where the column is 1.

Audi A2 appears **once**. Its one-hot column is 1 on a single row and 0 everywhere else. The model can use that single row to absorb that car’s price completely. The coefficient looks important and predicts nothing on a future A2, because you have not learned a pattern. You have memorised one listing.

Rear-wheel drive has 4 cars. Electric has 5. Convertible has 8. Perl effect has 32. Renault Duster has 34. These are the same problem in a milder form.

A practical rule for this assignment: if a level has only a few dozen rows, do not give it its own coefficient. Either merge it into a similar level, or merge all rare levels of that column into `"Other"`.

## The hand example

Suppose body type and price look like this:

| Body | Rows | Average price |
| --- | ---: | ---: |
| Sedan | 8,000 | 17,000 |
| Coupe | 8 | 40,000 |

A Coupe coefficient is the model saying “add about 23,000 because this car is a Coupe”. That addition is an average of eight cars. One luxury Coupe moves it by thousands. Folding Coupe into a rare-body bucket, or into the nearest common body if the business meaning matches, stops one listing from owning a coefficient.

This is **not** the same problem as a 50/50 class balance in a classifier. The brief says “class imbalances” because a level with 8 rows against a level with 8,000 rows will dominate counts and produce an unstable dummy. You are not trying to make every body type equally common. You are trying to avoid coefficients estimated from a handful of cars.

## What `Type` means, and a defensible grouping

The brief defines the words:

| Level | Rows | Meaning |
| --- | ---: | --- |
| Used | 11,095 | Ordinary used car |
| New | 1,649 | New-like, not necessarily unused. May have several pre-registrations |
| Pre-registered | 1,364 | Already registered by the seller |
| Employee's car | 1,011 | Staff car, short time, short distance |
| Demonstration | 796 | Trial car, short time, short distance |

None of these is a one-row level. Do not merge them only because the counts differ. Merge them only if the **price mechanism** is the same.

A grouping you can justify in a comment:

- `Used` stays `Used`.
- `New` and `Pre-registered` become `Nearly new`, because both are registration stories rather than high-use cars.
- `Employee's car` and `Demonstration` become `Short use`, because the brief says both were driven briefly and not far.

You could also keep all five. Five levels with at least 796 rows each are estimable. The mark is the justification, not a secret grouping. If you keep all five, say why the brief’s definitions did **not** convince you to merge.

## A consistent rule for the other columns

Apply one written rule, for example: “any level with fewer than 50 cars is recoded”.

| Column | Levels under a 50-car rule | A sensible recode |
| --- | --- | --- |
| `make_model` | Audi A2 (1), Renault Duster (34) | `Other`, or fold A2 into a rare-Audi bucket. Do not fold Duster into Clio; they are different products. `Other` is the honest label. |
| `body_type` | Coupe (25), Convertible (8) | `Other`. Off-Road (56) and Transporter (88) sit near the cut. State the cut and stick to it. |
| `Fuel` | Electric (5). LPG/CNG has 64, so it survives a 50-car rule. | Electric → `Other`. |
| `Paint_Type` | Perl effect (32) | Fold into the closest common paint, or `Other`. |
| `Drive_chain` | rear (4) | Fold `rear` into `Other` or into `4WD` only if you argue both are “not front”. `Other` is cleaner. |

```python
rare = df["make_model"].value_counts()
rare_levels = rare[rare < 50].index
df["make_model"] = df["make_model"].where(~df["make_model"].isin(rare_levels), "Other")
```

Recount `value_counts()` after the recode and show that the tiny levels are gone. That before-and-after pair is the evidence for the mark.

## What usually goes wrong

- Merging `Used` with everything else because it is the majority. You would destroy the main condition signal.
- Choosing the threshold after looking at which merge improves the test score. Fix the rule from the training counts, then apply the same map to the test set. Levels that appear only in test data must also become `Other`.
- Replacing rare cars by deleting their rows. Those cars are real listings. Recoding the label keeps the row and throws away only the unstable dummy.
