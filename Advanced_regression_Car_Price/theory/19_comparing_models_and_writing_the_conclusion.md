# 19. Comparing the three models, and writing the conclusion

Notebook: **3.4.1** (2 marks), **3.4.2** (3 marks), **section 4** (10 marks).

Section 4 is not a summary of the theory files. It is a set of claims about **your** fitted models, each claim tied to a number or a plot you produced. The questions in the brief are the outline. Answer them in that order.

## 3.4.1 The metric table

Put ordinary linear regression, Ridge, and Lasso in one table. Same rows as test cars. Same y. Same inverse transform if you report currency.

| Model | Alpha | Train MAE (log) | Test MAE (log) | Test RMSE (log) | Test R² | Test MAE (currency) | Coefficients exactly zero |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Linear | — | from your fit |  |  |  |  | 0, unless a column was unused |
| Ridge | your best |  |  |  |  |  | should be 0 |
| Lasso | your best |  |  |  |  |  | the count you computed |

Fill this from code, not from these notes. The notes do not know your split, your encoding, or your alpha.

How to read a row:

- Test MAE answers “how far off is a typical car?”
- The train-minus-test gap answers “did we overfit?”
- Currency MAE answers “is that miss small relative to a median price of 16,900?” A currency MAE of 1,500 is a very different business result from a currency MAE of 6,000. Divide by 16,900 if you want a rough percentage of the typical car. That ratio is a communication device. It is not R².

A model “wins” only on a metric you named in advance. If Lasso has the best test MAE by 0.001 on the log scale, say the three models are practically tied. Do not crown a winner on noise.

## 3.4.2 The coefficient comparison

Align the three coefficient vectors by feature name. Then:

1. Sort by the absolute ordinary-regression coefficient and show the largest ones, for example the top 15, as grouped bars: linear, Ridge, and Lasso side by side.
2. Separately, list the features whose Lasso coefficient is zero, and show those features’ linear and Ridge coefficients. The pattern you are looking for is: Lasso is zero, Ridge is small but not zero, linear is larger or unstable. That picture is the whole difference between the two penalties (topics 16 and 17).
3. Check the sign on the features you already understand. Age and kilometres should usually be negative on log price. Power should usually be positive. If age comes out positive, something is wrong with the coding, the scaling, or a collinear partner that stole the sign. Do not invent a market story for a sign that is a bug. Go back to VIF and the heatmap.

A coefficient can be large because the feature matters, or because the feature is rare and one-hot. After topic 5, rare levels should already be gone. If a leftover `"Other"` bucket has a huge coefficient and only a few dozen cars, say that the estimate is thin.

## The conclusion, question by question

Write in prose. A marker cannot award 10 marks for a table with no sentences. Use the questions already printed in section 4.

**What did regularisation do?**  
State the best Ridge alpha, the best Lasso alpha, how many coefficients Lasso set to zero, and whether the large coefficients shrank relative to ordinary regression. Refer to the bar chart.

**Did performance improve?**  
Quote test MAE for all three models on one scale. If the test MAE improved, say by how much, in log units and in currency. If it did not, say that too. A small move, such as a currency MAE change of a few tens against a median price of 16,900, is “no material improvement”.

**If it did not improve, why is that still a reasonable outcome?**  
Because ordinary least squares was already close to the right complexity. The training and test errors were already close, so there was little variance to remove. The penalty then adds bias and cannot help the test score. Regularisation can still be worth using because the coefficients are calmer when features overlap (high VIF). Separate “predicts better” from “coefficients are safer to explain”.

**Was there overfitting?**  
Quote train MAE and test MAE for the linear model. Name the gap. If the gap was large and Ridge or Lasso reduced it, that is the overfitting you found and partly corrected. If the gap was small, you did not find meaningful overfitting.

**Was the data sufficient?**  
15,915 rows is a large listing file for a linear model with tens of columns. It is less ample if you kept hundreds of equipment flags. Mention the levels you merged because they had almost no rows (Audi A2, rear-wheel drive, Electric, Convertible). Sufficiency also means columns: the file has no damage history, no region, no service records, and no options that were present but not written in the four list fields. Those missing facts put a floor under the error. No alpha removes that floor.

**Is a linear model sufficient?**  
Answer from the residual plot, not from taste. If the residual cloud is a band around zero, a linear model is a reasonable summary and the three models being close supports that. If the cloud curves, a linear model is missing a bend. Say what you would try next (a squared term, a different target transform, or a non-linear model) without pretending you fitted it. The assignment does not require you to fit that next model. It requires you to notice.

## A shape you can follow

One paragraph of results (the table, in words).  
One paragraph on coefficients and what Lasso removed.  
One paragraph on overfitting, sufficiency, and the residual plot.  
One paragraph on what the reseller should not conclude: the model prices listings that look like this file, it does not appraise a damaged car, and the test MAE is slightly flattered if alpha was chosen on that same test set.

## What usually goes wrong

- Writing the definitions of Ridge and Lasso again and never quoting a test MAE. Section 4 is about outcomes.
- Claiming a feature is “the most important” from an unscaled coefficient.
- Explaining a performance drop after regularisation as a coding failure. It is often the correct statistical result. Report it.
- Pasting these notes into the notebook. The notes are the reasoning. The notebook needs your outputs.
