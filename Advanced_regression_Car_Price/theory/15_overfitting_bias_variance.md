# 15. Overfitting, bias, and variance

This topic is the reason section 3 exists. The conclusion (10 marks) asks whether you found overfitting, whether regularisation helped, and whether a linear model and this dataset were enough.

## The two ways a model fails

**Bias** is systematic miss. A straight line forced through a curved relationship stays wrong no matter how many cars you add. The residual plot in topic 14 is the picture of bias in the functional form.

**Variance** is sensitivity to which cars happened to be in the training file. A model with a separate coefficient for every rare option will change those coefficients sharply if a few listings are swapped. The predictions on new cars jump around. That instability is high variance.

Overfitting is high variance in practice: the model fits the training cars, including their accidents and odd options, and looks worse on new cars.

Underfitting is high bias in practice: the model is too simple, so training error and test error are both large and close to each other.

## The comparison that detects it

Fit on the training set. Score **both** sets with the same metric.

| Pattern | What it means |
| --- | --- |
| Train MAE and test MAE are close, and both are acceptable for the reseller | The linear model is generalising. Little overfitting. |
| Train MAE is much smaller than test MAE | The model memorised training cars. Overfitting. |
| Both MAEs are large, and they are close | Underfitting, or the features do not contain the price. Adding Ridge will not invent a missing column. |
| Test MAE is slightly **smaller** than train MAE | Can happen by chance on one split. It is not “negative overfitting”. Draw one split’s gap; do not over-read a small reversal. |

“Much smaller” needs your units. If log-price MAE is 0.12 on train and 0.13 on test, that is a small gap. If it is 0.05 on train and 0.20 on test, the model is memorising. Report the numbers rather than the adjective alone.

\(R^2\) gives the same reading: a large drop from train \(R^2\) to test \(R^2\) is overfitting.

## Why this dataset can overfit even though there are 15,915 rows

15,915 rows is enough for a model with a few dozen columns. It is less comfortable after you explode the equipment lists. If you keep 200 equipment flags plus model, body, fuel, and paint dummies, you can have several hundred columns. Rows per column fall, and Lasso’s ability to set coefficients to zero becomes the point of section 3.3.

A second source of overfitting is a dummy for a level with 1 row (Audi A2), which topic 5 already asked you to remove. Regularisation is the second line of defence, not the excuse to skip that cleaning.

## What regularisation does to this tradeoff

Ridge and Lasso add a penalty so the fit is **not** the pure RSS minimiser. Training RSS is allowed to get worse. In return, coefficients stay smaller and predictions usually move less when the training sample changes. Test error can fall. It does not have to. If the ordinary model was not overfitting, the penalty only adds bias, and test error can rise a little. That outcome is a valid conclusion. Topic 19 shows how to write it without apologising.

## A numerical sketch

Suppose ordinary regression has train MAE 900 and test MAE 1,600, in currency after the inverse transform. The gap of 700 is the overfitting signature.

Ridge, with a moderate \(\alpha\), might move those to train MAE 1,050 and test MAE 1,300. Training got worse, test got better, the gap shrank. Regularisation helped generalisation.

If instead Ridge moves them to 1,100 and 1,650, test got worse. The ordinary model was already near the right complexity, and the penalty over-shrunk real signal. You still report it.

These figures are illustrations of the patterns, not the results in your file. Your conclusion must use the MAEs you compute.

## What usually goes wrong

- Declaring overfitting because training \(R^2\) is below 1. No model of used-car prices will fit perfectly. Overfitting is a **gap** between train and test, not an imperfect train score.
- Tuning \(\alpha\) on the test set and then using the small train-test gap as proof that you did not overfit. You used the test set to choose the model, so the gap is optimistic. Topic 18 separates the search from the final quote.
