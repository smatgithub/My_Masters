# 2. Reading this dataset

## What is in the file

`Car_Price_data.csv` has **15,915 rows** and **23 columns**. Each row is one listed car. There are no blank cells in the file as it is stored. That does not mean every column is ready to model. Several columns are text, two look numeric but are really categories, and four columns hide many features inside one string.

The brief’s data dictionary is the contract. Use these roles:

| Role | Columns |
| --- | --- |
| Target | `price` |
| Numeric predictors | `km`, `Gears`, `age`, `Previous_Owners`, `hp_kW`, `Displacement_cc`, `Weight_kg`, `cons_comb` |
| Binary numeric | `Inspection_new` (0 or 1) |
| Categorical predictors | `make_model`, `body_type`, `vat`, `Type`, `Fuel`, `Paint_Type`, `Upholstery_type`, `Gearing_Type`, `Drive_chain` |
| Multi-value text, not a single category | `Comfort_Convenience`, `Entertainment_Media`, `Extras`, `Safety_Security` |

`Gears` is stored as `7.0`. It is still a count of gears, so treat it as numeric. `Inspection_new` is only 0 or 1. You may keep it numeric. Do not one-hot encode it into two columns that sum to 1; that duplicate is perfect collinearity, which breaks ordinary linear regression.

## Counts that should shape your decisions

These frequencies come from the file. They are why later topics talk about rare levels.

| Column | What dominates | What is rare |
| --- | --- | --- |
| `make_model` | Audi A3 (3,097), Audi A1, Opel Insignia, Opel Astra, Opel Corsa, Renault Clio, Renault Espace | Renault Duster (34), Audi A2 (1) |
| `body_type` | Sedans (8,004), Station wagon, Compact | Transporter (88), Off-Road (56), Coupe (25), Convertible (8) |
| `Type` | Used (11,095) | New (1,649), Pre-registered (1,364), Employee's car (1,011), Demonstration (796) |
| `Fuel` | Benzine (8,548), Diesel (7,298) | LPG/CNG (64), Electric (5) |
| `Paint_Type` | Metallic (15,246) | Uni/basic (637), Perl effect (32) |
| `Gearing_Type` | Manual (8,149), Automatic (7,297) | Semi-automatic (469) |
| `Drive_chain` | front (15,707) | 4WD (204), rear (4) |
| `Upholstery_type` | Cloth (12,234) | Part/Full Leather (3,681) |
| `vat` | VAT deductible (15,044) | Price negotiable (871) |

`Type` is not a rare-level problem. Every type has hundreds of cars. The brief still asks you to think about what those labels mean. Rare-level problems are Audi A2, rear-wheel drive, Electric, Convertible, and Perl effect.

## What one row actually contains

The first car is an Audi A1 sedan, listed at 15,770, with 56,013 km, Diesel, 7 gears, age 3, 2 previous owners, 66 kW, a new inspection, metallic paint, cloth, automatic, 1,422 cc, 1,220 kg, front-wheel drive, and combined consumption 3.8 L/100 km.

Its comfort cell is not one category. It is a comma-separated list:

`Air conditioning, Armrest, Automatic climate control, Cruise control, ...`

If you one-hot encode that whole string, two cars that share 12 options and differ by one option become unrelated categories. Topic 6 is about splitting that list into yes/no columns.

## How loading should look in the notebook

Section 1.1 is only the setup. Import pandas, NumPy, matplotlib or seaborn, and scikit-learn as you need them. Load with an explicit path relative to the notebook:

```python
import pandas as pd

df = pd.read_csv("Car_Price_data.csv")
df.shape          # (15915, 23)
df.head()
df.dtypes
df.isna().sum()
```

Then print `describe()` for the numeric columns and `value_counts()` for the categories. Do this before any plot. The shape and the missing-value count are the evidence that later comments refer back to.

## What usually goes wrong

- Treating `price` as a predictor. It is the target. If it stays in `X`, the model cheats and the metrics become meaningless.
- Letting pandas turn `Inspection_new` into a float and then standardising it without comment. It is fine numerically, but you should know it is a flag.
- Encoding `Comfort_Convenience` as a normal categorical column. The notebook warns you about this in section 2.1.3.
