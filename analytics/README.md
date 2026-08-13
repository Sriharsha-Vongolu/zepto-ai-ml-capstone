# Module 2 — Analytics Pipeline

This module implements the Titanic analytics and predictive-modeling workflow required for Module 2 of the capstone project.

The module is designed as one continuous pipeline. The Titanic dataset is loaded only once using Seaborn in `01_eda.ipynb` and is immediately saved as `titanic.csv` for offline use. The later modeling stage continues using this committed dataset rather than making another `sns.load_dataset()` call.

## Files

* `01_eda.ipynb` — dataset profiling, cleaning, exploratory analysis, visualizations, and standardization check
* `02_modeling.ipynb` — predictive modeling pipeline
* `titanic.csv` — committed offline fallback generated from the single Seaborn dataset load
* `best_pipeline.joblib` — complete fitted preprocessing and classification pipeline generated during Part B

## Installation

Activate the project virtual environment and install the required packages:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install pandas numpy seaborn matplotlib scikit-learn imbalanced-learn joblib jupyter
```

## Dataset Loading

The Titanic dataset is loaded once in `01_eda.ipynb` using:

```python
df = sns.load_dataset("titanic")
```

Immediately after loading, the dataset is saved as:

```python
df.to_csv(TITANIC_CSV, index=False)
```

The saved `titanic.csv` is committed inside `/analytics` and acts as the offline fallback for the remainder of the module.

# Part A — Profiling, Cleaning, and Data Story

## Dataset Profile

The original dataset contains:

* **891 rows**
* **15 columns**

The notebook reports:

* `df.shape`
* `df.info()`
* `df.describe()`
* missing-value percentage for every affected column

## Missing-Value Analysis and Cleaning

Missing-value handling follows the required percentage-based threshold rule:

* Less than 5% missing → drop affected rows
* 5%–30% missing → impute
* More than 30% missing → explicitly decide whether to drop the column or represent missingness separately

The measured missing-value percentages and cleaning decisions are:

| Column        | Missing % | Cleaning Decision   |
| ------------- | --------: | ------------------- |
| `deck`        |    77.22% | Drop the column     |
| `age`         |    19.87% | Impute using median |
| `embarked`    |     0.22% | Drop affected rows  |
| `embark_town` |     0.22% | Drop affected rows  |

`deck` was dropped because **77.22%** of its values were missing. At this level of missingness, imputing most of the column would introduce too much artificial information.

`age` had **19.87%** missing values, which falls within the required 5%–30% range. Because `age` is numeric, its missing values were replaced using the median.

`embarked` and `embark_town` each had only **0.22%** missing values. Because this is below the 5% threshold, the affected rows were removed.

After cleaning, the dataset contains:

* **889 rows**
* **14 columns**
* **0 remaining missing values** in all retained columns

# Univariate Analysis

Histograms and box plots are produced for both `age` and `fare`.

## IQR Outlier Analysis

Outliers are detected using the required IQR rule:

```text
Lower bound = Q1 - 1.5 × IQR
Upper bound = Q3 + 1.5 × IQR
```

The calculated results are:

| Feature |      Q1 |      Q3 |     IQR | Lower Bound | Upper Bound | Outlier Count |
| ------- | ------: | ------: | ------: | ----------: | ----------: | ------------: |
| Age     | 22.0000 | 35.0000 | 13.0000 |      2.5000 |     54.5000 |        **65** |
| Fare    |  7.8958 | 31.0000 | 23.1042 |    -26.7605 |     65.6563 |       **114** |

The `fare` variable contains substantially more IQR-defined outliers than `age`, which is also visible in its box plot.

## Fare Distribution

The calculated fare statistics are:

| Statistic |   Value |
| --------- | ------: |
| Mean      | 32.0967 |
| Median    | 14.4542 |
| Mode      |  8.0500 |

The ordering is:

```text
Mean > Median > Mode
32.0967 > 14.4542 > 8.0500
```

Therefore, the `fare` distribution is **right-skewed**. The comparatively high mean indicates that a smaller number of passengers with very high fares pull the average upward, while most observations remain concentrated at lower fare values.

# Bivariate Analysis

## Survival Rate by Sex

Boolean masking was used to calculate survival rates separately for each sex.

| Sex    |         Survival Rate |
| ------ | --------------------: |
| Female | 0.740385 (**74.04%**) |
| Male   | 0.188908 (**18.89%**) |

Female passengers had a substantially higher survival rate than male passengers. Approximately 74% of female passengers survived compared with approximately 19% of male passengers, showing a strong relationship between sex and survival outcome.

## Survival Rate by Passenger Class

| Passenger Class |         Survival Rate |
| --------------- | --------------------: |
| 1               | 0.626168 (**62.62%**) |
| 2               | 0.472826 (**47.28%**) |
| 3               | 0.242363 (**24.24%**) |

Survival probability decreases consistently as passenger class number increases. First-class passengers had the highest survival rate at approximately 62.62%, while third-class passengers had the lowest survival rate at approximately 24.24%.

## Survival Rate by Sex and Passenger Class

The combined analysis uses boolean masking with the `&` operator.

| Sex    | Passenger Class |         Survival Rate |
| ------ | --------------: | --------------------: |
| Female |               1 | 0.967391 (**96.74%**) |
| Female |               2 | 0.921053 (**92.11%**) |
| Female |               3 | 0.500000 (**50.00%**) |
| Male   |               1 | 0.368852 (**36.89%**) |
| Male   |               2 | 0.157407 (**15.74%**) |
| Male   |               3 | 0.135447 (**13.54%**) |

Sex and passenger class together show an even clearer pattern. First-class and second-class female passengers had survival rates above 90%, whereas third-class male passengers had the lowest survival rate at approximately 13.54%.

Passenger class also mattered within each sex. Female survival declined from approximately 96.74% in first class to 50% in third class, while male survival declined from approximately 36.89% in first class to 13.54% in third class.

# Correlation Analysis

The correlation matrix is intentionally restricted to exactly the required six numeric columns:

```text
survived
pclass
age
sibsp
parch
fare
```

The derived boolean columns `adult_male` and `alone` are excluded.

The two strongest correlations were selected programmatically by ranking all unique off-diagonal feature pairs by the absolute value of their correlation coefficients.

| Rank | Feature 1 | Feature 2 |   Correlation |
| ---- | --------- | --------- | ------------: |
| 1    | `pclass`  | `fare`    | **-0.548193** |
| 2    | `sibsp`   | `parch`   |  **0.414542** |

### Strongest Correlation — `pclass` and `fare`

The correlation between `pclass` and `fare` is **-0.548193**, the largest absolute off-diagonal correlation in the matrix. Because higher `pclass` numbers represent lower passenger classes, the negative relationship indicates that passengers in higher-status classes generally paid higher fares, while passengers in third class generally paid lower fares.

### Second-Strongest Correlation — `sibsp` and `parch`

The correlation between `sibsp` and `parch` is **0.414542**. This moderate positive relationship suggests that passengers travelling with siblings or spouses were also more likely to be travelling with parents or children, which is consistent with passengers travelling as family groups.

# Multivariate Data Story

## Chart 1 — Survival Rate by Sex

The first chart shows a large difference in survival probability between female and male passengers. Female passengers had a survival rate of approximately **74.04%**, compared with only **18.89%** for male passengers.

This makes sex one of the clearest observed factors associated with survival in the dataset.

## Chart 2 — Survival Rate by Passenger Class

The second chart shows that survival probability decreased from first class to third class. First-class passengers had a survival rate of approximately **62.62%**, second-class passengers approximately **47.28%**, and third-class passengers approximately **24.24%**.

The result indicates that passenger class was strongly associated with survival.

## Chart 3 — Survival Rate by Sex and Passenger Class

Combining sex and passenger class reveals that both factors contributed to the survival pattern. First-class females had the highest observed survival rate at approximately **96.74%**, followed by second-class females at approximately **92.11%**.

Third-class males had the lowest observed survival rate at approximately **13.54%**. The chart therefore shows that the survival advantage observed for female passengers existed across classes, while passenger class also created substantial differences within each sex.

## Chart 4 — Fare Distribution by Survival Outcome

The box plot shows that the fare distribution for survivors is generally shifted toward higher values than the distribution for passengers who did not survive. Survivors also contain several very high-fare observations, including the largest fare values in the dataset.

This pattern is consistent with the passenger-class analysis: higher fares are associated with lower `pclass` numbers, and passengers in the higher-status classes also showed higher survival rates. Fare therefore contributes to the overall data story, although the plot alone does not establish a causal relationship between fare and survival.

## Overall Data Story

The exploratory analysis shows that **sex and passenger class are the clearest observed factors associated with survival**. Female passengers had much higher survival rates than male passengers, while first-class passengers were considerably more likely to survive than third-class passengers.

The combined analysis strengthens this pattern: first- and second-class female passengers had the highest survival rates, while third-class male passengers had the lowest. Fare provides supporting context because higher fares are associated with higher-status passenger classes, which also showed better survival outcomes.

# Exploratory Standardization Check

As an EDA-stage sanity check, `age` and `fare` were standardized using `StandardScaler`.

Before standardization:

| Feature |      Mean | Standard Deviation |
| ------- | --------: | -----------------: |
| Age     | 29.315152 |          12.984932 |
| Fare    | 32.096681 |          49.697504 |

After standardization:

| Feature |         Mean | Population Standard Deviation |
| ------- | -----------: | ----------------------------: |
| Age     | 2.717486e-16 |                           1.0 |
| Fare    | 1.398706e-16 |                           1.0 |

The transformed means are effectively zero, with the very small non-zero values caused by floating-point precision. Both transformed population standard deviations are exactly 1.0, confirming that the standardization was applied correctly.

This scaling is used only as an exploratory EDA check. It does **not** feed into the modeling pipeline. The modeling stage performs its own preprocessing after the train/test split so that preprocessing transformations are fitted only on the training data.

# Part B — Predictive Modeling

Part B is implemented in `02_modeling.ipynb`.

The following sections will be added after Part B is completed:

* stratified train/test split and class-balance justification
* preprocessing pipeline
* Logistic Regression results
* Decision Tree results and visualization
* Random Forest results
* classifier comparison table
* imbalance-handling comparison
* Random Forest hyperparameter tuning and OOB score
* fare regression metrics and residual interpretation
* final classifier recommendation
* complete saved-pipeline validation
