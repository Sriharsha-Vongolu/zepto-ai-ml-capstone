# Module 2 — Analytics Pipeline

This module implements the profiling, cleaning, exploratory data analysis, classification, imbalance handling, hyperparameter tuning, regression, and model-persistence requirements for Module 2 of the capstone project.

The Titanic dataset is loaded only once using Seaborn in `01_eda.ipynb`. Immediately after loading, the original DataFrame is saved as `titanic.csv` to provide the required committed offline fallback.

`02_modeling.ipynb` reads this same `titanic.csv` file and does not independently call `sns.load_dataset()`.

## Files

* `01_eda.ipynb` — profiling, cleaning, exploratory data analysis, visualizations, and standardization check
* `02_modeling.ipynb` — classification, imbalance handling, tuning, regression, comparison, and model persistence
* `titanic.csv` — committed offline fallback produced from the single Seaborn dataset load
* `best_pipeline.joblib` — fitted preprocessing and classification pipeline saved during Part B

## Installation

Activate the project virtual environment and install the required packages:

```powershell id="5n4atv"
.\.venv\Scripts\Activate.ps1

python -m pip install pandas numpy seaborn matplotlib scikit-learn imbalanced-learn joblib jupyter
```

## Execution Order

Run the notebooks in this order:

1. `01_eda.ipynb`
2. `02_modeling.ipynb`

`01_eda.ipynb` performs the one permitted Seaborn Titanic dataset load and creates `titanic.csv`.

`02_modeling.ipynb` continues the workflow by reading `titanic.csv`.

---

# Part A — Profiling, Cleaning, and Data Story

## Dataset Profile

The original Titanic dataset contains:

* **891 rows**
* **15 columns**

The notebook reports:

* `df.shape`
* `df.info()`
* `df.describe()`
* percentage of missing values for every affected column

## Missing-Value Analysis

The measured missing-value percentages are:

| Column        | Missing % | Cleaning Decision  |
| ------------- | --------: | ------------------ |
| `deck`        |    77.22% | Drop column        |
| `age`         |    19.87% | Median imputation  |
| `embarked`    |     0.22% | Drop affected rows |
| `embark_town` |     0.22% | Drop affected rows |

The required threshold rule was applied as follows:

* Less than 5% missing → drop affected rows.
* 5%–30% missing → impute missing values.
* Very high missingness → explicitly decide whether the column should be dropped or missingness represented as a separate category.

### `deck`

`deck` contained **77.22% missing values**. Because such a large proportion of the values was unavailable, the column was dropped rather than imputing most of its observations with artificial values.

### `age`

`age` contained **19.87% missing values**, placing it in the required 5%–30% imputation range. Since `age` is numeric, its missing values were replaced using the median.

### `embarked` and `embark_town`

Both `embarked` and `embark_town` contained approximately **0.22% missing values**. Since this is below the 5% threshold, rows containing these missing values were removed.

After cleaning, the dataset contains:

* **889 rows**
* **14 columns**
* **0 missing values** in the retained columns

---

# Univariate Analysis

Histograms and box plots were produced for both:

* `age`
* `fare`

## IQR Outlier Detection

Outliers were identified using:

```text id="fjq20q"
Lower Bound = Q1 - 1.5 × IQR
Upper Bound = Q3 + 1.5 × IQR
```

Results:

| Feature |      Q1 |      Q3 |     IQR | Lower Bound | Upper Bound | Outlier Count |
| ------- | ------: | ------: | ------: | ----------: | ----------: | ------------: |
| Age     | 22.0000 | 35.0000 | 13.0000 |      2.5000 |     54.5000 |        **65** |
| Fare    |  7.8958 | 31.0000 | 23.1042 |    -26.7605 |     65.6563 |       **114** |

`fare` contains more IQR-defined outliers than `age`, which is also visible in its box plot.

## Fare Distribution

The calculated statistics for `fare` are:

| Statistic |   Value |
| --------- | ------: |
| Mean      | 32.0967 |
| Median    | 14.4542 |
| Mode      |  8.0500 |

The ordering is:

```text id="hlumcv"
Mean > Median > Mode
32.0967 > 14.4542 > 8.0500
```

Therefore, the fare distribution is **right-skewed**. A smaller number of very high fare observations pull the mean substantially above the median and mode.

---

# Bivariate Analysis

Boolean masking was used to calculate the required survival-rate breakdowns.

## Survival by Sex

| Sex    | Survival Rate |
| ------ | ------------: |
| Female |    **74.04%** |
| Male   |    **18.89%** |

Female passengers had a much higher survival rate than male passengers. Approximately 74% of female passengers survived compared with approximately 19% of male passengers, showing a strong association between sex and survival outcome.

## Survival by Passenger Class

| Passenger Class | Survival Rate |
| --------------- | ------------: |
| 1               |    **62.62%** |
| 2               |    **47.28%** |
| 3               |    **24.24%** |

Survival declined consistently from first class to third class. First-class passengers had the highest survival rate, while third-class passengers had the lowest.

## Survival by Sex and Passenger Class

| Sex    | Passenger Class | Survival Rate |
| ------ | --------------: | ------------: |
| Female |               1 |    **96.74%** |
| Female |               2 |    **92.11%** |
| Female |               3 |    **50.00%** |
| Male   |               1 |    **36.89%** |
| Male   |               2 |    **15.74%** |
| Male   |               3 |    **13.54%** |

The combined analysis shows that both sex and passenger class were strongly associated with survival. First-class females had the highest observed survival rate at approximately 96.74%, while third-class males had the lowest rate at approximately 13.54%.

Passenger class also mattered within each sex. Female survival declined from approximately 96.74% in first class to 50% in third class, while male survival declined from approximately 36.89% in first class to approximately 13.54% in third class.

---

# Correlation Analysis

The correlation matrix is restricted to exactly the following six required columns:

```text id="5xiflu"
survived
pclass
age
sibsp
parch
fare
```

The derived boolean columns `adult_male` and `alone` are intentionally excluded.

All unique off-diagonal feature pairs were ranked by their absolute correlation coefficients.

The two strongest correlations were:

| Rank | Feature 1 | Feature 2 |   Correlation |
| ---- | --------- | --------- | ------------: |
| 1    | `pclass`  | `fare`    | **-0.548193** |
| 2    | `sibsp`   | `parch`   |  **0.414542** |

## `pclass` and `fare`

The correlation between `pclass` and `fare` is **-0.548193**, which is the strongest absolute off-diagonal relationship in the matrix.

Because lower numerical `pclass` values represent higher passenger classes, the negative correlation shows that higher-class passengers generally paid higher fares, while third-class passengers generally paid lower fares.

## `sibsp` and `parch`

The correlation between `sibsp` and `parch` is **0.414542**.

This positive relationship indicates that passengers travelling with siblings or spouses also tended to be more likely to travel with parents or children, consistent with passengers travelling in family groups.

---

# Multivariate Data Story

## Chart 1 — Survival Rate by Sex

Female passengers had a survival rate of approximately **74.04%**, while male passengers had a survival rate of approximately **18.89%**.

The large difference makes sex one of the clearest observed factors associated with survival in this dataset.

## Chart 2 — Survival Rate by Passenger Class

First-class passengers had approximately **62.62%** survival, second-class passengers approximately **47.28%**, and third-class passengers approximately **24.24%**.

The chart shows a clear decrease in survival probability from first class to third class, indicating a strong association between passenger class and survival.

## Chart 3 — Survival Rate by Sex and Passenger Class

The combined chart shows that both sex and passenger class contributed to the survival pattern.

First-class females had the highest observed survival rate at approximately **96.74%**, while third-class males had the lowest observed survival rate at approximately **13.54%**. Female passengers retained a survival advantage within every passenger class, while class also created substantial differences within each sex.

## Chart 4 — Fare Distribution by Survival Outcome

The fare distribution for survivors is generally shifted toward higher values than the distribution for passengers who did not survive.

This is consistent with the passenger-class findings because higher fares are associated with higher-status passenger classes, which also had higher survival rates. This relationship is observational and does not by itself establish that paying a higher fare caused survival.

## Overall Data Story

The exploratory analysis indicates that **sex and passenger class are the clearest observed factors associated with survival**.

Female passengers survived at substantially higher rates than male passengers, while first-class passengers had considerably higher survival rates than third-class passengers. The combined sex-and-class analysis strengthens this pattern, with first- and second-class females having the highest survival rates and third-class males having the lowest.

Fare provides additional context because higher fares are associated with higher-status passenger classes, which also showed better survival outcomes.

---

# Exploratory Standardization Check

As an EDA-only sanity check, `age` and `fare` were standardized using `StandardScaler`.

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

The resulting means are effectively zero, with the tiny remaining values caused by floating-point precision. Both population standard deviations equal 1.0, confirming that standardization was applied correctly.

This scaling was performed only as the required exploratory EDA check and is not passed into the classification pipeline.

---

# Part B — Predictive Modeling

## Classification Features and Target

The classification target is:

```text id="m5mvvy"
survived
```

The following predictor columns are used:

```text id="fcu7zs"
pclass
age
sibsp
parch
fare
sex
embarked
```

## Class Balance

The original target distribution is:

| Class              | Count | Percentage |
| ------------------ | ----: | ---------: |
| Not Survived (`0`) |   549 | **61.62%** |
| Survived (`1`)     |   342 | **38.38%** |

The dataset therefore contains more non-survivors than survivors.

## Stratified Train/Test Split

The data was split before any model preprocessing using:

* **80% training data**
* **20% test data**
* `random_state=42`
* `stratify=y`

This produced:

| Dataset  | Rows |
| -------- | ---: |
| Training |  712 |
| Test     |  179 |

The class proportions were preserved:

| Class              | Train % | Test % |
| ------------------ | ------: | -----: |
| Not Survived (`0`) |  61.66% | 61.45% |
| Survived (`1`)     |  38.34% | 38.55% |

Stratification was used because the target is not perfectly balanced. Preserving approximately the same class proportions in the training and test sets prevents the evaluation split from accidentally containing a materially different class distribution from the training data.

---

# Classification Preprocessing

All modeling preprocessing is implemented inside a `ColumnTransformer` and complete model `Pipeline`.

## Numeric Features

```text id="i7qwa4"
pclass
age
sibsp
parch
fare
```

Numeric preprocessing consists of:

1. Median imputation
2. `StandardScaler`

## Categorical Features

```text id="gki9uw"
sex
embarked
```

Categorical preprocessing consists of:

1. Most-frequent-value imputation
2. One-hot encoding

The train/test split occurs before preprocessing. The complete pipelines are fitted using `X_train` only, so the imputer, encoder, and scaler learn their parameters only from the training data.

The test data is subsequently passed through the already-fitted pipeline in transform/predict mode, avoiding test-data leakage.

---

# Classification Models

Three classifiers were trained on the identical train/test split:

* Logistic Regression
* Decision Tree
* Random Forest

The Decision Tree was also visualized using `plot_tree` with transformed feature names and class names.

## Classifier Comparison

| Model               |   Accuracy |  Precision |     Recall |         F1 |        AUC |
| ------------------- | ---------: | ---------: | ---------: | ---------: | ---------: |
| Logistic Regression | **0.8045** | **0.7931** | **0.6667** | **0.7244** | **0.8437** |
| Decision Tree       | **0.7654** | **0.7547** | **0.5797** | **0.6557** | **0.7971** |
| Random Forest       | **0.8156** | **0.8000** | **0.6957** | **0.7442** | **0.8287** |

Confusion matrices were generated for all three classifiers, and ROC curves were plotted using predicted survival probabilities.

### Logistic Regression

Logistic Regression achieved an accuracy of **0.8045**, precision of **0.7931**, recall of **0.6667**, F1 score of **0.7244**, and AUC of **0.8437**.

It achieved the highest AUC of the three primary classifiers, indicating strong ranking ability across classification thresholds.

### Decision Tree

The Decision Tree achieved an accuracy of **0.7654**, precision of **0.7547**, recall of **0.5797**, F1 score of **0.6557**, and AUC of **0.7971**.

It produced the weakest overall test-set metrics among the three classifiers.

### Random Forest

Random Forest achieved an accuracy of **0.8156**, precision of **0.8000**, recall of **0.6957**, F1 score of **0.7442**, and AUC of **0.8287**.

It achieved the highest accuracy, precision, and F1 score among the three primary classifiers, although Logistic Regression produced a slightly higher AUC.

---

# Imbalance Handling Comparison

The imbalance-handling experiment used Logistic Regression as the common classifier so that only the handling strategy changed.

Three approaches were compared:

1. No imbalance handling
2. `class_weight="balanced"`
3. SMOTE oversampling

SMOTE was included inside an `imblearn` pipeline after preprocessing and was therefore applied during training only. The test set was never oversampled.

Results:

| Strategy              |  Precision |     Recall |         F1 |
| --------------------- | ---------: | ---------: | ---------: |
| Baseline              | **0.7931** | **0.6667** | **0.7244** |
| Class Weight Balanced | **0.7297** | **0.7826** | **0.7552** |
| SMOTE                 | **0.7397** | **0.7826** | **0.7606** |

## Imbalance Conclusion

The baseline model produced the highest precision at **0.7931**, but its recall was lower at **0.6667**.

Both class weighting and SMOTE increased recall to **0.7826**, demonstrating that the imbalance-handling approaches detected a larger proportion of surviving passengers. SMOTE produced a slightly better precision than class weighting, **0.7397 versus 0.7297**, which resulted in the highest F1 score of the three imbalance variants at **0.7606**.

Therefore, **SMOTE performed best for this imbalance-handling comparison when F1 score is used as the selection criterion**, because it provided the strongest balance between precision and recall.

---

# Random Forest Hyperparameter Tuning

`GridSearchCV` was performed on the Random Forest pipeline using F1 score with 5-fold cross-validation.

The following Random Forest hyperparameters were tuned:

* `n_estimators`
* `max_depth`
* `max_features`

The estimator was constructed with:

```python id="r4wsnr"
RandomForestClassifier(
    oob_score=True,
    bootstrap=True,
    random_state=42
)
```

The best parameter combination was:

| Hyperparameter | Best Value |
| -------------- | ---------- |
| `max_depth`    | **5**      |
| `max_features` | **sqrt**   |
| `n_estimators` | **100**    |

The corresponding out-of-bag score was:

```text id="7rin07"
OOB score = 0.8272
```

This confirms that the tuned estimator successfully generated the required OOB evaluation.

---

# Regression Side-Task — Predicting Fare

A multivariate Linear Regression model was trained to predict `fare`.

The predictor columns were:

```text id="0ug73p"
pclass
age
sibsp
parch
survived
sex
embarked
```

Numeric predictors were median-imputed and standardized, while categorical predictors were most-frequent-imputed and one-hot encoded.

## Regression Metrics

| Model                          |         MAE |        RMSE |         R² | Adjusted R² |
| ------------------------------ | ----------: | ----------: | ---------: | ----------: |
| Multivariate Linear Regression | **20.8977** | **30.5328** | **0.3975** |  **0.3617** |

The model's MAE of **20.8977** means that the absolute prediction error averages approximately 20.90 fare units on the test data.

The RMSE is higher at **30.5328**, indicating that some larger prediction errors increase the squared-error metric. The model explains approximately **39.75%** of the observed test-set variation in fare according to R², while the Adjusted R² is approximately **36.17%** after accounting for the number of transformed predictors.

## Residual Analysis

The residual plot does **show evidence of heteroscedasticity**.

The residuals do not maintain a constant random spread around zero. Their dispersion becomes noticeably larger as predicted fares increase, and there is visible structure in the residual pattern, including several large positive and negative residuals at higher predicted fares.

Therefore, the residual variance is not constant across the prediction range, which indicates heteroscedasticity in the Linear Regression model.

---

# Final Model Comparison

Classification and regression metrics measure different types of performance and are therefore presented as separate metric groups.

## Classification Metrics

| Model               | Accuracy | Precision | Recall |     F1 |    AUC |
| ------------------- | -------: | --------: | -----: | -----: | -----: |
| Logistic Regression |   0.8045 |    0.7931 | 0.6667 | 0.7244 | 0.8437 |
| Decision Tree       |   0.7654 |    0.7547 | 0.5797 | 0.6557 | 0.7971 |
| Random Forest       |   0.8156 |    0.8000 | 0.6957 | 0.7442 | 0.8287 |

## Regression Metrics

| Model             |     MAE |    RMSE |     R² | Adjusted R² |
| ----------------- | ------: | ------: | -----: | ----------: |
| Linear Regression | 20.8977 | 30.5328 | 0.3975 |      0.3617 |

The classification metrics and regression metrics are not directly comparable because they measure different prediction tasks.

---

# Final Classifier Recommendation

Among the three primary classifiers, I would deploy the **Random Forest** because it achieved the strongest overall balance of the evaluated test-set classification metrics. It produced the highest accuracy at **0.8156**, the highest precision at **0.8000**, and the highest F1 score at **0.7442**, while maintaining a recall of **0.6957**. Logistic Regression produced a slightly higher AUC of **0.8437** compared with the Random Forest's **0.8287**, but its accuracy and F1 score were lower at **0.8045** and **0.7244** respectively. Based on the combination of accuracy, precision, recall, F1, and AUC rather than a single metric, Random Forest provides the preferred result among the three required classifiers.

---

# Saved Complete Pipeline

For the primary three-classifier comparison, F1 score was used to identify the best-performing classifier.

Random Forest produced the highest F1 score among those three models:

```text id="3gg7i5"
Random Forest F1 = 0.7442
```

The fitted Random Forest pipeline, including its preprocessing steps and final classifier, is saved using:

```python id="tm7pa9"
joblib.dump(best_pipeline, BEST_PIPELINE_PATH)
```

to:

```text id="r55gh0"
analytics/best_pipeline.joblib
```

The saved object contains the preprocessing and estimator together, allowing it to receive raw, unprocessed feature rows.

## Reload Validation

The saved pipeline is reloaded using:

```python id="ka8bd7"
loaded_pipeline = joblib.load(BEST_PIPELINE_PATH)
```

A raw test observation was then passed to both the original fitted pipeline and the reloaded pipeline.

The resulting predictions were:

```text id="xn3o35"
Original pipeline prediction: [0]
Reloaded pipeline prediction: [0]
Predictions match: True
```

The matching predictions confirm that the persisted complete pipeline can be reloaded and used end-to-end on raw input data.
