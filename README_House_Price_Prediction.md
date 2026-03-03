# 🏠 House Price Prediction using Linear & Ridge Regression

## 📌 Project Overview

This project demonstrates a complete end-to-end Machine Learning
workflow for predicting house prices using Linear Regression and Ridge
Regression.

The goal was not just to train a model, but to build a clean and
professional ML pipeline including:

-   Proper data cleaning
-   Handling extreme outliers
-   Avoiding data leakage
-   Feature preprocessing using Pipeline & ColumnTransformer
-   Model comparison (Linear vs Ridge)
-   Alpha tuning for Ridge Regression
-   Prediction on unseen data

------------------------------------------------------------------------

## 📂 Dataset

The dataset contains approximately 4600 house records with features such
as:

-   Bedrooms
-   Bathrooms
-   Square footage (living & lot)
-   Floors
-   Waterfront
-   View
-   Condition
-   Year built / renovated
-   City
-   Price (Target Variable)

------------------------------------------------------------------------

## 🧹 Data Cleaning

The following preprocessing steps were performed:

-   Dropped unnecessary and high-cardinality columns:

    -   `date`
    -   `country`
    -   `statezip`
    -   `street`

-   Removed extreme outliers using 99th percentile filtering:

    ``` python
    upper_limit = df["price"].quantile(0.99)
    df = df[df["price"] <= upper_limit]
    ```

This stabilized the model performance significantly.

------------------------------------------------------------------------

## 🔀 Train-Test Split

Data leakage was avoided by splitting the dataset **before**
preprocessing:

``` python
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)
```

------------------------------------------------------------------------

## 🏗️ Preprocessing Pipeline

Used:

-   `SimpleImputer` (median for numeric, most_frequent for categorical)
-   `StandardScaler` (important for Ridge regularization)
-   `OneHotEncoder(handle_unknown="ignore")`
-   `ColumnTransformer`
-   `Pipeline`

This ensures clean, reproducible, production-style ML workflow.

------------------------------------------------------------------------

## 📈 Model Performance

### 🔹 Linear Regression

-   R² Score ≈ 0.65
-   RMSE ≈ 171,000

### 🔹 Ridge Regression (Best Alpha = 0.01)

-   R² Score ≈ 0.65
-   RMSE ≈ 171,000

After proper cleaning, Linear Regression performed almost as well as
Ridge, showing that the dataset became well-conditioned after outlier
handling.

------------------------------------------------------------------------

## 🎯 Key Insights

-   Linear Regression is sensitive to extreme outliers.
-   Removing top 1% price values significantly improved stability.
-   Ridge regularization helps when multicollinearity or large
    coefficients exist.
-   Proper preprocessing is more important than blindly switching
    models.
-   Avoiding data leakage is critical for trustworthy evaluation.

------------------------------------------------------------------------

## 🏠 Unseen House Prediction Example

``` python
new_house = pd.DataFrame({
    "bedrooms": [3],
    "bathrooms": [2],
    "sqft_living": [1800],
    "sqft_lot": [4000],
    "floors": [1],
    "waterfront": [0],
    "view": [0],
    "condition": [3],
    "sqft_above": [1500],
    "sqft_basement": [300],
    "yr_built": [2005],
    "yr_renovated": [0],
    "city": ["Seattle"]
})
```

Predicted Price ≈ \$455,000

------------------------------------------------------------------------

## 🚀 Tech Stack

-   Python
-   Pandas
-   NumPy
-   Scikit-Learn
-   Matplotlib

------------------------------------------------------------------------

## 🧠 What This Project Demonstrates

-   Strong understanding of regression fundamentals
-   Regularization concepts (L2 Penalty)
-   Model evaluation (R² & RMSE)
-   Clean ML pipeline architecture
-   Real-world prediction workflow

------------------------------------------------------------------------

## 📌 Conclusion

This project emphasizes building models the right way --- clean data,
proper validation, and structured experimentation.

Instead of chasing unrealistic accuracy, the focus was on building a
stable and interpretable regression pipeline.
