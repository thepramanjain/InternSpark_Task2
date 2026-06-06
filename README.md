# House Price Prediction Pipeline

A professional machine learning regression pipeline for predicting residential house prices. The project incorporates custom Scikit-Learn feature engineering, advanced data preprocessing, log-transformed target scaling, and comparative regression modeling to select and save the best-performing model.

---

## 📁 Repository Structure

*   `HousePricePred.ipynb`: Jupyter Notebook containing the full exploratory data analysis (EDA), custom pipeline construction, model comparison, evaluation, and saving logic.
*   `data.csv`: The raw housing dataset containing features such as bedrooms, bathrooms, square footage, locations, sale dates, and prices.
*   `house_price_model.joblib`: The serialized, production-ready inference pipeline containing the feature engineering, preprocessing, and the trained best regressor.
*   `README.md`: Project documentation and guides (this file).

---

## ⚙️ Pipeline & Feature Engineering

The project implements a custom Scikit-Learn transformer, `HouseFeatureEngineer` (inheriting from `BaseEstimator` and `TransformerMixin`), to derive high-value features from raw inputs:

### 1. Custom Feature Extraction
*   **Temporal Features**: Parses sale dates to extract `sale_year`, `sale_month`, and `sale_day`.
*   **House Age**: Computes the age of the property at the time of sale (`sale_year - yr_built`).
*   **Renovation Flags & Age**: Evaluates if the house has been renovated (`is_renovated` as a binary flag) and calculates the `renovation_age` (time since last renovation, falling back to house age if never renovated).
*   **Space Proxies**:
    *   `total_rooms_proxy`: Combined count of bedrooms and bathrooms.
    *   `living_to_lot_ratio`: Ratio of living area (`sqft_living`) to the total lot area (`sqft_lot`).

### 2. Preprocessing & Scaling
Features are routed through specialized Scikit-Learn preprocessing branches:
*   **Skewed Numerical Features** (`sqft_living`, `sqft_lot`, `sqft_above`, `sqft_basement`, `living_to_lot_ratio`): Median imputation $\rightarrow$ Log-transformation (`log1p`) to normalize skewed distributions $\rightarrow$ Standard scaling (`StandardScaler`).
*   **Regular Numerical Features** (e.g., bedrooms, floors, waterfront, view, age features): Median imputation $\rightarrow$ Standard scaling.
*   **Categorical Features** (`city`, `statezip`): Most frequent imputation $\rightarrow$ One-Hot Encoding (`OneHotEncoder` with `min_frequency=20` to reduce dimensionality and handle unknown values at inference).

### 3. Target Variable Transformation
To handle the highly skewed distribution of house prices and prevent large price variance from dominating the loss function, the target variable (`price`) is log-transformed using `TransformedTargetRegressor` (`func=np.log1p` and `inverse_func=np.expm1`).

---

## 📊 Model Comparison & Results

We compared three models using a 80-20 train-test split. The metrics evaluated are Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), and R² Score:

| Model | RMSE | MAE | R² Score |
| :--- | :--- | :--- | :--- |
| **Linear Regression** | **$993,263** | **$196,248** | **0.0326** |
| Gradient Boosting | $997,588 | $193,443 | $0.0242 |
| Random Forest | $1,004,440 | $194,978 | 0.0107 |

*Note: The overall R² scores are low due to significant noise in the target prices, including properties listed with a price of `$0` (which were kept to ensure robust preprocessing handling) and some massive multi-million dollar outliers.*

---

## 🚀 Getting Started

### Prerequisites

Ensure you have Python 3.8+ and the necessary libraries installed:

```bash
pip install numpy pandas scikit-learn joblib matplotlib seaborn
```

### Loading the Model & Making Predictions

The saved model (`house_price_model.joblib`) is a complete Scikit-Learn pipeline. When you pass raw data, it automatically applies the custom feature engineering, data preprocessing, model inference, and reverse log-transforms the prediction to the original dollar scale.

Here is a quick Python snippet demonstrating how to use the saved model:

```python
import joblib
import pandas as pd

# 1. Load the serialized model pipeline
model = joblib.load('house_price_model.joblib')

# 2. Define a sample house dictionary (with raw fields)
sample_house = pd.DataFrame([
    {
        'date': '2014-05-02 00:00:00',
        'bedrooms': 3,
        'bathrooms': 2.0,
        'sqft_living': 1800,
        'sqft_lot': 7500,
        'floors': 1.0,
        'waterfront': 0,
        'view': 0,
        'condition': 4,
        'sqft_above': 1800,
        'sqft_basement': 0,
        'yr_built': 1995,
        'yr_renovated': 0,
        'street': '123 Example St',
        'city': 'Seattle',
        'statezip': 'WA 98115',
        'country': 'USA',
    }
])

# 3. Predict the price (returns the actual price in dollars)
predicted_price = model.predict(sample_house)[0]
print(f"Predicted price: ${predicted_price:,.2f}")
# Output: Predicted price: $579,683.65
```

---

## 📈 Visualizations

After running the pipeline, two primary visualization sheets are generated:
1.  **`price_distribution.png`**: Inspects raw vs log-transformed target variables.
2.  **`residual_analysis.png`**: Inspects residual spread, residual distribution, and actual vs. predicted values for validation.
