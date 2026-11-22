# Food Delivery Time Prediction

A machine-learning project that estimates food delivery time using an optimized XGBoost Regressor and a Streamlit web application.

## Project workflow

- Data cleaning and validation
- Haversine distance feature engineering
- One-hot encoding of order and vehicle categories
- Train/test split and model comparison
- XGBoost hyperparameter tuning with GridSearchCV
- Evaluation using MAE, RMSE and R-squared

## Streamlit application

Customers enter:
- Restaurant-to-customer distance (km)
- Order type
- Delivery vehicle

Delivery-person age and rating are fixed representative inputs (29 years and 4.7/5). The app discloses these assumptions.

## Project files

- `app.py` — Streamlit interface and prediction logic
- `xgb_delivery_model.json` — trained XGBoost model
- `model_config.json` — feature order, categories, defaults and metrics
- `requirements.txt` — Python dependencies

## Run locally

Create and activate a virtual environment:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies and run:

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud

1. Open https://share.streamlit.io/ and sign in with GitHub.
2. Choose **Create app**.
3. Select this repository, the `main` branch and `app.py` as the main file path.
4. Deploy and share the generated app URL.

## Model information

Selected XGBoost parameters:

`learning_rate=0.05, max_depth=3, n_estimators=200`

Held-out test metrics:
- MAE: 5.67 minutes
- RMSE: 7.21 minutes
- R²: 39.73%

Predictions are estimates, not guarantees. Actual delivery time can vary due to traffic, weather, restaurant preparation and other conditions.
