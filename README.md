# Food Delivery Time Prediction

A machine-learning project that estimates food delivery time using an optimized XGBoost Regressor and a Streamlit web application.

## Project overview

The model was developed using a structured ML workflow:
- Data cleaning and validation
- Haversine distance feature engineering
- One-hot encoding of order and vehicle categories
- Train/test split and model comparison
- XGBoost hyperparameter tuning with GridSearchCV
- Evaluation using MAE, RMSE and R-squared

## Streamlit application

The customer-facing app accepts:
- Delivery distance (km)
- Order type
- Delivery vehicle

Delivery-person age and rating are fixed representative inputs (29 years and 4.7 rating) as disclosed in the app.

## Run locally

```bash
python -m venv .venv
```

Activate on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Run the app:

```bash
streamlit run app.py
```

## Project files

- `app.py` — Streamlit application
- `xgb_delivery_model.json` — trained XGBoost model
- `model_config.json` — feature order, categories, defaults and metrics
- `requirements.txt` — Python dependencies

## Deployment

Deploy through [Streamlit Community Cloud](https://share.streamlit.io/) by selecting this GitHub repository, the `main` branch and `app.py` as the app entry point.

## Notes

Predictions are estimates, not guarantees. Actual delivery time can vary with traffic, weather, preparation time and other conditions.
