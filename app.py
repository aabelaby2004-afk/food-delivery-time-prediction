import json
from pathlib import Path

import pandas as pd
import streamlit as st
from xgboost import XGBRegressor


# -------------------- Page configuration --------------------
st.set_page_config(
    page_title="Food Delivery Time Estimator",
    page_icon="🛵",
    layout="centered"
)

BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "xgb_delivery_model.json"
CONFIG_PATH = BASE_DIR / "model_config.json"


# -------------------- Load trained model --------------------
@st.cache_resource
def load_model():
    with open(CONFIG_PATH, "r", encoding="utf-8") as file:
        config = json.load(file)

    model = XGBRegressor()
    model.load_model(str(MODEL_PATH))
    return model, config


try:
    model, config = load_model()
except Exception as error:
    st.error(
        "The prediction model could not be loaded. Make sure "
        "`xgb_delivery_model.json` and `model_config.json` are in the same "
        "folder as `app.py`."
    )
    st.exception(error)
    st.stop()


# -------------------- Header --------------------
st.title("🛵 Food Delivery Time Estimator")
st.write(
    "Enter the delivery distance, order type and vehicle type to estimate "
    "the delivery time."
)
st.divider()


# -------------------- Fixed model inputs --------------------
defaults = config["defaults"]
distance_min, distance_max = config["training_distance_range_km"]

with st.expander("How is the estimate calculated?"):
    st.write(
        "The XGBoost model uses five inputs. You can enter the delivery distance, "
        "order type and vehicle type below. To keep the customer form simple, "
        "delivery-person age and rating are held at fixed representative values:"
    )
    st.markdown(
        f"""
        - Delivery person's age: **{defaults['delivery_person_age']:.0f} years (fixed)**
        - Delivery person's rating: **{defaults['delivery_person_rating']:.1f}/5 (fixed)**
        """
    )
    st.caption(
        "The fixed age and rating are assumptions, not details about the actual "
        "delivery person. The estimate is not a guarantee of actual delivery time."
    )


# -------------------- Customer inputs --------------------
st.subheader("Enter delivery details")

with st.form("delivery_prediction_form"):
    distance = st.number_input(
        "Restaurant to customer distance (km)",
        min_value=0.1,
        max_value=100.0,
        value=5.0,
        step=0.1,
        format="%.1f",
        help="Enter the delivery distance in kilometres."
    )

    order_type = st.selectbox(
        "Order type",
        options=config["order_categories"],
        index=config["order_categories"].index(defaults["order_type"])
    )

    vehicle_type = st.selectbox(
        "Delivery vehicle",
        options=config["vehicle_categories"],
        index=config["vehicle_categories"].index(defaults["vehicle_type"]),
        format_func=lambda value: value.replace("_", " ").title()
    )

    submitted = st.form_submit_button(
        "Estimate delivery time",
        type="primary",
        use_container_width=True
    )


# -------------------- Make prediction --------------------
if submitted:
    # Age and rating remain fixed; the customer selects the other inputs.
    input_row = pd.DataFrame(
        {
            "Delivery_person_Age": [defaults["delivery_person_age"]],
            "Delivery_person_Ratings": [defaults["delivery_person_rating"]],
            "distance_km": [distance],
            "Type_of_order": pd.Categorical(
                [order_type],
                categories=config["order_categories"]
            ),
            "Type_of_vehicle": pd.Categorical(
                [vehicle_type],
                categories=config["vehicle_categories"]
            )
        }
    )

    # Apply the same one-hot encoding categories used during model training.
    encoded_values = pd.get_dummies(
        input_row[["Type_of_order", "Type_of_vehicle"]],
        drop_first=True,
        dtype=int
    )

    prediction_input = pd.concat(
        [
            input_row.drop(["Type_of_order", "Type_of_vehicle"], axis=1),
            encoded_values
        ],
        axis=1
    )

    # Ensure the model receives features in exactly the training order.
    prediction_input = prediction_input.reindex(
        columns=config["feature_columns"],
        fill_value=0
    )

    predicted_minutes = float(model.predict(prediction_input)[0])
    predicted_minutes = max(predicted_minutes, 0.0)

    st.divider()
    st.success("Your delivery time estimate is ready.")
    st.metric(
        label="Estimated delivery time",
        value=f"{predicted_minutes:.1f} minutes"
    )

    if distance < distance_min or distance > distance_max:
        st.warning(
            f"This distance is outside the range represented in the training data "
            f"({distance_min:.1f}–{distance_max:.1f} km). The estimate may be less reliable."
        )

    st.caption(
        f"Distance: {distance:.1f} km · Order: {order_type} · "
        f"Vehicle: {vehicle_type.replace('_', ' ')}"
    )
    st.caption(
        "Age and rating are fixed at the values shown in the explanation above. "
        "Actual delivery time may vary due to traffic, weather, restaurant preparation "
        "and other conditions."
    )


# -------------------- Model performance --------------------
with st.expander("View model performance"):
    perf = config["metrics"]
    col1, col2, col3 = st.columns(3)
    col1.metric("MAE", f"{perf['mae_minutes']:.2f} min")
    col2.metric("RMSE", f"{perf['rmse_minutes']:.2f} min")
    col3.metric("R²", f"{perf['r2_percent']:.2f}%")
    st.caption(
        "Metrics are from the held-out test set before the final model was refitted "
        "on all available cleaned training records."
    )
    st.write("Selected XGBoost parameters:", config["best_params"])


st.divider()
st.caption("Food Delivery Time Estimator · XGBoost Regression")
