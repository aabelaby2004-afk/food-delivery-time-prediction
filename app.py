import json
from html import escape
from pathlib import Path

import pandas as pd
import streamlit as st
from xgboost import XGBRegressor


# -------------------- Page configuration --------------------
st.set_page_config(
    page_title="Food Delivery Time Estimator",
    page_icon="🛵",
    layout="wide",
    initial_sidebar_state="collapsed",
)

BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "xgb_delivery_model.json"
CONFIG_PATH = BASE_DIR / "model_config.json"


# -------------------- Custom visual theme --------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&display=swap');

    :root {
        --ink: #172033;
        --muted: #64748b;
        --line: #e5eaf2;
        --blue: #2457e6;
        --blue-dark: #173a9c;
        --orange: #ff9b54;
        --paper: #f4f7fb;
    }

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    [data-testid="stAppViewContainer"] {
        background:
            radial-gradient(circle at 8% 4%, rgba(36, 87, 230, 0.08), transparent 27rem),
            radial-gradient(circle at 95% 15%, rgba(255, 155, 84, 0.10), transparent 24rem),
            var(--paper);
    }

    [data-testid="stHeader"] {
        background: transparent;
    }

    [data-testid="stMainBlockContainer"] {
        max-width: 1160px;
        padding-top: 2.2rem;
        padding-bottom: 3rem;
    }

    h1, h2, h3, p, label {
        color: var(--ink);
    }

    .hero {
        position: relative;
        overflow: hidden;
        padding: 2.4rem 2.5rem 2.3rem;
        border-radius: 28px;
        color: #ffffff;
        background: linear-gradient(118deg, #142b59 0%, #1c4da8 62%, #2869d8 100%);
        box-shadow: 0 18px 45px rgba(23, 58, 156, 0.18);
        margin-bottom: 1.6rem;
    }

    .hero:after {
        content: "🛵";
        position: absolute;
        right: 5%;
        top: 4%;
        font-size: 7.5rem;
        opacity: 0.15;
        transform: rotate(-12deg);
    }

    .hero-kicker {
        display: inline-block;
        padding: 0.35rem 0.72rem;
        border: 1px solid rgba(255,255,255,0.24);
        border-radius: 999px;
        color: #dbeafe;
        background: rgba(255,255,255,0.10);
        font-size: 0.72rem;
        font-weight: 800;
        letter-spacing: 0.12em;
        text-transform: uppercase;
    }

    .hero h1 {
        position: relative;
        z-index: 1;
        color: #ffffff !important;
        font-size: clamp(2.1rem, 4.5vw, 3.35rem);
        line-height: 1.08;
        letter-spacing: -0.045em;
        margin: 1.05rem 0 0.6rem;
        max-width: 720px;
    }

    .hero h1 span {
        color: #ffbd86;
    }

    .hero p {
        position: relative;
        z-index: 1;
        color: #dbeafe !important;
        font-size: 1.04rem;
        max-width: 650px;
        margin: 0;
        line-height: 1.65;
    }

    .hero-bottom {
        display: flex;
        flex-wrap: wrap;
        gap: 0.55rem;
        margin-top: 1.35rem;
    }

    .hero-pill {
        color: #eef5ff;
        background: rgba(255,255,255,0.12);
        border: 1px solid rgba(255,255,255,0.16);
        padding: 0.42rem 0.75rem;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 600;
    }

    .eyebrow {
        color: var(--blue);
        font-size: 0.72rem;
        font-weight: 800;
        letter-spacing: 0.13em;
        text-transform: uppercase;
        margin-bottom: 0.35rem;
    }

    .section-heading {
        color: var(--ink);
        font-size: 1.55rem;
        font-weight: 800;
        letter-spacing: -0.035em;
        margin: 0 0 0.25rem;
    }

    .section-subtitle {
        color: var(--muted);
        font-size: 0.92rem;
        margin: 0 0 1.15rem;
    }

    [data-testid="stForm"] {
        background: #ffffff;
        border: 1px solid var(--line);
        border-radius: 22px;
        padding: 1.35rem 1.45rem 1.15rem;
        box-shadow: 0 10px 30px rgba(24, 39, 75, 0.055);
    }

    [data-testid="stWidgetLabel"] p {
        color: #26334a !important;
        font-weight: 700 !important;
        font-size: 0.88rem !important;
    }

    [data-testid="stNumberInput"] input,
    [data-testid="stSelectbox"] div[data-baseweb="select"] > div {
        border-radius: 12px;
    }

    div.stButton > button,
    [data-testid="stFormSubmitButton"] > button {
        border: 0;
        border-radius: 12px;
        min-height: 3.15rem;
        font-weight: 800;
        letter-spacing: 0.01em;
        background: linear-gradient(100deg, #2457e6, #3677f6);
        box-shadow: 0 8px 18px rgba(36, 87, 230, 0.18);
        transition: transform 120ms ease, box-shadow 120ms ease;
    }

    div.stButton > button:hover,
    [data-testid="stFormSubmitButton"] > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 11px 22px rgba(36, 87, 230, 0.24);
    }

    .info-card {
        height: 100%;
        background: #ffffff;
        border: 1px solid var(--line);
        border-radius: 22px;
        padding: 1.45rem;
        box-shadow: 0 10px 30px rgba(24, 39, 75, 0.055);
    }

    .info-card h3 {
        color: var(--ink);
        font-size: 1.13rem;
        margin: 0.15rem 0 0.45rem;
        letter-spacing: -0.02em;
    }

    .info-card p {
        color: var(--muted);
        font-size: 0.88rem;
        line-height: 1.55;
    }

    .fixed-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 0.7rem;
        margin: 1rem 0;
    }

    .fixed-item {
        background: #f4f7ff;
        border: 1px solid #e4ebff;
        border-radius: 14px;
        padding: 0.85rem;
    }

    .fixed-label {
        color: #64748b;
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }

    .fixed-value {
        color: #173a9c;
        font-size: 1.32rem;
        font-weight: 800;
        margin-top: 0.15rem;
    }

    .note-box {
        background: #fff7ed;
        border: 1px solid #fed7aa;
        border-radius: 13px;
        padding: 0.85rem 0.95rem;
        color: #9a3412;
        font-size: 0.81rem;
        line-height: 1.5;
    }

    .result-card {
        background: linear-gradient(120deg, #102d63 0%, #1d4fa8 70%, #2869d8 100%);
        border-radius: 24px;
        padding: 1.55rem 1.7rem;
        color: #ffffff;
        box-shadow: 0 16px 35px rgba(23, 58, 156, 0.17);
        margin: 1.5rem 0 1rem;
    }

    .result-kicker {
        color: #cfe0ff;
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.13em;
        font-weight: 800;
    }

    .result-number {
        color: #ffffff;
        font-size: clamp(2.7rem, 6vw, 4.2rem);
        font-weight: 800;
        line-height: 1.12;
        letter-spacing: -0.055em;
        margin: 0.45rem 0 0.3rem;
    }

    .result-number span {
        color: #ffbd86;
        font-size: 1.25rem;
        letter-spacing: 0;
        font-weight: 700;
    }

    .result-caption {
        color: #dbeafe;
        font-size: 0.88rem;
    }

    .result-tags {
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
        margin-top: 1rem;
    }

    .result-tag {
        color: #ffffff;
        background: rgba(255,255,255,0.13);
        border: 1px solid rgba(255,255,255,0.18);
        border-radius: 999px;
        padding: 0.4rem 0.72rem;
        font-size: 0.78rem;
        font-weight: 600;
    }

    .empty-result {
        border: 1px dashed #cbd5e1;
        border-radius: 20px;
        padding: 1.2rem;
        background: rgba(255,255,255,0.62);
        color: #64748b;
        text-align: center;
        margin: 1.4rem 0 1rem;
    }

    .metric-label {
        color: var(--muted);
        font-size: 0.85rem;
    }

    .footer {
        border-top: 1px solid #e1e7f0;
        margin-top: 2rem;
        padding-top: 1rem;
        color: #8290a5;
        font-size: 0.78rem;
        text-align: center;
    }

    [data-testid="stExpander"] {
        background: rgba(255,255,255,0.78);
        border: 1px solid var(--line);
        border-radius: 15px;
    }

    @media (max-width: 700px) {
        .hero {
            padding: 1.6rem 1.35rem;
            border-radius: 22px;
        }
        .hero:after {
            font-size: 4.5rem;
            right: 2%;
        }
        [data-testid="stMainBlockContainer"] {
            padding-top: 1rem;
        }
        [data-testid="stForm"] {
            padding: 1rem;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# -------------------- Load trained model --------------------
@st.cache_resource
def load_model():
    with open(CONFIG_PATH, "r", encoding="utf-8") as file:
        model_config = json.load(file)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "xgb_delivery_model.json was not found beside app.py."
        )

    model = XGBRegressor()
    model.load_model(str(MODEL_PATH))
    return model, model_config


try:
    model, config = load_model()
except Exception as error:
    st.error("The prediction model could not be loaded.")
    st.exception(error)
    st.stop()


# -------------------- Header --------------------
st.markdown(
    """
    <div class="hero">
        <div class="hero-kicker">Smart delivery · Powered by XGBoost</div>
        <h1>Your delivery,<br><span>timed better.</span></h1>
        <p>
            Get a quick delivery-time estimate. Choose your distance, order type
            and delivery vehicle — we'll do the prediction.
        </p>
        <div class="hero-bottom">
            <span class="hero-pill">⚡ Instant estimate</span>
            <span class="hero-pill">📍 Distance-based</span>
            <span class="hero-pill">🤖 Machine learning</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# -------------------- Fixed model inputs --------------------
defaults = config["defaults"]
distance_min, distance_max = config["training_distance_range_km"]


# -------------------- Customer form and explainer --------------------
left, right = st.columns([1.18, 0.82], gap="large")

with left:
    st.markdown('<div class="eyebrow">01 · Delivery details</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-heading">Build your estimate</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-subtitle">Tell us a little about this delivery.</div>',
        unsafe_allow_html=True,
    )

    with st.form("delivery_prediction_form"):
        distance = st.number_input(
            "Restaurant to customer distance (km)",
            min_value=0.1,
            max_value=100.0,
            value=5.0,
            step=0.1,
            format="%.1f",
            help="Enter the delivery distance in kilometres.",
        )

        form_col1, form_col2 = st.columns(2)
        with form_col1:
            order_type = st.selectbox(
                "Order type",
                options=config["order_categories"],
                index=config["order_categories"].index(defaults["order_type"]),
            )

        with form_col2:
            vehicle_type = st.selectbox(
                "Delivery vehicle",
                options=config["vehicle_categories"],
                index=config["vehicle_categories"].index(defaults["vehicle_type"]),
                format_func=lambda value: value.replace("_", " ").title(),
            )

        submitted = st.form_submit_button(
            "✦  Estimate delivery time",
            type="primary",
            use_container_width=True,
        )


with right:
    st.markdown(
        f"""
        <div class="info-card">
            <div class="eyebrow">02 · Behind the estimate</div>
            <h3>A little context goes a long way.</h3>
            <p>
                The model uses five features. You provide three of them;
                two delivery-person details are held at fixed representative values.
            </p>
            <div class="fixed-grid">
                <div class="fixed-item">
                    <div class="fixed-label">Age · fixed</div>
                    <div class="fixed-value">{defaults['delivery_person_age']:.0f} yrs</div>
                </div>
                <div class="fixed-item">
                    <div class="fixed-label">Rating · fixed</div>
                    <div class="fixed-value">{defaults['delivery_person_rating']:.1f}/5</div>
                </div>
            </div>
            <div class="note-box">
                <strong>Good to know</strong><br>
                Age and rating are assumptions, not details about the actual
                delivery person. The prediction is an estimate, not a guarantee.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# -------------------- Prediction --------------------
if submitted:
    input_row = pd.DataFrame(
        {
            "Delivery_person_Age": [defaults["delivery_person_age"]],
            "Delivery_person_Ratings": [defaults["delivery_person_rating"]],
            "distance_km": [distance],
            "Type_of_order": pd.Categorical(
                [order_type],
                categories=config["order_categories"],
            ),
            "Type_of_vehicle": pd.Categorical(
                [vehicle_type],
                categories=config["vehicle_categories"],
            ),
        }
    )

    encoded_values = pd.get_dummies(
        input_row[["Type_of_order", "Type_of_vehicle"]],
        drop_first=True,
        dtype=int,
    )

    prediction_input = pd.concat(
        [
            input_row.drop(["Type_of_order", "Type_of_vehicle"], axis=1),
            encoded_values,
        ],
        axis=1,
    )

    prediction_input = prediction_input.reindex(
        columns=config["feature_columns"],
        fill_value=0,
    )

    predicted_minutes = max(float(model.predict(prediction_input)[0]), 0.0)

    st.session_state["last_prediction"] = {
        "minutes": predicted_minutes,
        "distance": distance,
        "order_type": order_type,
        "vehicle_type": vehicle_type,
    }


# -------------------- Prediction result --------------------
st.markdown('<div class="eyebrow">03 · Your result</div>', unsafe_allow_html=True)

if "last_prediction" in st.session_state:
    result = st.session_state["last_prediction"]
    minutes = result["minutes"]
    result_order = escape(result["order_type"])
    result_vehicle = escape(result["vehicle_type"].replace("_", " ").title())

    st.markdown(
        f"""
        <div class="result-card">
            <div class="result-kicker">Estimated delivery time</div>
            <div class="result-number">{minutes:.1f} <span>minutes</span></div>
            <div class="result-caption">Your estimated arrival time, based on the details entered.</div>
            <div class="result-tags">
                <span class="result-tag">📍 {result['distance']:.1f} km</span>
                <span class="result-tag">🍱 {result_order}</span>
                <span class="result-tag">🛵 {result_vehicle}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if result["distance"] < distance_min or result["distance"] > distance_max:
        st.warning(
            f"This distance is outside the range represented in the training data "
            f"({distance_min:.1f}–{distance_max:.1f} km). The estimate may be less reliable."
        )
else:
    st.markdown(
        """
        <div class="empty-result">
            <div style="font-size:1.8rem; margin-bottom:0.35rem;">⏱️</div>
            <strong>Your estimate will appear here.</strong><br>
            <span style="font-size:0.85rem;">Enter the delivery details above and select “Estimate delivery time”.</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


# -------------------- Model information --------------------
with st.expander("📊  Model performance and technical details"):
    perf = config["metrics"]
    metric1, metric2, metric3 = st.columns(3)
    metric1.metric("Mean Absolute Error", f"{perf['mae_minutes']:.2f} min")
    metric2.metric("Root Mean Squared Error", f"{perf['rmse_minutes']:.2f} min")
    metric3.metric("Test R²", f"{perf['r2_percent']:.2f}%")

    st.caption(
        "These metrics are from the held-out test set. MAE and RMSE are measured "
        "in minutes; R² describes variance explained on that test split."
    )
    st.markdown("**Selected XGBoost parameters**")
    st.code(
        f"n_estimators = {config['best_params']['n_estimators']}\n"
        f"max_depth = {config['best_params']['max_depth']}\n"
        f"learning_rate = {config['best_params']['learning_rate']}",
        language="python",
    )

with st.expander("ℹ️  How the estimate is calculated"):
    st.write(
        "The application sends distance, order type and vehicle type to the saved "
        "XGBoost model. It applies the same one-hot encoding and feature order used "
        "during training. Delivery-person age and rating remain fixed at the values "
        "shown above."
    )
    st.caption(
        "Real delivery times can also be affected by traffic, weather, restaurant "
        "preparation and other factors not included in this model."
    )


# -------------------- Footer --------------------
st.markdown(
    """
    <div class="footer">
        FOOD DELIVERY TIME ESTIMATOR &nbsp;·&nbsp; XGBOOST REGRESSION<br>
        Built with Python and Streamlit
    </div>
    """,
    unsafe_allow_html=True,
)
