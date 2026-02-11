import os
import json
import joblib
import pandas as pd
import streamlit as st


st.set_page_config(page_title="Rent Predictor", layout="centered")

BASE = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE, "model")
PIPE_PATH = os.path.join(MODEL_DIR, "pipeline.pkl")
CITIES_PATH = os.path.join(MODEL_DIR, "cities.json")


def load_model():
    if os.path.exists(PIPE_PATH):
        return joblib.load(PIPE_PATH)
    return None

def load_cities():
    if os.path.exists(CITIES_PATH):
        with open(CITIES_PATH, "r") as f:
            return json.load(f)
    # fallback: a few common cities
    return ["Kolkata", "Mumbai", "Delhi", "Bengaluru", "Chennai"]


pipeline = load_model()
cities = load_cities()

st.markdown("""
<style>
  .stApp { background: linear-gradient(180deg,#f8fafc,#ffffff); }
  .big-number { font-size:28px; font-weight:700; }
</style>
""", unsafe_allow_html=True)

st.title("Modern Rent Predictor")
st.caption("Predict monthly rent from BHK, Size (sqft) and City — linear regression model")

with st.form("input_form"):
    col1, col2 = st.columns([1, 2])
    with col1:
        bhk = st.number_input("BHK", min_value=1, max_value=10, value=2, step=1)
        size = st.number_input("Size (sqft)", min_value=20, max_value=10000, value=800, step=10)
    with col2:
        city = st.selectbox("City", options=cities, index=0)
        st.write("\n")

    submitted = st.form_submit_button("Predict Rent")

if submitted:
    if pipeline is None:
        st.error("Model not found. Please run `train.py` to create the model first.")
    else:
        X = pd.DataFrame([{"BHK": bhk, "Size": size, "City": city}])
        pred = pipeline.predict(X)[0]
        pred = max(0, float(pred))
        st.markdown(f"**Predicted Monthly Rent:** <span class='big-number'>₹ {pred:,.0f}</span>", unsafe_allow_html=True)
        st.metric(label="Estimated Rent", value=f"₹ {pred:,.0f}")

        st.info("This prediction uses a simple linear regression trained on the dataset in the repo.")
