import streamlit as st
import pandas as pd
import requests

st.title("🛍️ SHEIN Demand Prediction")

# Load dataset to fetch brand & category options
df = pd.read_csv("cleaned_shein_products.csv")

# Extract unique brands and categories
brands = sorted(df["brand"].dropna().unique())
categories = sorted(df["category"].dropna().unique())

# User input fields
initial_price = st.number_input("Initial Price", min_value=0.0, format="%.2f")
final_price = st.number_input("Final Price", min_value=0.0, format="%.2f")
rating = st.slider("Rating", 0.0, 5.0, step=0.1)

# Dropdown menus for category and brand selection
category = st.selectbox("Select Category", categories)
brand = st.selectbox("Select Brand", brands)

if st.button("Predict Demand"):
    data = {
        "initial_price": float(initial_price),
        "final_price": float(final_price),
        "rating": float(rating),
        "category": category,
        "brand": brand
    }

    try:
        response = requests.post("http://127.0.0.1:5000/predict", json=data)
        response_json = response.json()

        if response.status_code == 200:
            st.success(f"📈 Predicted Demand Score: {response_json['demand_score']}")
        else:
            st.error(f"❌ API Error: {response_json.get('error', 'Unknown error')}")

    except requests.exceptions.RequestException as e:
        st.error(f"❌ API Request failed: {str(e)}")
