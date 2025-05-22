import streamlit as st
import pandas as pd
import joblib
import numpy as np

st.title("📊 Voorspeller: Productie o.b.v. Klimaat & Energie")

@st.cache_resource
def laad_model():
    model = joblib.load("data/voorspelling_model.pkl")
    with open("data/features.txt") as f:
        features = [line.strip() for line in f]
    return model, features

model, features = laad_model()

st.markdown("Voer onderstaande waarden in om een productievoorspelling te maken:")

# Dynamisch invoervelden genereren
user_input = {}
for feat in features:
    user_input[feat] = st.number_input(feat, value=0.0)

# Voorbereiden invoer
input_df = pd.DataFrame([user_input])

# Voorspellen
if st.button("Voorspel Productie"):
    voorspelling = model.predict(input_df)[0]
    st.success(f"📦 Verwachte productie: {voorspelling:.2f} kg")
