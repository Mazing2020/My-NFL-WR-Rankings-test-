import streamlit as st
import numpy as np
import joblib

model = joblib.load("final_gb_model.pkl")
encoder = joblib.load("draft_bin_encoder.pkl")

st.title("RWRS²: Rookie Wide Receiver Success Score")
st.write("Enter a rookie WR's data below to get their RWRS² score:")

draft_pick = st.number_input("Draft Pick (1-256)", 1, 256, 100)
early_declare = st.selectbox("Early Declare?", ["Yes", "No"])
breakout_age = st.number_input("Breakout Age", 17.0, 23.0, 20.0)
dominator = st.slider("Dominator Rating", 0, 100, 70)
athleticism = st.slider("Athleticism Score", 0, 100, 80)
route_running = st.slider("Route Running", 0, 100, 85)
landing_spot = st.slider("Landing Spot & Opportunity", 0, 100, 70)

draft_score = (1 - draft_pick / 256) * 100
early_score = 100 if early_declare == "Yes" else 50
breakout_score = 100 if breakout_age <= 19 else 80 if breakout_age <= 20 else 60 if breakout_age <= 21 else 40
interaction = breakout_score * athleticism
draft_bins = encoder.transform(np.array([[draft_score]]))

X = np.array([[draft_score, early_score, breakout_score, dominator, athleticism, route_running, landing_spot]])
X_augmented = np.column_stack((X, interaction, draft_bins))

if st.button("Calculate RWRS² Score"):
    score = model.predict(X_augmented)[0]
    st.success(f"RWRS² Score: {score:.2f} (Lower = Better Rank)")
