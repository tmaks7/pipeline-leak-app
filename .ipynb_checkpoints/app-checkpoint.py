import streamlit as st
import numpy as np

st.title("Pipeline Leak Detection")

pressure_diff = st.number_input("Pressure Difference")
flow_diff = st.number_input("Flow Difference")

if st.button("Predict"):
    # Dummy logic (replace with your trained model later)
    if pressure_diff > 8 or flow_diff > 15:
        st.error("Leak Detected")
    else:
        st.success("Normal Operation")