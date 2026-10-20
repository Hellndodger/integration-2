import io
from pathlib import Path

import requests
import streamlit as st

API_URL = "http://api:8000/predict"

st.title("Speech Commands classifier")

uploaded_file = st.file_uploader("Upload a WAV file", type=["wav"])

if uploaded_file is not None:
    files = {"file": (uploaded_file.name, uploaded_file.read(), "audio/wav")}
    with st.spinner("Running inference..."):
        try:
            response = requests.post(API_URL, files=files, timeout=30)
            if response.status_code == 200:
                result = response.json()
                st.success("Prediction complete")
                st.metric("Prediction", result["label"])
                st.metric("Confidence", f"{result['confidence']:.2%}")
                st.metric("Latency", f"{result['latency_ms']:.2f} ms")
            else:
                st.error(f"API error: {response.text}")
        except requests.RequestException as exc:
            st.error(f"Connection failed: {exc}")
