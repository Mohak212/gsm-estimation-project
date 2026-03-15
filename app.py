import streamlit as st
import json
import os
from scripts.predict import predict_gsm

# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="Fabric GSM Analyzer",
    layout="centered",
)

# ------------------ GLOBAL STYLES ------------------
st.markdown("""
<style>
    body {
        background-color: #f4f6f8;
    }
    .app-title {
        font-size: 36px;
        font-weight: 700;
        color: #1f2933;
        text-align: center;
    }
    .app-subtitle {
        font-size: 16px;
        color: #616e7c;
        text-align: center;
        margin-bottom: 30px;
    }
    .section-title {
        font-size: 18px;
        font-weight: 600;
        color: #1f2933;
        margin-bottom: 10px;
    }
    .card {
        background-color: white;
        padding: 24px;
        border-radius: 10px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.06);
        margin-top: 20px;
    }
    .gsm-value {
        font-size: 48px;
        font-weight: 700;
        color: #0b5ed7;
        text-align: center;
    }
    .gsm-unit {
        font-size: 16px;
        color: #52606d;
        text-align: center;
    }
    .meta-text {
        font-size: 14px;
        color: #52606d;
        text-align: center;
        margin-top: 6px;
    }
</style>
""", unsafe_allow_html=True)

# ------------------ LOAD CONFIG ------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(BASE_DIR, "config", "gsm_ranges.json")) as f:
    gsm_ranges = json.load(f)

# ------------------ HEADER ------------------
st.markdown("<div class='app-title'>Fabric GSM Analyzer</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='app-subtitle'>Microscopic image–based GSM estimation system</div>",
    unsafe_allow_html=True
)

# ------------------ IMAGE INPUT ------------------
st.markdown("<div class='section-title'>Image Acquisition</div>", unsafe_allow_html=True)
image_source = st.radio(
    "",
    ["Live Microscope Camera", "Upload Image"],
    horizontal=True
)

image = None

if image_source == "Live Microscope Camera":
    image = st.camera_input("Capture fabric sample")

else:
    image = st.file_uploader(
        "Upload microscopic fabric image",
        type=["jpg", "jpeg", "png"]
    )

# ------------------ FABRIC SELECTION ------------------
st.markdown("<div class='section-title'>Fabric Parameters</div>", unsafe_allow_html=True)

cloth_type = st.selectbox(
    "Fabric Type",
    list(gsm_ranges.keys())
)

# ------------------ ANALYSIS ------------------
if image and cloth_type:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.image(image, use_column_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    if st.button("Run GSM Analysis", use_container_width=True):
        with st.spinner("Analyzing micro-structure…"):
            temp_path = "temp_image.jpg"
            with open(temp_path, "wb") as f:
                f.write(image.getbuffer())

            gsm_value = predict_gsm(temp_path, cloth_type)
            min_gsm = gsm_ranges[cloth_type]["min"]
            max_gsm = gsm_ranges[cloth_type]["max"]

        os.remove(temp_path)

        # ------------------ RESULT ------------------
        st.markdown("<div class='card'>", unsafe_allow_html=True)

        st.markdown("<div class='section-title'>Analysis Result</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='gsm-value'>{gsm_value}</div>", unsafe_allow_html=True)
        st.markdown("<div class='gsm-unit'>grams per square meter (GSM)</div>", unsafe_allow_html=True)

        st.markdown(
            f"<div class='meta-text'>Fabric type: <b>{cloth_type.capitalize()}</b></div>",
            unsafe_allow_html=True
        )
        st.markdown(
    f"<div class='meta-text'>Fabric type: <b>{cloth_type.capitalize()}</b></div>",
    unsafe_allow_html=True
)


        st.markdown("</div>", unsafe_allow_html=True)

        st.caption(
            "Estimation generated using fabric-specific AI models trained on GSM-constrained microscopic datasets."
        )
