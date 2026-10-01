import json
import os
import tempfile

import streamlit as st
from PIL import Image
from inference_sdk import InferenceHTTPClient


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Fabric Defect Detection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# ROBOFLOW CONFIGURATION
# ============================================================

WORKSPACE_NAME = "vanshika-rai"

WORKFLOW_ID = (
    "vanshikas-project-object-detection-box-2-"
    "vvanshika-s-project-object-detection-box-2-1-"
    "rfdetr-large-t1-logic-2"
)

ROBOFLOW_API_URL = "https://serverless.roboflow.com"


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>
        .stApp {
            background-color: #0b0d14;
            color: #f5f7fb;
        }

        .block-container {
            max-width: 1450px;
            padding-top: 2rem;
            padding-bottom: 3rem;
        }

        h1, h2, h3 {
            color: #f8fafc;
        }

        .hero {
            padding: 28px 30px;
            border-radius: 20px;
            background: linear-gradient(
                120deg,
                #171c2d 0%,
                #111827 55%,
                #24152c 100%
            );
            border: 1px solid #2c3347;
            margin-bottom: 28px;
        }

        .hero-title {
            font-size: 34px;
            font-weight: 750;
            letter-spacing: -1px;
            margin-bottom: 8px;
        }

        .hero-subtitle {
            color: #aab4c8;
            font-size: 15px;
            line-height: 1.6;
        }

        .section-title {
            font-size: 22px;
            font-weight: 700;
            margin: 8px 0 16px 0;
        }

        div.stButton > button {
            background: #ff4b5c;
            color: white;
            border: none;
            border-radius: 10px;
            padding: 0.72rem 1.5rem;
            font-weight: 700;
            font-size: 16px;
            min-height: 48px;
            transition: 0.2s ease;
        }

        div.stButton > button:hover {
            background: #e63d4d;
            color: white;
            border: none;
        }

        div.stButton > button:focus {
            color: white;
            box-shadow: 0 0 0 2px #ff8791;
        }

        [data-testid="stFileUploader"] {
            background: #111520;
            border: 1px dashed #3b4358;
            border-radius: 14px;
            padding: 12px;
        }

        [data-testid="stJson"] {
            background: #111520;
            border: 1px solid #292f40;
            border-radius: 14px;
            padding: 18px;
        }

        .footer {
            color: #707b91;
            text-align: center;
            font-size: 12px;
            margin-top: 35px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# ROBOFLOW CLIENT
# ============================================================

def get_roboflow_client():
    """Create a Roboflow client using Streamlit Secrets."""

    api_key = st.secrets.get("ROBOFLOW_API_KEY", "")

    if not api_key:
        raise ValueError(
            "Roboflow API key is missing. "
            "Add ROBOFLOW_API_KEY in Streamlit Cloud Secrets."
        )

    return InferenceHTTPClient(
        api_url=ROBOFLOW_API_URL,
        api_key=api_key,
    )


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="hero">
        <div class="hero-title">
            🔍 Fabric Defect Detection
        </div>
        <div class="hero-subtitle">
            Upload a fabric image and analyze it using your
            Roboflow object detection workflow.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# IMAGE UPLOAD AND PREVIEW
# ============================================================

upload_col, preview_col = st.columns(
    [1, 1.5],
    gap="large",
)

with upload_col:
    st.markdown(
        '<div class="section-title">Upload Image</div>',
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        "Choose a fabric image",
        type=["jpg", "jpeg", "png", "webp"],
        help="Upload a fabric image for defect detection.",
        key="fabric_image_uploader",
    )

    analyze_clicked = st.button(
        "Analyze",
        type="primary",
        use_container_width=True,
        disabled=uploaded_file is None,
    )


with preview_col:
    st.markdown(
        '<div class="section-title">Image Preview</div>',
        unsafe_allow_html=True,
