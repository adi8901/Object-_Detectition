import json
import os
import tempfile

import streamlit as st
from PIL import Image
from inference_sdk import InferenceHTTPClient


# -------------------- PAGE CONFIG --------------------

st.set_page_config(
    page_title="Fabric Defect Detection",
    page_icon="🔍",
    layout="wide",
)


# -------------------- ROBOFLOW CONFIG --------------------

WORKSPACE_NAME = "vanshika-rai"

WORKFLOW_ID = (
    "vanshikas-project-object-detection-box-2-"
    "vvanshika-s-project-object-detection-box-2-1-"
    "rfdetr-large-t1-logic-2"
)

ROBOFLOW_API_URL = "https://serverless.roboflow.com"


# -------------------- CUSTOM CSS --------------------

st.markdown(
    """
    <style>
    .stApp {
        background-color: #0b0d14;
        color: #f5f7fb;
    }

    .block-container {
        max-width: 1400px;
        padding-top: 2rem;
    }

    .hero {
        padding: 28px;
        border-radius: 18px;
        background: linear-gradient(
            120deg, #171c2d, #111827, #24152c
        );
        border: 1px solid #2c3347;
        margin-bottom: 28px;
    }

    .hero-title {
        font-size: 32px;
        font-weight: 750;
        margin-bottom: 8px;
    }

    .hero-subtitle {
        color: #aab4c8;
        font-size: 15px;
    }

    div.stButton > button {
        background: #ff4b5c;
        color: white;
        border: none;
        border-radius: 10px;
        min-height: 46px;
        font-weight: 700;
        font-size: 16px;
    }

    div.stButton > button:hover {
        background: #e63d4d;
        color: white;
    }

    [data-testid="stFileUploader"] {
        background: #111520;
        border: 1px dashed #3b4358;
        border-radius: 14px;
        padding: 12px;
    }

    .section-title {
        font-size: 22px;
        font-weight: 700;
        margin-bottom: 16px;
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


# -------------------- HEADER --------------------

st.markdown(
    """
    <div class="hero">
        <div class="hero-title">
            🔍 Fabric Defect Detection
        </div>
        <div class="hero-subtitle">
            Upload a fabric image and analyze it using
            your Roboflow object detection workflow.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# -------------------- IMAGE UPLOAD --------------------

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


# -------------------- IMAGE PREVIEW --------------------

original_image = None

with preview_col:
    st.markdown(
        '<div class="section-title">Image Preview</div>',
        unsafe_allow_html=True,
    )

    if uploaded_file is not None:
        try:
            original_image = Image.open(
                uploaded_file
            ).convert("RGB")

            # Reduce preview resolution only.
            preview_image = original_image.copy()
            preview_image.thumbnail((300, 200))

            st.image(
                preview_image,
                caption=uploaded_file.name,
                width=500,
            )

        except Exception:
            st.error("Unable to read this image.")
    else:
        st.info("Upload an image to preview it here.")


# -------------------- CLEAR OLD RESULTS --------------------

current_filename = (
    uploaded_file.name
    if uploaded_file is not None
    else None
)

if st.session_state.get("current_filename") != current_filename:
    st.session_state.pop("analysis_result", None)
    st.session_state["current_filename"] = current_filename


# -------------------- ANALYZE IMAGE --------------------

if analyze_clicked and uploaded_file is not None:

    if original_image is None:
        st.error("Please upload a valid image.")
        st.stop()

    api_key = st.secrets.get("ROBOFLOW_API_KEY", "")

    if not api_key:
        st.error(
            "Roboflow API key is missing. Add "
            "ROBOFLOW_API_KEY in Streamlit Secrets."
        )
        st.stop()

    temp_path = None

    try:
        client = InferenceHTTPClient(
            api_url=ROBOFLOW_API_URL,
            api_key=api_key,
        )

        with st.spinner("Analyzing image..."):

            # Save the original-resolution image.
            with tempfile.NamedTemporaryFile(
                suffix=".jpg",
                delete=False,
            ) as temp_file:
                temp_path = temp_file.name

                original_image.save(
                    temp_path,
                    format="JPEG",
                )

            # Call the Roboflow workflow.
            result = client.run_workflow(
                workspace_name=WORKSPACE_NAME,
                workflow_id=WORKFLOW_ID,
                images={"image": temp_path},
                use_cache=False,
            )

        st.session_state["analysis_result"] = result

    except Exception as exc:
        st.error("Image analysis failed.")

        with st.expander("Technical error details"):
            st.code(str(exc))

    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass


# -------------------- JSON RESPONSE ONLY --------------------

if "analysis_result" in st.session_state:

    st.divider()

    st.markdown(
        '<div class="section-title">Analysis Response</div>',
        unsafe_allow_html=True,
    )

    result = st.session_state["analysis_result"]

    st.json(result, expanded=True)

    json_data = json.dumps(
        result,
        indent=2,
        ensure_ascii=False,
        default=str,
    )

    st.download_button(
        label="Download JSON",
        data=json_data,
        file_name="fabric_detection_result.json",
        mime="application/json",
    )


# -------------------- FOOTER --------------------

st.markdown(
    """
    <div class="footer">
        Fabric Defect Detection · Powered by Roboflow
    </div>
    """,
    unsafe_allow_html=True,
)
