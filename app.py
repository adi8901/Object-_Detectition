import json
import os
import tempfile

import streamlit as st
from PIL import Image, ImageDraw, ImageFont
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

        .metric-card {
            background: #131722;
            border: 1px solid #292f40;
            border-radius: 15px;
            padding: 20px;
            min-height: 115px;
        }

        .metric-label {
            color: #9ca8bd;
            font-size: 13px;
            margin-bottom: 10px;
        }

        .metric-value {
            font-size: 28px;
            font-weight: 750;
            color: #ffffff;
        }

        .result-card {
            background: #111520;
            border: 1px solid #292f40;
            border-radius: 16px;
            padding: 18px;
            margin-bottom: 15px;
        }

        .result-label {
            color: #aab4c8;
            font-size: 13px;
            margin-bottom: 6px;
        }

        .result-value {
            color: #ffffff;
            font-size: 17px;
            font-weight: 650;
            overflow-wrap: anywhere;
        }

        .status-pill {
            display: inline-block;
            background: #12382c;
            color: #72e3b0;
            border: 1px solid #226b50;
            border-radius: 20px;
            padding: 5px 11px;
            font-size: 12px;
            font-weight: 700;
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

        [data-testid="stTabs"] button {
            font-weight: 600;
        }

        [data-testid="stDataFrame"] {
            border: 1px solid #292f40;
            border-radius: 12px;
            overflow: hidden;
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
# HELPER FUNCTIONS
# ============================================================

def get_client():
    """Create a Roboflow client using the Streamlit secret."""

    api_key = st.secrets.get("ROBOFLOW_API_KEY", "")

    if not api_key:
        raise ValueError(
            "Roboflow API key is missing. "
            "Add ROBOFLOW_API_KEY in Streamlit Secrets."
        )

    return InferenceHTTPClient(
        api_url=ROBOFLOW_API_URL,
        api_key=api_key,
    )


def find_predictions(data):
    """
    Find a predictions list inside a potentially nested
    Roboflow workflow response.
    """

    if isinstance(data, dict):
        predictions = data.get("predictions")

        if isinstance(predictions, list):
            return predictions

        for value in data.values():
            found = find_predictions(value)

            if found is not None:
                return found

    elif isinstance(data, list):
        # Some workflows return a list containing output objects.
        for item in data:
            if isinstance(item, dict):
                found = find_predictions(item)

                if found is not None:
                    return found

    return None


def get_class_name(prediction):
    """Get a readable class name from a prediction."""

    return str(
        prediction.get("class")
        or prediction.get("class_name")
        or prediction.get("label")
        or prediction.get("name")
        or "Unknown"
    )


def get_confidence(prediction):
    """Return confidence as a number between 0 and 1."""

    value = prediction.get(
        "confidence",
        prediction.get("score", 0),
    )

    try:
        value = float(value)
    except (TypeError, ValueError):
        value = 0.0

    # Accommodate confidence values expressed as percentages.
    if value > 1:
        value = value / 100

    return max(0.0, min(value, 1.0))


def get_box(prediction, image_width, image_height):
    """
    Read a standard Roboflow bounding box.

    Roboflow object detection commonly returns x/y as the
    center of the box, with width and height in pixels.
    """

    if all(
        key in prediction
        for key in ("x", "y", "width", "height")
    ):
        x = float(prediction["x"])
        y = float(prediction["y"])
        width = float(prediction["width"])
        height = float(prediction["height"])

        # Handle normalized coordinates if the entire box
        # appears to use values between 0 and 1.
        if (
            max(abs(x), abs(y), abs(width), abs(height)) <= 1
            and image_width > 1
            and image_height > 1
        ):
            x *= image_width
            width *= image_width
            y *= image_height
            height *= image_height

        left = x - width / 2
        top = y - height / 2
        right = x + width / 2
        bottom = y + height / 2

        return (
            max(0, min(left, image_width)),
            max(0, min(top, image_height)),
            max(0, min(right, image_width)),
            max(0, min(bottom, image_height)),
        )

    # Alternative bounding-box format: [x1, y1, x2, y2].
    bbox = prediction.get("bbox")

    if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
        x1, y1, x2, y2 = map(float, bbox)

        return (
            max(0, min(x1, image_width)),
            max(0, min(y1, image_height)),
            max(0, min(x2, image_width)),
            max(0, min(y2, image_height)),
        )

    return None


def draw_detections(image, predictions, confidence_threshold):
    """Draw detection boxes and labels on a copy of the image."""

    annotated = image.copy().convert("RGB")
    draw = ImageDraw.Draw(annotated)

    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 15)
    except OSError:
        font = ImageFont.load_default()

    image_width, image_height = annotated.size
    visible_predictions = []

    for prediction in predictions:
        if not isinstance(prediction, dict):
            continue

        confidence = get_confidence(prediction)

        if confidence < confidence_threshold:
            continue

        box = get_box(
            prediction,
            image_width,
            image_height,
        )

        if box is None:
            continue

        class_name = get_class_name(prediction)
        left, top, right, bottom = box

        if right <= left or bottom <= top:
            continue

        label = f"{class_name}  {confidence:.1%}"

        # Bounding box.
        draw.rectangle(
            (left, top, right, bottom),
            outline="#ff4b5c",
            width=4,
        )

        # Label background.
        try:
            text_box = draw.textbbox(
                (0, 0),
                label,
                font=font,
            )
            text_width = text_box[2] - text_box[0]
            text_height = text_box[3] - text_box[1]
        except AttributeError:
            text_width, text_height = draw.textsize(
                label,
                font=font,
            )

        label_top = max(0, top - text_height - 12)

        draw.rectangle(
            (
                left,
                label_top,
                left + text_width + 12,
                label_top + text_height + 10,
            ),
            fill="#ff4b5c",
        )

        draw.text(
            (left + 6, label_top + 4),
            label,
            fill="white",
            font=font,
        )

        visible_predictions.append(prediction)

    return annotated, visible_predictions


def make_detection_rows(predictions, image):
    """Create a table-friendly representation of predictions."""

    rows = []

    for prediction in predictions:
        if not isinstance(prediction, dict):
            continue

        box = get_box(
            prediction,
            image.width,
            image.height,
        )

        if box is None:
            continue

        left, top, right, bottom = box

        rows.append(
            {
                "Class": get_class_name(prediction),
                "Confidence": f"{get_confidence(prediction):.1%}",
                "X": round(left),
                "Y": round(top),
                "Width": round(right - left),
                "Height": round(bottom - top),
            }
        )

    return rows


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="hero">
        <div class="hero-title">🔍 Fabric Defect Detection</div>
        <div class="hero-subtitle">
            Upload a fabric image and analyze it using your
            Roboflow object detection workflow. Review the
            annotated image, detection details, and raw JSON.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# INPUT SECTION
# ============================================================

left_col, right_col = st.columns(
    [1, 1.6],
    gap="large",
)

with left_col:
    st.markdown(
        '<div class="section-title">Upload Image</div>',
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        "Choose a fabric image",
        type=["jpg", "jpeg", "png", "webp"],
        help="Upload a clear image of the fabric you want to inspect.",
    )

    confidence_threshold = st.slider(
        "Confidence threshold",
        min_value=0,
        max_value=100,
        value=25,
        step=5,
        help="Only show detections at or above this confidence.",
    )

    st.caption(
        "Lower thresholds show more predictions. "
        "Higher thresholds show fewer, more confident predictions."
    )

    analyze_clicked = st.button(
        "Analyze",
        type="primary",
        use_container_width=True,
        disabled=uploaded_file is None,
    )


with right_col:
    st.markdown(
        '<div class="section-title">Image Preview</div>',
        unsafe_allow_html=True,
    )

    if uploaded_file is not None:
        try:
            original_image = Image.open(uploaded_file).convert("RGB")

            st.image(
                original_image,
                caption=uploaded_file.name,
                use_container_width=True,
            )

        except Exception:
            st.error("Unable to read this image. Please upload another file.")
            original_image = None

    else:
        original_image = None

        st.info(
            "Upload an image to preview it here. "
            "Click Analyze to run the detection workflow."
        )


# ============================================================
# RUN ROBOFLOW WORKFLOW
# ============================================================

if analyze_clicked and uploaded_file is not None:
    if original_image is None:
        st.error("Please upload a valid image.")
        st.stop()

    progress = st.status(
        "Analyzing your image...",
        expanded=True,
    )

    temp_path = None

    try:
        progress.write("Preparing image...")
        client = get_client()

        # The Roboflow SDK accepts an image file path.
        with tempfile.NamedTemporaryFile(
            suffix=".jpg",
            delete=False,
        ) as temp_file:
            temp_path = temp_file.name
            original_image.save(temp_path, format="JPEG")

        progress.write("Sending image to Roboflow...")

        result = client.run_workflow(
            workspace_name=WORKSPACE_NAME,
            workflow_id=WORKFLOW_ID,
            images={"image": temp_path},
            use_cache=False,
        )

        progress.update(
            label="Analysis completed",
            state="complete",
            expanded=False,
        )

        st.session_state["analysis_result"] = result
        st.session_state["analysis_image"] = original_image
        st.session_state["analysis_filename"] = uploaded_file.name

    except Exception as exc:
        progress.update(
            label="Analysis failed",
            state="error",
            expanded=True,
        )

        st.error(
            "The image could not be analyzed. "
            "Check your Roboflow workflow, API key, and app logs."
        )

        with st.expander("Technical error details"):
            st.code(str(exc))

    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass


# ============================================================
# RESULTS
# ============================================================

if "analysis_result" in st.session_state:
    result = st.session_state["analysis_result"]
    result_image = st.session_state["analysis_image"]

    st.divider()

    st.markdown(
        '<div class="section-title">Analysis Results</div>',
        unsafe_allow_html=True,
    )

    predictions = find_predictions(result)

    if predictions is None:
        predictions = []

        st.warning(
            "The workflow returned a response, but a standard "
            "'predictions' list was not found. Open the Raw JSON "
            "tab to inspect the workflow output."
        )

    threshold = confidence_threshold / 100

    annotated_image, visible_predictions = draw_detections(
        result_image,
        predictions,
        threshold,
    )

    detection_rows = make_detection_rows(
        visible_predictions,
        result_image,
    )

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    total_detections = len(visible_predictions)

    if visible_predictions:
        average_confidence = sum(
            get_confidence(item)
            for item in visible_predictions
        ) / total_detections
    else:
        average_confidence = 0

    classes = sorted(
        {
            get_class_name(item)
            for item in visible_predictions
        }
    )

    metric1, metric2, metric3 = st.columns(3)

    with metric1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Detections</div>
                <div class="metric-value">{total_detections}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with metric2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Average Confidence</div>
                <div class="metric-value">{average_confidence:.1%}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with metric3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Classes Found</div>
                <div class="metric-value">{len(classes)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.write("")

    # --------------------------------------------------------
    # TABS: VISUAL RESULTS + JSON
    # --------------------------------------------------------

    visual_tab, json_tab = st.tabs(
        ["🖼️ Visual Results", "📄 Raw JSON"]
    )

    with visual_tab:
        image_col, details_col = st.columns(
            [1.5, 1],
            gap="large",
        )

        with image_col:
            st.markdown("### Annotated Image")

            st.image(
                annotated_image,
                use_container_width=True,
            )

            image_bytes = None

            from io import BytesIO

            image_buffer = BytesIO()
            annotated_image.save(
                image_buffer,
                format="PNG",
            )
            image_bytes = image_buffer.getvalue()

            st.download_button(
                label="Download Annotated Image",
                data=image_bytes,
                file_name="fabric_detection_result.png",
                mime="image/png",
                use_container_width=True,
            )

        with details_col:
            st.markdown("### Detection Details")

            if detection_rows:
                for index, row in enumerate(detection_rows, start=1):
                    st.markdown(
                        f"""
                        <div class="result-card">
                            <div class="status-pill">
                                Detection {index}
                            </div>
                            <br><br>
                            <div class="result-label">Class</div>
                            <div class="result-value">
                                {row["Class"]}
                            </div>
                            <br>
                            <div class="result-label">Confidence</div>
                            <div class="result-value">
                                {row["Confidence"]}
                            </div>
                            <br>
                            <div class="result-label">Bounding Box</div>
                            <div class="result-value">
                                X: {row["X"]} &nbsp; Y: {row["Y"]}
                                <br>
                                Width: {row["Width"]} &nbsp;
                                Height: {row["Height"]}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                st.markdown("### Detection Table")
                st.dataframe(
                    detection_rows,
                    use_container_width=True,
                    hide_index=True,
                )

            else:
                st.success(
                    "No detections met the selected confidence "
                    "threshold."
                )

                st.caption(
                    "Try lowering the confidence threshold or "
                    "analyzing another image."
                )

    with json_tab:
        st.markdown("### Complete Roboflow Response")

        st.caption(
            "This is the raw response returned by your workflow, "
            "including fields that may not appear in the visual results."
        )

        st.json(result, expanded=True)

        st.download_button(
            label="Download JSON",
            data=json.dumps(
                result,
                indent=2,
                ensure_ascii=False,
                default=str,
            ),
            file_name="fabric_detection_result.json",
            mime="application/json",
            use_container_width=True,
        )

    st.markdown(
        """
        <div class="footer">
            Fabric Defect Detection · Powered by Roboflow
        </div>
        """,
        unsafe_allow_html=True,
    )
