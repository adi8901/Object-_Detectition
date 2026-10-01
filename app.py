import os
import tempfile

import streamlit as st
from PIL import Image, ImageDraw
from inference_sdk import (
    InferenceHTTPClient,
    InferenceConfiguration,
)

# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="Fabric Defect Detection",
    page_icon="🔍",
    layout="wide",
)

st.title("🔍 Fabric Defect Detection")
st.write(
    "Upload a fabric image to detect defects using "
    "your Roboflow Object Detection model."
)

# -----------------------------
# Roboflow workflow details
# -----------------------------
WORKSPACE_NAME = "vanshika-rai"

WORKFLOW_ID = (
    "vanshikas-project-object-detection-box-2-"
    "vvanshika-s-project-object-detection-box-2-"
    "1-rfdetr-large-t1-logic-2"
)

# -----------------------------
# User interface
# -----------------------------
confidence_threshold = st.slider(
    "Confidence threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.20,
    step=0.05,
)

uploaded_file = st.file_uploader(
    "Upload a fabric image",
    type=["jpg", "jpeg", "png"],
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")

    st.subheader("Uploaded Image")
    st.image(image, use_container_width=True)

    if st.button("Detect Objects", type="primary"):

        image_path = None

        try:
            # Read API key from Streamlit Secrets
            api_key = st.secrets["ROBOFLOW_API_KEY"]

            client = InferenceHTTPClient(
                api_url="https://serverless.roboflow.com",
                api_key=api_key,
            ).configure(
                InferenceConfiguration(
                    api_key_transport="header"
                )
            )

            # Save uploaded image temporarily
            with tempfile.NamedTemporaryFile(
                suffix=".jpg",
                delete=False,
            ) as temp_file:
                image.save(temp_file.name, format="JPEG")
                image_path = temp_file.name

            # Run the hosted Roboflow workflow
            with st.spinner("Detecting objects..."):
                result = client.run_workflow(
                    workspace_name=WORKSPACE_NAME,
                    workflow_id=WORKFLOW_ID,
                    images={"image": image_path},
                    use_cache=False,
                )

            # -----------------------------
            # Process workflow response
            # -----------------------------
            output = result

            if isinstance(output, list) and output:
                output = output[0]

            if isinstance(output, dict):
                output = output.get("predictions", output)

            if isinstance(output, list) and output:
                output = output[0]

            if not isinstance(output, dict):
                st.error("Unexpected response from Roboflow.")
                st.json(result)
                st.stop()

            predictions = output.get("predictions", [])
            image_info = output.get("image", {})

            # Handle nested object-detection output
            if isinstance(predictions, dict):
                image_info = predictions.get("image", image_info)
                predictions = predictions.get("predictions", [])

            if not isinstance(predictions, list):
                predictions = []

            # -----------------------------
            # Draw bounding boxes
            # -----------------------------
            annotated_image = image.copy()
            draw = ImageDraw.Draw(annotated_image)

            image_width, image_height = image.size

            inference_width = image_info.get(
                "width", image_width
            )
            inference_height = image_info.get(
                "height", image_height
            )

            scale_x = image_width / inference_width
            scale_y = image_height / inference_height

            detected = []

            for prediction in predictions:

                confidence = prediction.get("confidence", 0)

                if confidence < confidence_threshold:
                    continue

                class_name = prediction.get(
                    "class", "Unknown"
                )

                x = prediction.get("x", 0) * scale_x
                y = prediction.get("y", 0) * scale_y
                width = prediction.get("width", 0) * scale_x
                height = prediction.get("height", 0) * scale_y

                left = max(
                    0, int(x - width / 2)
                )
                top = max(
                    0, int(y - height / 2)
                )
                right = min(
                    image_width - 1,
                    int(x + width / 2),
                )
                bottom = min(
                    image_height - 1,
                    int(y + height / 2),
                )

                label = f"{class_name} ({confidence:.0%})"

                draw.rectangle(
                    [left, top, right, bottom],
                    outline="red",
                    width=3,
                )

                draw.text(
                    (left, max(0, top - 18)),
                    label,
                    fill="red",
                )

                detected.append({
                    "Class": class_name,
                    "Confidence": f"{confidence:.1%}",
                    "X": round(x, 1),
                    "Y": round(y, 1),
                    "Width": round(width, 1),
                    "Height": round(height, 1),
                })

            # -----------------------------
            # Display results
            # -----------------------------
            st.subheader("Detection Results")

            if detected:
                col1, col2 = st.columns([3, 2])

                with col1:
                    st.image(
                        annotated_image,
                        caption="Detected objects",
                        use_container_width=True,
                    )

                with col2:
                    st.metric(
                        "Objects detected",
                        len(detected),
                    )
                    st.dataframe(
                        detected,
                        use_container_width=True,
                    )

            else:
                st.info(
                    "No objects found above the selected "
                    "confidence threshold."
                )

                st.image(
                    image,
                    caption="Original image",
                    use_container_width=True,
                )

            with st.expander("View raw Roboflow response"):
                st.json(result)

        except Exception as error:
            st.error(f"Detection failed: {error}")

        finally:
            if image_path and os.path.exists(image_path):
                os.remove(image_path)
