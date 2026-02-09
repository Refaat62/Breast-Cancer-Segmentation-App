import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Table
from reportlab.lib.pagesizes import A4
from reportlab.platypus import TableStyle
import tempfile

# =========================
# Load Model
# =========================
model = load_model("Breast Cancer Segmentation.h5", compile=False)
IMG_SIZE = 256

# =========================
# Preprocessing
# =========================
def preprocess(img):
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# =========================
# Prediction
# =========================
def predict(img):
    pred = model.predict(img)[0]
    mask = (pred > 0.3).astype(np.uint8)
    return mask

# =========================
# Risk Logic
# =========================
def risk_level(tumor_ratio):
    if tumor_ratio < 0.005:
        return "Low Risk", "Routine follow-up after 6 months recommended."
    elif tumor_ratio < 0.02:
        return "Medium Risk", "Further imaging (Ultrasound / MRI) advised."
    else:
        return "High Risk", "Immediate oncologist consultation & biopsy recommended."

# =========================
# Streamlit UI
# =========================
st.set_page_config(page_title="Breast Cancer AI System", layout="centered")
st.title("🩺 Breast Cancer Clinical Decision Support System")

uploaded = st.file_uploader("Upload Breast Image", type=["jpg","png","jpeg"])

if uploaded is not None:

    file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    st.subheader("Original Image")
    st.image(image_rgb, use_column_width=True)

    input_img = preprocess(image_rgb)
    mask = predict(input_img)

    st.subheader("Predicted Tumor Mask")
    st.image(mask, clamp=True, use_column_width=True)

    # Overlay
    overlay = cv2.resize(mask, (image_rgb.shape[1], image_rgb.shape[0]))
    overlay = overlay * 255
    heatmap = cv2.applyColorMap(overlay.astype(np.uint8), cv2.COLORMAP_JET)
    combined = cv2.addWeighted(image_rgb, 0.7, heatmap, 0.3, 0)

    st.subheader("Tumor Highlighted")
    st.image(combined, use_column_width=True)

    # Tumor Ratio
    tumor_ratio = np.sum(mask) / (IMG_SIZE * IMG_SIZE)
    percentage = tumor_ratio * 100

    st.subheader("Tumor Area Percentage")
    st.write(f"{percentage:.2f}%")

    # Recommendation System
    level, recommendation = risk_level(tumor_ratio)

    st.subheader("Risk Assessment")
    if level == "Low Risk":
        st.success(level)
    elif level == "Medium Risk":
        st.warning(level)
    else:
        st.error(level)

    st.write(recommendation)

    # =========================
    # Generate PDF Report
    # =========================
    if st.button("Generate Medical Report (PDF)"):

        pdf_path = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf").name
        doc = SimpleDocTemplate(pdf_path, pagesize=A4)
        elements = []

        styles = getSampleStyleSheet()
        elements.append(Paragraph("<b>Breast Cancer AI Report</b>", styles["Title"]))
        elements.append(Spacer(1, 0.3 * inch))

        data = [
            ["Tumor Area (%)", f"{percentage:.2f}%"],
            ["Risk Level", level],
            ["Recommendation", recommendation]
        ]

        table = Table(data, colWidths=[2.5*inch, 3.5*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.lightgrey),
            ('GRID', (0,0), (-1,-1), 1, colors.grey)
        ]))

        elements.append(table)
        elements.append(Spacer(1, 0.5 * inch))

        doc.build(elements)

        with open(pdf_path, "rb") as f:
            st.download_button(
                "Download Report",
                f,
                file_name="Breast_Cancer_Report.pdf"
            )
