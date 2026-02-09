import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.pagesizes import A4
import tempfile
import gdown
import os


@st.cache_resource
def load_my_model():
    
    file_id = '1FASCzbHajt4dWvEktdGRMqkEBxlkMbFt'
    url = f'https://drive.google.com/uc?id={file_id}'
    output = 'Breast_Cancer_Segmentation.h5'
    
    if not os.path.exists(output):
        with st.spinner('جاري تحميل الموديل من Google Drive... قد يستغرق ذلك دقيقة'):
            gdown.download(url, output, quiet=False)
    
    return load_model(output, compile=False)

model = load_my_model()
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
    st.image(image_rgb, use_container_width=True)

    input_img = preprocess(image_rgb)
    mask = predict(input_img)

    st.subheader("Predicted Tumor Mask")
    # عرض الـ Mask كصورة أبيض وأسود
    st.image(mask * 255, use_container_width=True)

    # Overlay
    overlay = cv2.resize(mask, (image_rgb.shape[1], image_rgb.shape[0]))
    overlay = (overlay * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(overlay, cv2.COLORMAP_JET)
    combined = cv2.addWeighted(image_rgb, 0.7, heatmap, 0.3, 0)

    st.subheader("Tumor Highlighted")
    st.image(combined, use_container_width=True)

    # Tumor Ratio
    tumor_ratio = np.sum(mask) / (IMG_SIZE * IMG_SIZE)
    percentage = tumor_ratio * 100

    st.subheader("Tumor Area Percentage")
    st.write(f"**{percentage:.2f}%**")

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
            ["Analysis Item", "Value"],
            ["Tumor Area (%)", f"{percentage:.2f}%"],
            ["Risk Level", level],
            ["Recommendation", recommendation]
        ]

        table = Table(data, colWidths=[2.5*inch, 3.5*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.lightgrey),
            ('TEXTCOLOR', (0,0), (-1,0), colors.black),
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('GRID', (0,0), (-1,-1), 1, colors.grey),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0,0), (-1,0), 12),
        ]))

        elements.append(table)
        elements.append(Spacer(1, 0.5 * inch))
        elements.append(Paragraph("<i>Note: This is an AI-generated report for clinical decision support and should be reviewed by a specialist.</i>", styles["Italic"]))

        doc.build(elements)

        with open(pdf_path, "rb") as f:
            st.download_button(
                "Download Report",
                f,
                file_name="Breast_Cancer_Report.pdf",
                mime="application/pdf"
            )
