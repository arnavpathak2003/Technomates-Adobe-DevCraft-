import streamlit as st
import PyPDF2
from io import BytesIO
import re
import numpy as np
import pickle

# Load the label encoder
with open("/home/arnav-fedora/pdfClassify/label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)


def preprocess(text):
    text = re.sub("[_\n]", "", text)
    return re.sub(r" +", "", text)


st.set_page_config(
    page_title="PDF Classifier",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# Center-aligned header using markdown
st.markdown(
    "<h1 style='text-align: center;'>PDF Classifier</h1>", unsafe_allow_html=True
)

# Text
st.text(
    """Please enter your document below and it will be classified into the appropriate   
category."""
)

uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
if uploaded_file is not None:
    import tensorflow as tf
    import tensorflow_text as text

if st.button("Classify"):
    pdf_uploaded = uploaded_file is not None
    if not pdf_uploaded:
        st.write("Please upload a PDF file first")

    else:
        from tensorflow.keras.models import load_model

        model = load_model("/home/arnav-fedora/pdfClassify/pdf_classifier")

        pdf_file = PyPDF2.PdfReader(BytesIO(uploaded_file.read()))

        text = ""
        for page in pdf_file.pages:
            text += page.extract_text()
        text = preprocess(text)

        text = np.expand_dims(text, axis=0)

        y = model.predict(text)
        predicted_indices = np.argmax(y, axis=-1)
        original_labels = label_encoder.inverse_transform(predicted_indices)
        st.write(f"The given PDF is a {original_labels[0]} document.")
