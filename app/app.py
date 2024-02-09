import streamlit as st
import PyPDF2
from io import BytesIO

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
    pdf_file = PyPDF2.PdfFileReader(BytesIO(uploaded_file.read()))
    num_pages = pdf_file.getNumPages()
    st.write(f"Number of pages: {num_pages}")

    # Extract text from each page
    text = ""
    for page in range(num_pages):
        text += pdf_file.getPage(page).extractText()
    st.write(text)
