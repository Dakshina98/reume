import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import PyPDF2
import spacy

# Load pre-trained models
model = SentenceTransformer('all-MiniLM-L6-v2')

# Title of the app
st.title("AI-Based Resume Screener")

# File uploader for job description and resumes
job_desc_file = st.text_area("Enter Job Description")
uploaded_files = st.file_uploader("Upload Resumes (PDFs only)", type=["pdf"], accept_multiple_files=True)

if job_desc_file and uploaded_files:
    # Display job description
    st.subheader("Job Description:")
    st.write(job_desc_file)

    # Extract text from uploaded resumes
    st.subheader("Uploaded Resumes:")
    resume_texts = []
    for uploaded_file in uploaded_files:
        with st.spinner(f"Processing {uploaded_file.name}..."):
            reader = PyPDF2.PdfReader(uploaded_file)
            resume_text = " ".join(page.extract_text() for page in reader.pages)
            resume_texts.append(resume_text)
            st.write(f"Resume: {uploaded_file.name}")

    # Generate embeddings
    st.write("Generating embeddings...")
    job_desc_embedding = model.encode(job_desc_file)
    resume_embeddings = [model.encode(resume) for resume in resume_texts]

    # Perform similarity search
    st.write("Calculating similarity scores...")
    index = faiss.IndexFlatL2(384)  # Adjust for embedding dimension
    index.add(np.array(resume_embeddings))
    distances, indices = index.search(np.array([job_desc_embedding]), k=len(resume_texts))

    # Display ranked results
    st.subheader("Ranked Resumes:")
    for idx, score in zip(indices[0], distances[0]):
        st.write(f"Resume: {uploaded_files[idx].name} | Similarity Score: {1 - score:.2f}")
