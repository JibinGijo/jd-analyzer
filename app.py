import streamlit as st
from transformers import pipeline
import pdfplumber

st.title("JD Analyzer")
st.subheader("Resume and Job Description Matching")


common_skills = ["Python", "Java", "React", "Machine Learning", "SQL", "MySQL",
                 "Docker", "FastAPI", "NLP", "Computer Vision", "Flask", "TypeScript",
                 "Arduino", "ESP32", "Supabase", "PostgreSQL", "C++"]


jd=st.text_area("Paste Job Description")
uploaded_file = st.file_uploader("Upload Resume PDF", type="pdf")


if (st.button("Analyze")):
    if jd and uploaded_file:
        with pdfplumber.open(uploaded_file) as pdf:
            resume_text=""
            for page in pdf.pages:
                resume_text+=page.extract_text()
        classifier=pipeline("zero-shot-classification")
        resume_results=classifier(resume_text,common_skills)
        extracted_skills=[label for label,score in zip(resume_results["labels"],resume_results["scores"]) if score>0.05]

        st.write("### Skills found in your Resume:")
        st.write(",".join(extracted_skills))

        results=classifier(jd,extracted_skills)

        matched=[]
        missing=[]

        for label , score in zip(results["labels"],results["scores"]):
            if score>0.1:
                matched.append(f"{label} - {round(score*100, 2)}%")
            else:
                missing.append(label)
        
        st.write("Matched Skills:")
        for m in matched:
            st.success(m)
        
        st.write("Missing Skills:")
        for m in missing:
            st.error(m)
    else:
        st.warning("fill all details")

