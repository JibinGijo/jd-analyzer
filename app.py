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

        jd_results=classifier(jd,common_skills)
        jd_skills=[(label,score) for label,score in zip(jd_results["labels"],jd_results["scores"]) if score>0.05]


        resume_results=classifier(resume_text,common_skills)
        resume_skills=[label for label,score in zip(resume_results["labels"],resume_results["scores"]) if score>0.05]

        st.write("### Skills found in your Resume:")
        st.write(",".join(resume_skills))



        matched=[]
        missing=[]
        matched_scores=[]


        for m in jd_skills:
            if m[0] in resume_skills:
                matched.append(f"{m[0]} - {round(m[1]*100, 2)}%")
                matched_scores.append(m[1])
            else:
                missing.append(m[0])
        

        st.write("Resume Score:")
        if matched_scores:
            st.write(round((sum(matched_scores)/len(matched_scores))*100,2))
        else:
            st.write(0)
        
        st.write("Matched Skills:")
        for m in matched:
            st.success(m)
        st.write("Missing Skills:")
        for m in missing:
            st.error(m)
        
    else:
        st.warning("fill all details")

