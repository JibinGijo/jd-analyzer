import streamlit as st
from transformers import pipeline

st.title("JD Analyzer")
st.subheader("Resume and Job Description Matching")

jd=st.text_area("Paste Job Description")
resume=st.text_area("Paste Resume")
skills=st.text_area("Enter your skills separated by commas (e.g. Python, ML, React)")

if (st.button("Analyze")):
    if jd and resume and skills:
        skilllist = list(map(str.strip,skills.split(",")))
        classifier=pipeline("zero-shot-classification")
        results=classifier(jd,skilllist)
        st.write("results")
        for label , score in zip(results["labels"],results["scores"]):
            st.write(f"{label} - {round(score*100,2)}")
    else:
        st.warning("fill all details")

