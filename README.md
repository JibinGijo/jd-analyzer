# JD Analyzer

An AI-powered tool that matches your resume against a job description and tells you exactly where you stand.

## Live Demo
[jd-analyzer0.streamlit.app](https://jd-analyzer0.streamlit.app/)

## What it does
- Extracts skills from your resume PDF automatically
- Detects skills required by the job description
- Shows matched and missing skills
- Gives an overall match score

## How it works
1. Extracts text from uploaded resume using pdfplumber
2. Runs zero-shot classification on both resume and JD against a common skills list
3. Compares the two and calculates a match score

## Tech Stack
- Python
- Streamlit
- HuggingFace Transformers (zero-shot classification)
- pdfplumber