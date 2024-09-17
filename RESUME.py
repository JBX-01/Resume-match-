import streamlit as st
import google.generativeai as genai
import os
import PyPDF2 as pdf
from dotenv import load_dotenv
import json
from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load environment variables
load_dotenv()

# Configure Google Generative AI
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_gemini_response(input_text):
    model = genai.GenerativeModel('gemini-pro')
    response = model.generate_content(input_text)
    return response.text

def input_pdf_text(uploaded_file):
    try:
        reader = pdf.PdfReader(uploaded_file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text
    except Exception as e:
        st.error(f"Erreur lors du traitement du fichier : {e}")
        return None

def analyze_text_mistakes(text, lang):
    blob = TextBlob(text)
    mistakes = []
    for sentence in blob.sentences:
        corrections = sentence.correct()
        if corrections != sentence:
            mistakes.append({
                "error": "Grammar Issue",
                "message": f"Suggested correction: {corrections}",
                "suggestions": [],
                "context": sentence
            })
    return mistakes

def extract_keywords(text):
    return set(text.lower().split())

def calculate_match(cv_text, job_description_text):
    vectorizer = CountVectorizer().fit_transform([cv_text, job_description_text])
    vectors = vectorizer.toarray()
    cosine_sim = cosine_similarity([vectors[0]], [vectors[1]])
    return cosine_sim[0][0] * 100

def get_keywords_and_match(cv_text, job_description_text):
    keywords_cv = extract_keywords(cv_text)
    keywords_job = extract_keywords(job_description_text)
    match_percentage = calculate_match(cv_text, job_description_text)
    return keywords_cv, keywords_job, match_percentage

# Streamlit App
st.title("Smart ATS avec Analyse des Fautes")
st.text("Améliorez votre CV pour l'ATS et corrigez les fautes")

# Language choice
lang_choice = st.selectbox("Choisissez la langue de l'analyse", options=["Français", "English"])
lang_code = 'fr' if lang_choice == "Français" else 'en'

# Prompt choice based on language
input_prompt_en = """
Hey Act Like a skilled or very experienced ATS(Application Tracking System)
with a deep understanding of tech field, software engineering, data science, data analyst
and big data engineer. Your task is to evaluate the resume based on the given job description.
You must consider the job market is very competitive and you should provide 
the best assistance for improving resumes. Assign the percentage Matching based 
on the JD and
the missing keywords with high accuracy.
resume: {text}
description: {jd}

I want the response in one single string having the structure:
{{"JD Match": "%", "MissingKeywords": [], "Profile Summary": ""}}
"""

input_prompt_fr = """
Bonjour, agissez comme un ATS (Système de Suivi des Candidatures) très expérimenté et
ayant une compréhension approfondie du domaine de la technologie, de l'ingénierie logicielle, 
de la science des données, de l'analyse de données et de l'ingénierie big data. Votre tâche 
consiste à évaluer le CV en fonction de la description de poste fournie. Vous devez tenir 
compte du fait que le marché de l'emploi est très compétitif et fournir la meilleure assistance 
possible pour améliorer les CV. Attribuez un pourcentage de correspondance basé 
sur la description de poste (JD) et les mots-clés manquants avec une grande précision.
CV : {text}
Description du poste : {jd}

Je veux que la réponse soit sous forme d'une seule chaîne de caractères avec la structure suivante :
{{"Correspondance JD": "%", "MotsClésManquants": [], "RésuméProfil": ""}}
"""

if lang_choice == "Français":
    input_prompt = input_prompt_fr
else:
    input_prompt = input_prompt_en

jd = st.text_area("Collez la description de l'offre d'emploi" if lang_choice == "Français" else "Paste the Job Description")
uploaded_file = st.file_uploader("Téléchargez votre CV (PDF)" if lang_choice == "Français" else "Upload Your Resume (PDF)", type="pdf")

submit = st.button("Soumettre" if lang_choice == "Français" else "Submit")

if submit:
    if jd and uploaded_file is not None:
        # Extract text from the uploaded PDF
        resume_text = input_pdf_text(uploaded_file)
        
        if resume_text:
            # Text mistake analysis
            st.subheader("Analyse des fautes de texte" if lang_choice == "Français" else "Text Mistakes Analysis")
            mistakes = analyze_text_mistakes(resume_text, lang_code)
            if mistakes:
                for mistake in mistakes:
                    st.write(f"Erreur : {mistake['error']}" if lang_choice == "Français" else f"Error: {mistake['error']}")
                    st.write(f"Message : {mistake['message']}" if lang_choice == "Français" else f"Message: {mistake['message']}")
                    st.write(f"Suggestions : {', '.join(mistake['suggestions'])}" if lang_choice == "Français" else f"Suggestions: {', '.join(mistake['suggestions'])}")
                    st.write(f"Contexte : {mistake['context']}" if lang_choice == "Français" else f"Context: {mistake['context']}")
                    st.write("---")
            else:
                st.success("Aucune faute détectée !" if lang_choice == "Français" else "No grammar or spelling mistakes detected!")

            # ATS response via Gemini
            input_prompt_filled = input_prompt.format(text=resume_text, jd=jd)
            with st.spinner('Traitement en cours...' if lang_choice == "Français" else 'Processing resume...'):
                response = get_gemini_response(input_prompt_filled)
                st.subheader("Évaluation ATS" if lang_choice == "Français" else "ATS Evaluation")
                st.write(response)
            
            # Calculate and display match percentage
            keywords_cv, keywords_job, match_percentage = get_keywords_and_match(resume_text, jd)
            st.subheader("Match entre CV et Offre d'emploi" if lang_choice == "Français" else "CV and Job Offer Match")
            st.write(f"Pourcentage de correspondance : {match_percentage:.2f}%")
            st.write(f"Keywords in CV: {keywords_cv}")
            st.write(f"Keywords in Job Description: {keywords_job}")
        else:
            st.error("Échec de l'extraction du texte du PDF." if lang_choice == "Français" else "Failed to extract text from the PDF.")
    else:
        st.error("Veuillez fournir à la fois la description de l'emploi et un CV." if lang_choice == "Français" else "Please provide both the job description and a resume.")
