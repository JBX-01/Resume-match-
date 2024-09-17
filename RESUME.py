import streamlit as st
import google.generativeai as genai
import os
import PyPDF2 as pdf
from dotenv import load_dotenv
import json
from textblob import TextBlob
from textblob.exceptions import MissingCorpusError
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import importlib

# Function to download NLTK corpora
def download_nltk_corpora():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')

download_nltk_corpora()

# Load environment variables
load_dotenv()

# Configure Google Generative AI
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Try importing language_tool_python
language_tool_installed = importlib.util.find_spec("language_tool_python") is not None
if language_tool_installed:
    import language_tool_python
else:
    st.error("language_tool_python is not installed. Please install it using 'pip install language_tool_python'.")

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
        st.error(f"Error processing file: {e}")
        return None

def analyze_text_mistakes(text, lang):
    mistakes = []
    if lang == 'en':
        blob = TextBlob(text)
        try:
            for sentence in blob.sentences:
                corrections = sentence.correct()
                if corrections != sentence:
                    mistakes.append({
                        "error": "Grammar Issue",
                        "message": f"Suggested correction: {corrections}",
                        "suggestions": [],
                        "context": sentence
                    })
        except MissingCorpusError:
            st.error("TextBlob corpus missing. Please ensure the necessary corpora are downloaded.")
    elif lang == 'fr' and language_tool_installed:
        tool = language_tool_python.LanguageTool('fr')
        matches = tool.check(text)
        for match in matches:
            mistakes.append({
                "error": match.ruleId,
                "message": match.message,
                "suggestions": match.replacements,
                "context": text[match.offset:match.offset + match.errorLength]
            })
    return mistakes

def extract_keywords(text):
    return set(text.lower().split())

def calculate_match(cv_text, job_description_text, method):
    if method == 'Standard':
        vectorizer = CountVectorizer().fit_transform([cv_text, job_description_text])
        vectors = vectorizer.toarray()
        cosine_sim = cosine_similarity([vectors[0]], [vectors[1]])
        return cosine_sim[0][0] * 100
    return 0

def get_keywords_and_match(cv_text, job_description_text, method):
    keywords_cv = extract_keywords(cv_text)
    keywords_job = extract_keywords(job_description_text)
    match_percentage = calculate_match(cv_text, job_description_text, method)
    return keywords_cv, keywords_job, match_percentage

# Streamlit App
st.title("Smart ATS with Mistake Analysis")
st.text("Improve your resume for ATS and correct mistakes")

# Language choice
lang_choice = st.selectbox("Choose the language for analysis", options=["English", "Français"])
lang_code = 'en' if lang_choice == "English" else 'fr'

# Analysis method choice
analysis_method = st.selectbox("Choose the analysis method", options=["Standard", "Custom"])

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

jd = st.text_area("Paste the Job Description" if lang_choice == "English" else "Collez la description de l'offre d'emploi")
uploaded_file = st.file_uploader("Upload Your Resume (PDF)" if lang_choice == "English" else "Téléchargez votre CV (PDF)", type="pdf")

submit = st.button("Submit" if lang_choice == "English" else "Soumettre")

if submit:
    if jd and uploaded_file is not None:
        # Extract text from the uploaded PDF
        resume_text = input_pdf_text(uploaded_file)
        
        if resume_text:
            # Text mistake analysis
            st.subheader("Text Mistakes Analysis" if lang_choice == "English" else "Analyse des fautes de texte")
            mistakes = analyze_text_mistakes(resume_text, lang_code)
            if mistakes:
                for mistake in mistakes:
                    st.write(f"Error: {mistake['error']}" if lang_choice == "English" else f"Erreur : {mistake['error']}")
                    st.write(f"Message: {mistake['message']}" if lang_choice == "English" else f"Message : {mistake['message']}")
                    st.write(f"Suggestions: {', '.join(mistake['suggestions'])}" if lang_choice == "English" else f"Suggestions : {', '.join(mistake['suggestions'])}")
                    st.write(f"Context: {mistake['context']}" if lang_choice == "English" else f"Contexte : {mistake['context']}")
                    st.write("---")
            else:
                st.success("No grammar or spelling mistakes detected!" if lang_choice == "English" else "Aucune faute détectée !")

            # ATS response via Gemini
            input_prompt_filled = input_prompt.format(text=resume_text, jd=jd)
            with st.spinner('Processing resume...' if lang_choice == "English" else 'Traitement en cours...'):
                response = get_gemini_response(input_prompt_filled)
                st.subheader("ATS Evaluation" if lang_choice == "English" else "Évaluation ATS")
                st.write(response)
            
            # Calculate and display match percentage
            keywords_cv, keywords_job, match_percentage = get_keywords_and_match(resume_text, jd, analysis_method)
            st.subheader("CV and Job Offer Match" if lang_choice == "English" else "Match entre CV et Offre d'emploi")
            st.write(f"Match Percentage: {match_percentage:.2f}%" if lang_choice == "English" else f"Pourcentage de correspondance : {match_percentage:.2f}%")
            st.write(f"Keywords in CV: {keywords_cv}")
            st.write(f"Keywords in Job Description: {keywords_job}")
        else:
            st.error("Failed to extract text from the PDF." if lang_choice == "English" else "Échec de l'extraction du texte du PDF.")
    else:
        st.error("Please provide both the job description and a resume." if lang_choice == "English" else "Veuillez fournir à la fois la description de l'emploi et un CV.")
