import streamlit as st
import pandas as pd
import re
import spacy
import subprocess
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------------
# Streamlit Page Config
# -------------------------------
st.set_page_config(page_title="AI Resume Screener", page_icon="📝", layout="centered")

st.title("📝 AI Resume Screener")
st.write("Paste a **job description** and get the **top matching resumes** from the dataset.")

# -------------------------------
# Load Dataset
# -------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("Resume.csv")
    return df

df = load_data()

# -------------------------------
# Ensure SpaCy Model is Installed
# -------------------------------
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    st.warning("Downloading SpaCy model for the first time... ⏳")
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")

# -------------------------------
# Text Cleaning Function
# -------------------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^a-zA-Z ]', '', text)
    doc = nlp(text)
    return ' '.join([token.lemma_ for token in doc if not token.is_stop])

# Preprocess Resumes (sample 100 for speed)
df['cleaned_resume'] = df['Resume'].apply(clean_text)
resumes = df['cleaned_resume'].tolist()[:100]

# -------------------------------
# Streamlit UI
# -------------------------------
job_desc = st.text_area("📌 Paste Job Description Here", height=150)
top_k = st.slider("How many top resumes to show?", 1, 10, 5)

if st.button("🔍 Analyze"):
    if job_desc.strip() == "":
        st.warning("⚠️ Please enter a job description first.")
    else:
        # TF-IDF Vectorization
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform([job_desc] + resumes)

        # Compute Cosine Similarity
        scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()

        # Create Ranking
        ranking = pd.DataFrame({
            "Resume_Index": range(1, len(resumes)+1),
            "Match_Score": scores
        }).sort_values(by="Match_Score", ascending=False)

        top_results = ranking.head(top_k)

        # Show Results
        st.subheader(f"🏆 Top {top_k} Matching Resumes")
        st.dataframe(top_results.style.format({"Match_Score": "{:.2f}"}))

        # Show bar chart of scores
        st.bar_chart(top_results.set_index("Resume_Index")["Match_Score"])

        st.success("✅ Analysis complete! These are the best matching resumes.")
