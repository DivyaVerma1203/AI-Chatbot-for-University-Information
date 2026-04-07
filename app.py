import pandas as pd
import streamlit as st
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 1. Load Data from CSV
@st.cache_data
def load_data():
    df = pd.read_csv('qa_data.csv')
    questions = [str(q).strip() for q in df['question'].tolist()]
    answers = [str(a).strip() for a in df['answer'].tolist()]
    return questions, answers

questions, answers = load_data()

# 2. Text Normalization 
def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

questions_norm = [normalize_text(q) for q in questions]

# 3. TF-IDF Vectorization
vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    stop_words='english',
    min_df=1
)
X = vectorizer.fit_transform(questions_norm)

# 4. Chatbot Logic
def chatbot_response(user_input: str) -> str:
    if not user_input or len(user_input.strip()) < 3:
        return "Please ask a complete question."
    u_norm = normalize_text(user_input)
    user_vec = vectorizer.transform([u_norm])
    similarities = cosine_similarity(user_vec, X).flatten()
    best_idx = similarities.argmax()
    best_score = similarities[best_idx]

    # ------------------------------
    # Threshold Control (CRITICAL)
    # ------------------------------
    SIMILARITY_THRESHOLD = 0.55
    if best_score >= SIMILARITY_THRESHOLD:
        return answers[best_idx]

    # ------------------------------
    # Fallback Response
    # ------------------------------
    return (
        "Sorry, I could not find relevant information for your question. "
        "Please rephrase or contact the university administration."
    )

# 5. Streamlit UI
st.set_page_config(page_title="University AI Chatbot", layout="centered")
st.title("University Information Chatbot")
st.write("Ask questions related to exams, fees, certificates, etc.")

user_question = st.text_input("Enter your question:")
if user_question:
    response = chatbot_response(user_question)
    st.markdown(f"**Chatbot:** {response}")
