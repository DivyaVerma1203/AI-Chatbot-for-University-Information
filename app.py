import pyodbc
import streamlit as st
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 1. Connect to Azure SQL Database

conn = pyodbc.connect(
    "Driver={ODBC Driver 17 for SQL Server};"
    f"Server={st.secrets['DB_SERVER']};"
    f"Database={st.secrets['DB_NAME']};"
    f"Uid={st.secrets['DB_USER']};"
    f"Pwd={st.secrets['DB_PASSWORD']};"
    "Encrypt=yes;"
    "TrustServerCertificate=no;"
    "Connection Timeout=30;"
)
cursor = conn.cursor()

# 2. Fetch Questions & Answers

cursor.execute(
    "SELECT question, answer FROM faq WHERE question IS NOT NULL AND answer IS NOT NULL"
)
data = cursor.fetchall()

questions = [str(row[0]).strip() for row in data]
answers = [str(row[1]).strip() for row in data]

# 3. Text Normalization

def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

questions_norm = [normalize_text(q) for q in questions]

# 4. TF-IDF Vectorization (Semantic Matching)

vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    stop_words='english',
    min_df=1
)
X = vectorizer.fit_transform(questions_norm)

# 5. Chatbot Logic with Strict Thresholding

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

# 6. Streamlit UI

st.set_page_config(page_title="University AI Chatbot", layout="centered")

st.title("University Information Chatbot")
st.write("Ask questions related to exams, fees, certificates, etc.")

user_question = st.text_input("Enter your question:")

if user_question:
    response = chatbot_response(user_question)
    st.markdown(f"**Chatbot:** {response}")














