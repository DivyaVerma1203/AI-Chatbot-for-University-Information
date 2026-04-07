import pandas as pd
import streamlit as st
import re
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer, util

# 1. Load Data

@st.cache_data
def load_data():
    df = pd.read_csv('Chatbot Data.csv')
    questions = [str(q).strip() for q in df['question'].tolist()]
    answers   = [str(a).strip() for a in df['answer'].tolist()]
    return questions, answers

# 2. Load Best Semantic Model

@st.cache_resource
def load_model():
    # all-mpnet-base-v2 is the highest accuracy
    # sentence-transformer model available
    return SentenceTransformer('all-mpnet-base-v2')

# 3. Pre-compute Embeddings for Questions + Answers

@st.cache_resource
def get_embeddings(_model, questions, answers):
    q_embeddings = _model.encode(questions, convert_to_tensor=True)
    a_embeddings = _model.encode(answers,   convert_to_tensor=True)
    return q_embeddings, a_embeddings

# 4. Clean Text

def clean(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text

# 5. Topic Guard

UNIVERSITY_KEYWORDS = {
    'exam', 'exams', 'examination', 'result', 'results', 'marks', 'mark',
    'grade', 'grades', 'cgpa', 'attendance', 'admit', 'card', 'hall',
    'ticket', 'timetable', 'schedule', 'semester', 'subject', 'subjects',
    'fee', 'fees', 'registration', 'register', 'revaluation', 'backlog',
    'supplementary', 'degree', 'certificate', 'marksheet', 'practical',
    'internal', 'external', 'assignment', 'portal', 'student', 'university',
    'college', 'department', 'faculty', 'syllabus', 'paper', 'passing',
    'fail', 'failed', 'pass', 'duplicate', 'convocation', 'malpractice',
    'cheating', 'late', 'penalty', 'recheck', 'rechecking', 'answer',
    'question', 'download', 'upload', 'login', 'password', 'roll',
    'number', 'id', 'identity', 'document', 'medical', 'illness', 'miss',
    'absent', 'absence', 'calculator', 'mobile', 'phone', 'allowed',
    'prohibited', 'seating', 'arrangement', 'duration', 'timing', 'date',
    'when', 'how', 'where', 'what', 'apply', 'application', 'contact',
    'office', 'cell', 'coordinator', 'controller', 'correction', 'error',
    'mistake', 'discrepancy', 'update', 'profile', 'email', 'account'
}
 
def is_university_related(user_input: str) -> bool:
    words = set(clean(user_input).split())
    # Check if at least one university keyword exists in query
    matched = words & UNIVERSITY_KEYWORDS
    return len(matched) > 0
 
# 6. Keyword Boost

STOPWORDS = {
    'what','when','where','how','why','who','is','the','a','an','i',
    'my','can','do','did','are','in','of','for','to','get','me','will',
    'be','it','if','on','at','by','up','as','or','and','this','that',
    'with','from','have','has','had','was','were','been','about','should'
}
 
def keyword_boost(user_input: str, questions: list) -> np.ndarray:
    user_words = set(clean(user_input).split()) - STOPWORDS
    boost      = np.zeros(len(questions))
    if not user_words:
        return boost
    for i, q in enumerate(questions):
        q_words = set(clean(q).split()) - STOPWORDS
        overlap = user_words & q_words
        if overlap:
            boost[i] = len(overlap) / max(len(user_words), 1) * 0.10
    return boost

# 7. Main Response Function

SORRY_MESSAGE = (
    "Sorry, I could not find relevant information for your question. "
    "Please rephrase or contact the university administration."
)
 
def chatbot_response(user_input, questions, answers, model, q_embeddings, a_embeddings):
    if not user_input or len(user_input.strip()) < 3:
        return "Please ask a complete question."
 
    # ── Step 1: Topic Guard ──────────────────
    # Reject completely unrelated questions immediately
    if not is_university_related(user_input):
        return SORRY_MESSAGE
 
    # ── Step 2: Semantic Matching ────────────
    user_embedding = model.encode(user_input, convert_to_tensor=True)
 
    # Score against questions (primary match)
    q_scores = util.cos_sim(user_embedding, q_embeddings)[0].cpu().numpy()
 
    # Score against answers (secondary match)
    a_scores = util.cos_sim(user_embedding, a_embeddings)[0].cpu().numpy()
 
    # Keyword boost
    boost = keyword_boost(user_input, questions)
 
    # Final combined score
    final_scores = (0.75 * q_scores) + (0.15 * a_scores) + boost
 
    best_idx   = int(np.argmax(final_scores))
    best_score = final_scores[best_idx]
    q_score    = q_scores[best_idx]
 
    # ── Step 3: Strict Double Threshold ──────
    # BOTH combined score AND raw question score
    # must be high enough to return an answer
    COMBINED_THRESHOLD = 0.40   # combined score must be > 0.40
    QUESTION_THRESHOLD = 0.35   # raw question match must be > 0.35
 
    if best_score >= COMBINED_THRESHOLD and q_score >= QUESTION_THRESHOLD:
        return answers[best_idx]
 
    # If scores are too low → question is too different from stored data
    return SORRY_MESSAGE
    
# 7. Streamlit UI

st.set_page_config(page_title="University AI Chatbot", layout="centered")
st.title("🎓 University Information Chatbot")
st.write("Ask anything about exams, results, attendance, fees, and more!")
 
# Load everything
questions, answers         = load_data()
model                      = load_model()
q_embeddings, a_embeddings = get_embeddings(model, questions, answers)
 
# Chat input
user_question = st.text_input(
    "Your Question:",
    placeholder="e.g. When will my results come out?"
)
 
if user_question:
    with st.spinner("Finding best answer..."):
        response = chatbot_response(
            user_question, questions, answers,
            model, q_embeddings, a_embeddings
        )
    st.markdown(f"**🤖 Chatbot:** {response}")
 
# Sample questions
st.markdown("---")
st.markdown("💡 **Try asking in any way you like:**")
samples = [
    "Tell me about exam dates",
    "How do I get my hall ticket?",
    "What score do I need to pass?",
    "Is phone allowed in exam hall?",
    "My marksheet has wrong marks",
    "How many days for duplicate marksheet?",
    "What if I miss too many classes?",
]
for s in samples:
    st.write(f"• {s}")
