import streamlit as st
import pandas as pd
import anthropic
from pathlib import Path
from datetime import datetime
 
st.set_page_config(
    page_title="UniBot — University Assistant",
    page_icon="🎓",
    layout="centered"
)
 
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,300&display=swap');
 
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
 
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0a0e1a !important;
    color: #e8eaf0 !important;
}
 
#MainMenu, footer, header { visibility: hidden; }
.block-container {
    padding: 0 !important;
    max-width: 780px !important;
    margin: 0 auto !important;
}
 
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #0a0e1a 0%, #0d1525 50%, #0a0e1a 100%) !important;
    min-height: 100vh;
}
 
.hero {
    text-align: center;
    padding: 3rem 2rem 1.5rem;
    position: relative;
}
.hero::before {
    content: '';
    position: absolute;
    top: 0; left: 50%;
    transform: translateX(-50%);
    width: 600px; height: 300px;
    background: radial-gradient(ellipse, rgba(99,102,241,0.15) 0%, transparent 70%);
    pointer-events: none;
}
.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(99,102,241,0.12);
    border: 1px solid rgba(99,102,241,0.3);
    border-radius: 100px;
    padding: 6px 16px;
    font-size: 0.75rem;
    font-weight: 500;
    color: #a5b4fc;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    margin-bottom: 1.2rem;
}
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: clamp(2rem, 5vw, 3rem);
    font-weight: 800;
    line-height: 1.1;
    background: linear-gradient(135deg, #ffffff 0%, #a5b4fc 50%, #818cf8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.75rem;
}
.hero-sub {
    font-size: 1rem;
    color: #6b7280;
    font-weight: 300;
    max-width: 420px;
    margin: 0 auto 2rem;
    line-height: 1.6;
}
 
.chat-container {
    padding: 0 1.5rem;
    display: flex;
    flex-direction: column;
    gap: 1rem;
    margin-bottom: 1.5rem;
    min-height: 200px;
}
 
.msg-row {
    display: flex;
    gap: 10px;
    animation: fadeSlideIn 0.3s ease forwards;
}
@keyframes fadeSlideIn {
    from { opacity: 0; transform: translateY(10px); }
    to   { opacity: 1; transform: translateY(0); }
}
.msg-row.user { flex-direction: row-reverse; }
 
.avatar {
    width: 36px; height: 36px;
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 1rem;
    flex-shrink: 0;
    margin-top: 4px;
}
.avatar.bot {
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    box-shadow: 0 0 15px rgba(99,102,241,0.4);
}
.avatar.user-av {
    background: linear-gradient(135deg, #0ea5e9, #06b6d4);
}
 
.bubble {
    max-width: 78%;
    padding: 12px 16px;
    border-radius: 18px;
    font-size: 0.92rem;
    line-height: 1.6;
    position: relative;
}
.bubble.bot {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.08);
    border-top-left-radius: 4px;
    color: #e2e8f0;
}
.bubble.user {
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    border-top-right-radius: 4px;
    color: #ffffff;
}
.bubble-time {
    font-size: 0.7rem;
    color: #4b5563;
    margin-top: 4px;
    text-align: right;
}
 
.input-wrapper {
    position: sticky;
    bottom: 0;
    padding: 1rem 1.5rem 1.5rem;
    background: linear-gradient(to top, #0a0e1a 80%, transparent);
}
.input-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 16px;
    padding: 4px 4px 4px 16px;
    display: flex;
    align-items: center;
    gap: 8px;
    transition: border-color 0.2s;
}
.input-card:focus-within {
    border-color: rgba(99,102,241,0.5);
    box-shadow: 0 0 0 3px rgba(99,102,241,0.1);
}
 
[data-testid="stTextInput"] > div > div {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 0 !important;
}
[data-testid="stTextInput"] input {
    background: transparent !important;
    border: none !important;
    color: #e2e8f0 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.95rem !important;
    padding: 10px 0 !important;
    caret-color: #6366f1;
}
[data-testid="stTextInput"] input::placeholder {
    color: #4b5563 !important;
}
[data-testid="stTextInput"] input:focus {
    outline: none !important;
    box-shadow: none !important;
}
 
[data-testid="stButton"] button {
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 10px 20px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    font-size: 0.9rem !important;
    cursor: pointer !important;
    transition: all 0.2s !important;
    white-space: nowrap !important;
}
[data-testid="stButton"] button:hover {
    opacity: 0.9 !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 15px rgba(99,102,241,0.4) !important;
}
 
.chips-section {
    padding: 0 1.5rem 1rem;
}
.chips-label {
    font-size: 0.72rem;
    color: #4b5563;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 0.6rem;
    font-weight: 500;
}
.chips {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
}
.chip {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 100px;
    padding: 6px 14px;
    font-size: 0.8rem;
    color: #9ca3af;
    font-family: 'DM Sans', sans-serif;
}
 
.divider {
    height: 1px;
    background: linear-gradient(to right, transparent, rgba(255,255,255,0.06), transparent);
    margin: 0.5rem 1.5rem 1rem;
}
 
.stats-bar {
    display: flex;
    justify-content: center;
    gap: 2rem;
    padding: 0 1.5rem 1.5rem;
}
.stat { text-align: center; }
.stat-num {
    font-family: 'Syne', sans-serif;
    font-size: 1.3rem;
    font-weight: 700;
    color: #a5b4fc;
}
.stat-label {
    font-size: 0.72rem;
    color: #4b5563;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
 
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(99,102,241,0.3); border-radius: 2px; }
</style>
""", unsafe_allow_html=True)
 
# 2. Load CSV Data

@st.cache_data
def load_data():
    df = pd.read_csv('Chatbot Data.csv')
    questions = [str(q).strip() for q in df['question'].tolist()]
    answers   = [str(a).strip() for a in df['answer'].tolist()]
    return questions, answers
 
def build_context(questions, answers):
    context = ""
    for i, (q, a) in enumerate(zip(questions, answers), 1):
        context += f"Q{i}: {q}\nA{i}: {a}\n\n"
    return context
 
# 3. Claude API 

def get_answer(user_question, context):
    client = anthropic.Anthropic(api_key=st.secrets["ANTHROPIC_API_KEY"])
 
    system_prompt = f"""You are UniBot, a friendly and professional university information assistant.
You ONLY answer questions based on the Q&A data provided below.
 
STRICT RULES:
1. Only answer from the provided Q&A data. Do NOT use outside knowledge.
2. If the user's question matches or relates to any question in the data, give the EXACT corresponding answer.
3. If the question is NOT related to anything in the data, respond with EXACTLY:
   "Sorry, I could not find relevant information for your question. Please rephrase or contact the university administration."
4. Do NOT make up answers or add extra information.
5. Understand the MEANING of questions — different phrasings of the same question should give the same answer.
6. Keep answers clear, friendly, and helpful.
7. Never answer questions about location, directions, or anything not in the data.
 
UNIVERSITY Q&A DATA:
{context}"""
 
    message = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=1000,
        system=system_prompt,
        messages=[
            {"role": "user", "content": str(user_question)}
        ]
    )
    return message.content[0].text
 
# 4. Session State

if "messages" not in st.session_state:
    st.session_state.messages = []
if "input_key" not in st.session_state:
    st.session_state.input_key = 0
 
# Load data
questions, answers = load_data()
context = build_context(questions, answers)
 
# 5. Hero Header

st.markdown("""
<div class="hero">
    <div class="hero-badge">🎓 AI Powered Assistant</div>
    <div class="hero-title">University InfoBot</div>
    <div class="hero-sub">Ask me anything about exams, results, attendance, fees, and more.</div>
</div>
""", unsafe_allow_html=True)
 
st.markdown(f"""
<div class="stats-bar">
    <div class="stat">
        <div class="stat-num">{len(questions)}+</div>
        <div class="stat-label">Questions</div>
    </div>
    <div class="stat">
        <div class="stat-num">AI</div>
        <div class="stat-label">Powered</div>
    </div>
    <div class="stat">
        <div class="stat-num">24/7</div>
        <div class="stat-label">Available</div>
    </div>
</div>
<div class="divider"></div>
""", unsafe_allow_html=True)
 
# 6. Suggestion Chips

samples = [
    "When do exams start?",
    "How to get admit card?",
    "What is passing percentage?",
    "How to check results?",
    "Minimum attendance required?",
    "How to apply for revaluation?",
]
 
st.markdown('<div class="chips-section"><div class="chips-label">✦ Try asking</div><div class="chips">' +
    ''.join([f'<span class="chip">{s}</span>' for s in samples]) +
    '</div></div>', unsafe_allow_html=True)
 
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
 
# 7. Chat History

if st.session_state.messages:
    chat_html = '<div class="chat-container">'
    for msg in st.session_state.messages:
        t = msg["time"]
        if msg["role"] == "user":
            chat_html += f'''
            <div class="msg-row user">
                <div>
                    <div class="bubble user">{msg["content"]}</div>
                    <div class="bubble-time">{t}</div>
                </div>
                <div class="avatar user-av">👤</div>
            </div>'''
        else:
            chat_html += f'''
            <div class="msg-row">
                <div class="avatar bot">🤖</div>
                <div>
                    <div class="bubble bot">{msg["content"]}</div>
                    <div class="bubble-time">{t}</div>
                </div>
            </div>'''
    chat_html += '</div>'
    st.markdown(chat_html, unsafe_allow_html=True)
else:
    st.markdown("""
    <div class="chat-container" style="display:flex;align-items:center;justify-content:center;min-height:150px;">
        <div style="text-align:center;color:#374151;">
            <div style="font-size:2.5rem;margin-bottom:0.5rem;">💬</div>
            <div style="font-size:0.9rem;">Ask your first question above!</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
 
# 8. Input Area

st.markdown('<div class="input-wrapper"><div class="input-card">', unsafe_allow_html=True)
 
col1, col2 = st.columns([5, 1])
with col1:
    user_input = st.text_input(
        "question",
        placeholder="Ask anything about university...",
        label_visibility="collapsed",
        key=f"input_{st.session_state.input_key}"
    )
with col2:
    send = st.button("Send →")
 
st.markdown('</div></div>', unsafe_allow_html=True)

# 9. Handle Send

if send and user_input.strip():
    now = datetime.now().strftime("%I:%M %p")
 
    st.session_state.messages.append({
        "role": "user",
        "content": user_input.strip(),
        "time": now
    })
 
    with st.spinner("Thinking..."):
        try:
            response = get_answer(user_input.strip(), context)
        except Exception as e:
            response = f"Something went wrong. Please try again. ({str(e)})"
 
    st.session_state.messages.append({
        "role": "bot",
        "content": response,
        "time": datetime.now().strftime("%I:%M %p")
    })
 
    st.session_state.input_key += 1
    st.rerun()
