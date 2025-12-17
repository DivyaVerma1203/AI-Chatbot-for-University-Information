import streamlit as st
from sqlalchemy import create_engine

st.title("University Information Chatbot")

server = st.secrets["DB_SERVER"]
database = st.secrets["DB_NAME"]
username = st.secrets["DB_USER"]
password = st.secrets["DB_PASSWORD"]

engine = create_engine(
    f"mssql+pytds://{username}:{password}@{server}:1433/{database}"
)

try:
    with engine.connect() as conn:
        result = conn.execute("SELECT 1")
        st.success("Database connected successfully")
except Exception as e:
    st.error(e)






