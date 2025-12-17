import pyodbc
import streamlit as st

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








