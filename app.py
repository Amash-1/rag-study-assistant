# ==============================
# RAG Study Assistant - app.py
# ==============================

import streamlit as st
import os
from dotenv import load_dotenv

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma

# ==============================
# Load environment variables
# ==============================
load_dotenv()  # Reads .env file

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("❌ Google API key not found. Please set GOOGLE_API_KEY in your .env file.")

# ==============================
# Streamlit UI setup
# ==============================
st.set_page_config(page_title="📚 RAG Study Assistant", layout="wide")
st.title("📚 RAG Study Assistant")
st.markdown("Ask questions about your study material!")

# ==============================
# Initialize embeddings (force CPU on Streamlit Cloud)
# ==============================
EMBEDDINGS = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"}  # ✅ Force CPU so it works on Streamlit Cloud
)

# ==============================
# Initialize retriever
# ==============================
VECTORSTORE = Chroma(persist_directory="db", embedding_function=EMBEDDINGS)
RAG_RETRIEVER = VECTORSTORE.as_retriever()

# ==============================
# Initialize LLM
# ==============================
LLM_MODEL = "gemini-pro"  # You can switch to another Google model if needed
llm = ChatGoogleGenerativeAI(
    model=LLM_MODEL,
    temperature=0.0,
    api_key=GOOGLE_API_KEY
)

# ==============================
# Streamlit interaction
# ==============================
user_input = st.text_input("You:")

if user_input:
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=RAG_RETRIEVER
    )
    with st.spinner("Generating answer..."):
        answer = chain.run(user_input)
    st.markdown(f"**Assistant:** {answer}")

# ==============================
# Optional debug info
# ==============================
if st.checkbox("Show debug info"):
    st.write("✅ Google API Key Loaded:", bool(GOOGLE_API_KEY))
    st.write("📂 Retriever info:", RAG_RETRIEVER)
