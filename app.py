# ==============================
# RAG Study Assistant - app.py
# ==============================

import os
import streamlit as st
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv

# ==============================
# Load environment variables from .env (optional)
# ==============================
load_dotenv()  # loads .env if it exists
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") or "AIzaSyDL-ASniJ7-61zLWnPccGGt3mFxvQjvWQ4"

if not GOOGLE_API_KEY:
    raise ValueError(
        "Google API key not found. Please set GOOGLE_API_KEY in your .env file "
        "or directly in this script."
    )

# ==============================
# Streamlit UI setup
# ==============================
st.set_page_config(page_title="📚 RAG Study Assistant", layout="wide")
st.title("📚 RAG Study Assistant")
st.markdown("Ask questions about your study material!")

# ==============================
# Initialize embeddings
# ==============================
EMBEDDINGS = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# ==============================
# Initialize vectorstore and retriever
# ==============================
VECTORSTORE = Chroma(persist_directory="db", embedding_function=EMBEDDINGS)
RAG_RETRIEVER = VECTORSTORE.as_retriever()

# ==============================
# Initialize LLM
# ==============================
LLM_MODEL = "gpt-4"  # change if needed
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
    st.write("Google API Key Loaded:", GOOGLE_API_KEY)
    st.write("Retriever info:", RAG_RETRIEVER)
