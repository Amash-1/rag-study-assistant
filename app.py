# ==============================
# RAG Study Assistant - app.py
# ==============================

import os
from dotenv import load_dotenv
import streamlit as st

# LangChain imports
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma

# ==============================
# Load environment variables
# ==============================
dotenv_path = os.path.join(os.path.dirname(__file__), ".env")
if not load_dotenv(dotenv_path=dotenv_path):
    st.warning(".env file not found. Make sure it exists in the same folder as app.py")

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("Google API key not found. Please set GOOGLE_API_KEY in your .env file.")

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
# Initialize retriever
# ==============================
VECTORSTORE = Chroma(persist_directory="db", embedding_function=EMBEDDINGS)
RAG_RETRIEVER = VECTORSTORE.as_retriever()

# ==============================
# Initialize LLM
# ==============================
LLM_MODEL = "gpt-4"  # or any other model you want
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
# Optional: Show past queries or logs
# ==============================
if st.checkbox("Show debug info"):
    st.write("Google API Key Loaded:", GOOGLE_API_KEY)
    st.write("Retriever info:", RAG_RETRIEVER)
