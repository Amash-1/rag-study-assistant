# ==============================
# app.py – RAG Study Assistant
# ==============================

# ---------- Load environment variables ----------
from dotenv import load_dotenv
import os

# Explicitly load .env file from current folder
dotenv_path = os.path.join(os.path.dirname(__file__), ".env")
loaded = load_dotenv(dotenv_path=dotenv_path)
print(f".env loaded: {loaded}")

# Get Google API key
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
print(f"GOOGLE_API_KEY: {GOOGLE_API_KEY}")

if not GOOGLE_API_KEY:
    raise ValueError(
        "Google API key not found. Please set GOOGLE_API_KEY in your .env file."
    )

# ---------- Imports ----------
import streamlit as st
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma

# ---------- Constants ----------
LLM_MODEL = "gpt-4"  # or any other model you want

# ---------- Initialize embeddings ----------
EMBEDDINGS = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# ---------- Initialize RAG retriever ----------
def initialize_retriever():
    VECTORSTORE = Chroma(persist_directory="db", embedding_function=EMBEDDINGS)
    RETRIEVER = VECTORSTORE.as_retriever()
    return RETRIEVER, VECTORSTORE

RAG_RETRIEVER, VECTORSTORE = initialize_retriever()

# ---------- Initialize LLM ----------
llm = ChatGoogleGenerativeAI(
    model=LLM_MODEL,
    temperature=0.0,
    api_key=GOOGLE_API_KEY
)

# ---------- Streamlit App ----------
st.set_page_config(page_title="RAG Study Assistant", layout="wide")
st.title("📚 RAG Study Assistant")

st.write("Ask questions about your study material or request a quiz!")

user_input = st.text_input("You:")

if user_input:
    # Query the retriever
    retriever_result = RAG_RETRIEVER.get_relevant_documents(user_input)
    
    # Combine retriever results and generate answer
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=RAG_RETRIEVER
    )
    answer = chain.run(user_input)
    
    st.write("**Assistant:**", answer)
