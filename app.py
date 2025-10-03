# ==============================
# RAG Study Assistant App
# ==============================

# ------------------------------
# Load environment variables
# ------------------------------
from dotenv import load_dotenv
import os

dotenv_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(dotenv_path=dotenv_path)

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError(
        "Google API key not found. Please set GOOGLE_API_KEY in your .env file."
    )
print("Google API Key loaded successfully ✅")

# ------------------------------
# Imports
# ------------------------------
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
import streamlit as st

# ------------------------------
# Embeddings and Retriever Setup
# ------------------------------
EMBEDDINGS = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def initialize_retriever():
    vectorstore = Chroma(persist_directory="db", embedding_function=EMBEDDINGS)
    retriever = vectorstore.as_retriever()
    return retriever, vectorstore

RAG_RETRIEVER, VECTORSTORE = initialize_retriever()

# ------------------------------
# Initialize LLM
# ------------------------------
LLM_MODEL = "gpt-3.5-turbo"  # or any supported model
llm = ChatGoogleGenerativeAI(
    model=LLM_MODEL,
    temperature=0.0,
    api_key=GOOGLE_API_KEY
)

# ------------------------------
# Streamlit Interface
# ------------------------------
st.set_page_config(page_title="RAG Study Assistant", layout="wide")
st.title("📚 RAG Study Assistant")

st.write("""
Welcome! You can:
- Ask questions about your study material  
- Request a quiz on any topic  
- Go through past papers (e.g., 'Let me go through CSC231 2024 past paper')
""")

# User input
user_input = st.text_input("You:", placeholder="Type your question here...")

if st.button("Submit") and user_input:
    with st.spinner("Fetching answer..."):
        try:
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=RAG_RETRIEVER
            )
            answer = qa_chain.run(user_input)
            st.success(answer)
        except Exception as e:
            st.error(f"Error: {str(e)}")

st.write("---")
st.write("Type 'quit' in the input box to exit the conversation.")
