import streamlit as st
from dotenv import load_dotenv
import os

from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings


# ==============================
# Load environment variables
# ==============================
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    st.error("❌ Google API key not found. Please set GOOGLE_API_KEY in .env")
    st.stop()

# ==============================
# Initialize model + embeddings
# ==============================
st.set_page_config(page_title="RAG Study Assistant", page_icon="📘")
st.title("📘 RAG Study Assistant")
st.write("Ask me anything about your study materials, generate quizzes, or explore past papers!")

# Embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Load vector DB
persist_directory = "db"
vectordb = Chroma(persist_directory=persist_directory, embedding_function=embeddings)

# Retriever
retriever = vectordb.as_retriever(search_kwargs={"k": 3})

# LLM
llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=GOOGLE_API_KEY)

# QA Chain
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# ==============================
# Streamlit UI
# ==============================

# Tabs for features
tab1, tab2, tab3 = st.tabs(["💬 Chat Assistant", "📝 Quiz Generator", "📂 Past Papers"])

# --- Chat Assistant ---
with tab1:
    st.subheader("💬 Chat with your study assistant")
    user_question = st.text_input("Type your question here:")
    if user_question:
        response = qa_chain.run(user_question)
        st.markdown("### 🤖 Assistant’s Answer:")
        st.write(response)

# --- Quiz Generator ---
with tab2:
    st.subheader("📝 Generate a Quiz")
    topic = st.text_input("Enter a topic for your quiz:")
    if st.button("Generate Quiz"):
        if topic:
            quiz_prompt = f"Create a quiz with 5 multiple-choice questions about {topic}. Include correct answers."
            quiz = qa_chain.run(quiz_prompt)
            st.markdown("### 📝 Quiz")
            st.write(quiz)
        else:
            st.warning("Please enter a topic.")

# --- Past Papers ---
with tab3:
    st.subheader("📂 Past Papers Helper")
    paper_code = st.text_input("Enter your course code (e.g., CSC231 2024):")
    if st.button("Go through Past Paper"):
        if paper_code:
            past_paper_prompt = f"Go through {paper_code} past paper and explain answers clearly."
            past_paper_response = qa_chain.run(past_paper_prompt)
            st.markdown("### 📂 Past Paper Guidance")
            st.write(past_paper_response)
        else:
            st.warning("Please enter a course code.")

