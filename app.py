import os
import streamlit as st
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

st.set_page_config(page_title="TPC-Assist | RAG Chatbot", page_icon="🎓", layout="centered")

# --- MINIMALIST GREEN THEME STYLING ---
st.markdown(
    """
    <style>
    /* Minimalist Green Palette */
    :root {
        --primary-green: #1b4d3e;
        --accent-green: #2e7d32;
        --soft-green-bg: #f4f8f5;
    }
    
    /* Header Accent */
    h1 {
        color: #2e7d32 !important;
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        border-right: 1px solid rgba(46, 125, 50, 0.2);
    }
    
    /* Clear Button Customization */
    .stButton > button {
        border-color: var(--accent-green) !important;
        color: var(--accent-green) !important;
        border-radius: 8px;
    }
    .stButton > button:hover {
        background-color: var(--accent-green) !important;
        color: white !important;
    }
    
    /* Chat Expander Border Accent */
    .streamlit-expanderHeader {
        color: var(--accent-green) !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- SIDEBAR: Branding, Info & Controls ---
with st.sidebar:
    # Constrained logo width to prevent over-stretching
    if os.path.exists("tpc_logo.png"):
        col_a, col_b, col_c = st.columns([1, 2, 1])
        with col_b:
            st.image("tpc_logo.png", width=120)
        
    st.title("🎓 TPC-Assist")
    st.caption("RAG-Powered Admission & Policy Assistant")
    st.divider()
    
    st.markdown("### 📌 Project Metadata")
    st.markdown("**Course:** IS Elective 4 (GenAI Systems)")
    st.markdown("**Institution:** Talibon Polytechnic College")
    st.markdown("**Architecture:** RAG with ChromaDB")
    st.markdown("**Model:** Gemini 3.6 Flash")
    
    st.divider()
    st.markdown("### ⚙️ Presentation Tools")
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# --- MAIN UI HEADER ---
st.title("🎓 Talibon Polytechnic College Admission Assistant")
st.markdown("Ask context-aware questions regarding admission procedures, document checklists, and campus guidelines based on the official student manual.")

# --- RAG ENGINE INITIALIZATION ---
@st.cache_resource
def load_rag_components():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma(
        persist_directory="chroma_db",
        embedding_function=embeddings
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash", temperature=0)
    
    template = """Answer the question strictly based on the provided context below. If the answer cannot be found in the manual, state clearly: "I cannot find this information in the official TPC manual."

Context:
{context}

Question: {question}
"""
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | llm | StrOutputParser()
    
    return retriever, chain

try:
    retriever, rag_chain = load_rag_components()
except Exception as e:
    st.error(f"Failed to initialize RAG components: {e}")
    st.stop()

# --- SESSION STATE & CHAT HISTORY ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message and message["sources"]:
            with st.expander("📚 View Retrieved Reference Chunks"):
                for idx, doc in enumerate(message["sources"], 1):
                    st.caption(f"**Source Chunk {idx}:**\n{doc}")

# --- CHAT INPUT & EXECUTION ---
if query := st.chat_input("Type your question here..."):
    # Render user query
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)
        
    # Process Assistant Response
    with st.chat_message("assistant"):
        with st.spinner("Retrieving handbook context..."):
            # 1. Retrieve context chunks
            docs = retriever.invoke(query)
            context_text = "\n\n".join([d.page_content for d in docs])
            
            # 2. Generate grounded answer
            response = rag_chain.invoke({"context": context_text, "question": query})
            
            # 3. Render response
            st.markdown(response)
            
            # 4. Render Source Attribution
            extracted_sources = [d.page_content for d in docs]
            if extracted_sources:
                with st.expander("📚 View Retrieved Reference Chunks"):
                    for idx, chunk in enumerate(extracted_sources, 1):
                        st.caption(f"**Source Chunk {idx}:**\n{chunk}")
                        
            # Save to session
            st.session_state.messages.append({
                "role": "assistant", 
                "content": response,
                "sources": extracted_sources
            })