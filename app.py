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

# --- SESSION STATE INITIALIZATION ---
if "messages" not in st.session_state:
    st.session_state.messages = []

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
    st.markdown("**Architecture:** RAG with ChromaDB (Cosine)")
    st.markdown("**Model:** Gemini 3.8 Flash")
    
    turn_count = len([m for m in st.session_state.messages if m["role"] == "user"])
    st.markdown(f"**Memory State:** `{turn_count}` active turn(s)")
    
    st.divider()
    st.markdown("### ⚙️ Presentation Tools")
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# --- MAIN UI HEADER ---
st.title("🎓 Talibon Polytechnic College Admission Assistant")
st.markdown("Ask context-aware questions regarding admission procedures, document checklists, and campus guidelines based on the official student manual.")

# --- CONVERSATIONAL MEMORY HELPER ---
def format_chat_history(messages, max_turns: int = 5) -> str:
    """Formats recent session messages into a structured multi-turn transcript for LCEL chains."""
    if not messages:
        return "No previous conversation."
    recent_messages = messages[-(max_turns * 2):]
    formatted_turns = []
    for msg in recent_messages:
        speaker = "Student" if msg["role"] == "user" else "TPC-Assist"
        formatted_turns.append(f"{speaker}: {msg['content']}")
    return "\n".join(formatted_turns)

# --- RAG ENGINE INITIALIZATION ---
@st.cache_resource
def load_rag_components():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    persist_dir = "chroma_db"
    vectorstore = Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings,
        collection_metadata={"hnsw:space": "cosine"}
    )
    
    # Auto-initialize ChromaDB via automated ingestion pipeline if empty or incomplete
    try:
        chunk_count = vectorstore._collection.count()
    except Exception:
        chunk_count = 0

    if chunk_count < 20:
        try:
            from src.ingest import run_ingestion_pipeline
            vectorstore = run_ingestion_pipeline(reset_db=False)
        except Exception:
            manual_path = os.path.join("data", "extracted_enrollment_manual.md")
            if os.path.exists(manual_path):
                with open(manual_path, "r", encoding="utf-8") as f:
                    raw_text = f.read()
                from langchain_text_splitters import RecursiveCharacterTextSplitter
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
                chunks = text_splitter.split_text(raw_text)
                vectorstore.add_texts(chunks)
        
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    model_name = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
    llm = ChatGoogleGenerativeAI(model=model_name, temperature=0)
    
    # 1. History-Aware Standalone Query Reformulation Chain (Prompt 3)
    contextualize_q_template = """Given a chat history and the latest user question which might reference context in the chat history, formulate a standalone question which can be understood without the chat history. Do NOT answer the question, just reformulate it if needed and otherwise return it as is.

Chat History:
{chat_history}

Latest Question: {question}

Standalone Question:"""
    contextualize_q_prompt = ChatPromptTemplate.from_template(contextualize_q_template)
    contextualize_q_chain = contextualize_q_prompt | llm | StrOutputParser()
    
    # 2. Strict Grounded RAG Synthesis Chain with Multi-Turn Context (Prompt 1)
    qa_template = """You are an official academic assistant for Talibon Polytechnic College (TPC).
Answer the question strictly based on the provided context below.

Guidelines:
1. Do not use outside knowledge or assume policies not stated in the context.
2. Maintain a professional, polite, and helpful tone.
3. If the context does not contain sufficient information to answer the question, state clearly:
   "I cannot find this information in the official TPC manual."

Context:
{context}

Chat History:
{chat_history}

Question: {question}
"""
    qa_prompt = ChatPromptTemplate.from_template(qa_template)
    qa_chain = qa_prompt | llm | StrOutputParser()
    
    return retriever, contextualize_q_chain, qa_chain

try:
    retriever, contextualize_q_chain, rag_chain = load_rag_components()
except Exception as e:
    st.error(f"Failed to initialize RAG components: {e}")
    st.stop()

# Display previous chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message and message["sources"]:
            with st.expander("📚 View Retrieved Reference Chunks"):
                if message.get("standalone_query") and message["standalone_query"].lower() != message.get("original_query", "").lower():
                    st.caption(f"🔄 **Contextualized Search Query:** *{message['standalone_query']}*")
                for idx, doc in enumerate(message["sources"], 1):
                    st.caption(f"**Source Chunk {idx}:**\n{doc}")

# --- CHAT INPUT & EXECUTION ---
if query := st.chat_input("Type your question here..."):
    # Capture prior conversation history before appending current user message
    prior_history_str = format_chat_history(st.session_state.messages)
    has_prior_turns = len(st.session_state.messages) > 0

    # Render user query
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)
        
    # Process Assistant Response
    with st.chat_message("assistant"):
        with st.spinner("Retrieving handbook context..."):
            # 1. Reformulate follow-up question into standalone query if history exists
            if has_prior_turns:
                standalone_query = contextualize_q_chain.invoke({
                    "chat_history": prior_history_str,
                    "question": query
                }).strip()
            else:
                standalone_query = query

            # 2. Retrieve context chunks using the standalone query
            docs = retriever.invoke(standalone_query)
            context_text = "\n\n".join([d.page_content for d in docs])
            
            # 3. Generate grounded answer with conversation history and retrieved context
            response = rag_chain.invoke({
                "context": context_text,
                "chat_history": prior_history_str,
                "question": standalone_query
            })
            
            # 4. Render response
            st.markdown(response)
            
            # 5. Render Source Attribution & Reformulated Query
            extracted_sources = [d.page_content for d in docs]
            if extracted_sources:
                with st.expander("📚 View Retrieved Reference Chunks"):
                    if has_prior_turns and standalone_query.lower() != query.lower():
                        st.caption(f"🔄 **Contextualized Search Query:** *{standalone_query}*")
                    for idx, chunk in enumerate(extracted_sources, 1):
                        st.caption(f"**Source Chunk {idx}:**\n{chunk}")
                        
            # Save to session state
            st.session_state.messages.append({
                "role": "assistant", 
                "content": response,
                "sources": extracted_sources,
                "original_query": query,
                "standalone_query": standalone_query
            })
