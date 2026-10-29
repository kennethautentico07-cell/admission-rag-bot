import base64
import os
import re
import streamlit as st
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

st.set_page_config(
    page_title="TPC Office of Admissions | Policy & Enrollment Desk",
    page_icon="tpc_logo.png" if os.path.exists("tpc_logo.png") else "🎓",
    layout="centered",
    initial_sidebar_state="expanded",
)


def get_base64_logo(path: str = "tpc_logo.png") -> str:
    """Loads the official TPC seal as a base64 string for inline masthead rendering."""
    if not os.path.exists(path):
        return ""
    with open(path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")


LOGO_B64 = get_base64_logo()

# --- COLLEGIATE REGISTRAR PORTAL STYLING (Grounded in Official TPC Seal) ---
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Libre+Baskerville:ital,wght@0,400;0,700;1,400&family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');

    :root {
        --tpc-navy: #06395A;
        --tpc-navy-deep: #04263D;
        --tpc-gold: #C8961E;
        --tpc-gold-soft: #FBF6E9;
        --tpc-emerald: #1A6B41;
        --tpc-emerald-soft: #EDF7F1;
        --tpc-ink: #0B1F2E;
        --tpc-muted: #4A6070;
        --tpc-paper: #F7F8FA;
        --tpc-surface: #FFFFFF;
        --tpc-border: #DCE3EA;
    }

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
        color: var(--tpc-ink);
    }

    .stApp {
        background-color: var(--tpc-paper);
    }

    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 4rem !important;
        max-width: 840px !important;
    }

    /* Official TPC Institutional Masthead */
    .tpc-masthead {
        background: var(--tpc-surface);
        border: 1px solid var(--tpc-border);
        border-top: 4px solid var(--tpc-navy);
        border-radius: 6px;
        padding: 1.35rem 1.5rem;
        margin-bottom: 1.5rem;
        display: flex;
        align-items: center;
        gap: 1.25rem;
        box-shadow: 0 1px 3px rgba(6, 57, 90, 0.05);
        position: relative;
    }

    .tpc-masthead::after {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 2px;
        background: var(--tpc-gold);
    }

    .tpc-masthead-seal {
        width: 72px;
        height: 72px;
        flex-shrink: 0;
        object-fit: contain;
    }

    .tpc-masthead-body {
        flex: 1;
        min-width: 0;
    }

    .tpc-institution-name {
        font-family: 'Libre Baskerville', Georgia, serif;
        font-size: 1.35rem;
        font-weight: 700;
        color: var(--tpc-navy);
        line-height: 1.25;
        margin: 0 0 0.2rem 0;
        letter-spacing: -0.01em;
    }

    .tpc-office-subtitle {
        font-size: 0.92rem;
        font-weight: 600;
        color: var(--tpc-ink);
        margin: 0 0 0.35rem 0;
    }

    .tpc-masthead-meta {
        font-size: 0.8rem;
        color: var(--tpc-muted);
        margin: 0;
        line-height: 1.4;
    }

    /* Empty-State Registrar Desk Introduction */
    .tpc-desk-notice {
        background: var(--tpc-surface);
        border: 1px solid var(--tpc-border);
        border-left: 3px solid var(--tpc-gold);
        border-radius: 6px;
        padding: 1rem 1.25rem;
        margin-bottom: 1.25rem;
        font-size: 0.9rem;
        line-height: 1.55;
        color: var(--tpc-ink);
    }

    .tpc-desk-notice strong {
        color: var(--tpc-navy);
    }

    .tpc-section-heading {
        font-family: 'Libre Baskerville', Georgia, serif;
        font-size: 0.98rem;
        font-weight: 700;
        color: var(--tpc-navy);
        margin: 0.5rem 0 0.65rem 0;
    }

    /* Quick-Select Inquiry Buttons */
    .stButton > button {
        background-color: var(--tpc-surface) !important;
        color: var(--tpc-ink) !important;
        border: 1px solid var(--tpc-border) !important;
        border-radius: 6px !important;
        padding: 0.65rem 0.9rem !important;
        font-size: 0.86rem !important;
        font-weight: 500 !important;
        text-align: left !important;
        justify-content: flex-start !important;
        transition: border-color 0.15s ease, background-color 0.15s ease !important;
        box-shadow: none !important;
    }

    .stButton > button:hover {
        border-color: var(--tpc-navy) !important;
        background-color: #F0F5F9 !important;
        color: var(--tpc-navy) !important;
    }

    .stButton > button:focus-visible {
        outline: 2px solid var(--tpc-navy) !important;
        outline-offset: 2px !important;
    }

    /* Chat Message Containers */
    [data-testid="stChatMessage"] {
        background-color: var(--tpc-surface);
        border: 1px solid var(--tpc-border);
        border-radius: 6px;
        padding: 1rem 1.15rem;
        margin-bottom: 0.85rem;
    }

    [data-testid="stChatMessage"] p,
    [data-testid="stChatMessage"] li {
        font-size: 0.94rem;
        line-height: 1.65;
        max-width: 72ch;
    }

    /* Source Excerpt Cards inside Expander */
    .tpc-source-card {
        background-color: var(--tpc-paper);
        border: 1px solid var(--tpc-border);
        border-left: 3px solid var(--tpc-emerald);
        border-radius: 4px;
        padding: 0.75rem 0.95rem;
        margin-top: 0.55rem;
        font-size: 0.84rem;
        line-height: 1.55;
        color: var(--tpc-ink);
    }

    .tpc-source-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0.35rem;
        font-weight: 600;
        font-size: 0.78rem;
        color: var(--tpc-emerald);
    }

    .tpc-reformulated-banner {
        background-color: var(--tpc-gold-soft);
        border: 1px solid #E6D095;
        border-radius: 4px;
        padding: 0.5rem 0.75rem;
        margin-bottom: 0.5rem;
        font-size: 0.81rem;
        color: #5C4308;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: var(--tpc-surface) !important;
        border-right: 1px solid var(--tpc-border) !important;
    }

    .tpc-sidebar-seal-wrap {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        padding-bottom: 0.75rem;
        border-bottom: 2px solid var(--tpc-navy);
        margin-bottom: 0.85rem;
    }

    .tpc-sidebar-seal {
        width: 48px;
        height: 48px;
        object-fit: contain;
    }

    .tpc-sidebar-title {
        font-family: 'Libre Baskerville', Georgia, serif;
        font-size: 0.95rem;
        font-weight: 700;
        color: var(--tpc-navy);
        margin: 0;
        line-height: 1.25;
    }

    .tpc-sidebar-sub {
        font-size: 0.75rem;
        color: var(--tpc-muted);
        margin: 0.15rem 0 0 0;
    }

    .tpc-ledger-box {
        background: var(--tpc-paper);
        border: 1px solid var(--tpc-border);
        border-radius: 5px;
        padding: 0.7rem 0.85rem;
        margin-bottom: 0.85rem;
        font-size: 0.8rem;
        line-height: 1.5;
    }

    .tpc-ledger-box strong {
        color: var(--tpc-navy);
        display: block;
        margin-bottom: 0.25rem;
        font-size: 0.82rem;
    }

    @media (max-width: 640px) {
        .tpc-masthead {
            flex-direction: column;
            align-items: flex-start;
            padding: 1.1rem;
        }
        .tpc-masthead-seal {
            width: 56px;
            height: 56px;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- SESSION STATE INITIALIZATION ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_query" not in st.session_state:
    st.session_state.pending_query = None


def parse_chunk_for_display(raw_chunk: str, index: int) -> tuple[str, str]:
    """Extracts the handbook page header and clean body text from a retrieved chunk."""
    page_match = re.search(r"---\s*Page\s+([^\n-]+?)\s*---", raw_chunk)
    if page_match:
        page_label = f"Student Handbook • Page {page_match.group(1).strip()}"
        cleaned_body = re.sub(r"---\s*Page\s+[^\n-]+?\s*---\s*", "", raw_chunk).strip()
    else:
        page_label = f"Student Handbook Reference #{index}"
        cleaned_body = raw_chunk.strip()
    return page_label, cleaned_body


# --- SIDEBAR: Official Registrar Ledger & Controls ---
with st.sidebar:
    if LOGO_B64:
        st.markdown(
            f"""
            <div class="tpc-sidebar-seal-wrap">
                <img src="data:image/png;base64,{LOGO_B64}" class="tpc-sidebar-seal" alt="TPC Official Seal" />
                <div>
                    <p class="tpc-sidebar-title">Talibon Polytechnic College</p>
                    <p class="tpc-sidebar-sub">San Jose, Talibon, Bohol • Est. 2017</p>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown("### Talibon Polytechnic College")

    turn_count = len([m for m in st.session_state.messages if m["role"] == "user"])

    st.markdown(
        f"""
        <div class="tpc-ledger-box">
            <strong>Active Advisory Session</strong>
            Conversation Memory: <b>{turn_count} turn(s)</b><br>
            Knowledge Base: <b>TPC Handbook (26 Pages)</b><br>
            Vector Index: <b>ChromaDB (38 Cosine Chunks)</b><br>
            Synthesis Model: <b>Gemini 3.8 Flash</b>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.button("Reset Inquiry Session", use_container_width=True):
        st.session_state.messages = []
        st.session_state.pending_query = None
        st.rerun()

    st.markdown(
        """
        <div class="tpc-ledger-box" style="margin-top: 0.85rem;">
            <strong>5-Step Enrollment Sequence</strong>
            1. Document Verification (Registrar Window 1)<br>
            2. Admission Test (Guidance Center)<br>
            3. Academic Advising (Dean's Office)<br>
            4. Fee Assessment (Cashier's Office)<br>
            5. COR Validation (Registrar's Office)
        </div>
        <div class="tpc-ledger-box">
            <strong>CHED-Recognized Programs</strong>
            • BS Information Systems (BSIS)<br>
            • BS Accounting Info Systems (BSAIS)<br>
            • BS Agriculture (BSA)<br>
            • BS Criminology (BSCrim)<br>
            • Bachelor of Early Childhood Ed (BECEd)<br>
            • BA English Language<br>
            • BA Political Science
        </div>
        """,
        unsafe_allow_html=True,
    )

# --- MAIN INSTITUTIONAL MASTHEAD ---
seal_html = (
    f'<img src="data:image/png;base64,{LOGO_B64}" class="tpc-masthead-seal" alt="Talibon Polytechnic College Seal" />'
    if LOGO_B64
    else ""
)
st.markdown(
    f"""
    <header class="tpc-masthead">
        {seal_html}
        <div class="tpc-masthead-body">
            <h1 class="tpc-institution-name">Talibon Polytechnic College</h1>
            <p class="tpc-office-subtitle">Office of Admissions &amp; the College Registrar — Policy Advisory Desk</p>
            <p class="tpc-masthead-meta">
                Grounded in the Official TPC Student Handbook (Municipal Ordinance No. 2017-07) • Answers cite verified handbook sections
            </p>
        </div>
    </header>
    """,
    unsafe_allow_html=True,
)


# --- CONVERSATIONAL MEMORY HELPER ---
def format_chat_history(messages, max_turns: int = 5) -> str:
    """Formats recent session messages into a structured multi-turn transcript for LCEL chains."""
    if not messages:
        return "No previous conversation."
    recent_messages = messages[-(max_turns * 2) :]
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
        collection_metadata={"hnsw:space": "cosine"},
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
    st.error(f"Unable to initialize the handbook retrieval service: {e}")
    st.stop()


def render_source_citations(sources: list[str], original_query: str = "", standalone_query: str = ""):
    """Renders retrieved handbook chunks as structured Registrar reference cards."""
    if not sources:
        return
    with st.expander(f"Verify in Official TPC Student Handbook ({len(sources)} excerpts retrieved)"):
        if standalone_query and original_query and standalone_query.strip().lower() != original_query.strip().lower():
            st.markdown(
                f"""
                <div class="tpc-reformulated-banner">
                    <strong>Multi-Turn Context Applied:</strong> Follow-up question interpreted as
                    <em>"{standalone_query}"</em>
                </div>
                """,
                unsafe_allow_html=True,
            )
        for idx, raw_chunk in enumerate(sources, 1):
            page_label, body_text = parse_chunk_for_display(raw_chunk, idx)
            safe_body = body_text.replace("\n", "<br>")
            st.markdown(
                f"""
                <div class="tpc-source-card">
                    <div class="tpc-source-header">
                        <span>{page_label}</span>
                        <span>Excerpt #{idx}</span>
                    </div>
                    <div>{safe_body}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


# --- EMPTY STATE: ADMISSIONS QUICK-INQUIRY DIRECTORY ---
if not st.session_state.messages:
    st.markdown(
        """
        <div class="tpc-desk-notice">
            <strong>Welcome to the TPC Admissions &amp; Policy Desk.</strong>
            Select a common enrollment topic below or type a specific question. You can ask follow-up questions in the same session, and every response includes the exact handbook excerpts used.
        </div>
        <p class="tpc-section-heading">Common Registrar &amp; Admission Inquiries</p>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)
    with col1:
        if st.button("What documents do incoming freshmen need to submit?", use_container_width=True):
            st.session_state.pending_query = "What documents do incoming freshmen need to submit?"
            st.rerun()
        if st.button("What degree programs and courses are offered at TPC?", use_container_width=True):
            st.session_state.pending_query = "What degree programs and courses are offered at TPC?"
            st.rerun()
    with col2:
        if st.button("What are the admission requirements for transferees?", use_container_width=True):
            st.session_state.pending_query = "What are the admission requirements for transferees?"
            st.rerun()
        if st.button("Walk me through the 5-step enrollment workflow.", use_container_width=True):
            st.session_state.pending_query = "Walk me through the 5-step enrollment workflow."
            st.rerun()

# --- DISPLAY PREVIOUS CONVERSATION TURNS ---
for message in st.session_state.messages:
    avatar = "🎓" if message["role"] == "assistant" else None
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])
        if message["role"] == "assistant" and message.get("sources"):
            render_source_citations(
                sources=message["sources"],
                original_query=message.get("original_query", ""),
                standalone_query=message.get("standalone_query", ""),
            )

# --- CHAT INPUT & EXECUTION ---
chat_box_query = st.chat_input("Ask about freshman requirements, transferee documents, academic load, or enrollment steps...")
active_query = st.session_state.pending_query or chat_box_query
st.session_state.pending_query = None

if active_query:
    prior_history_str = format_chat_history(st.session_state.messages)
    has_prior_turns = len(st.session_state.messages) > 0

    st.session_state.messages.append({"role": "user", "content": active_query})
    with st.chat_message("user"):
        st.markdown(active_query)

    with st.chat_message("assistant", avatar="🎓"):
        with st.spinner("Consulting official TPC Student Handbook..."):
            # 1. Reformulate follow-up question into standalone query if history exists
            if has_prior_turns:
                standalone_query = contextualize_q_chain.invoke(
                    {
                        "chat_history": prior_history_str,
                        "question": active_query,
                    }
                ).strip()
            else:
                standalone_query = active_query

            # 2. Retrieve context chunks using the standalone query
            docs = retriever.invoke(standalone_query)
            context_text = "\n\n".join([d.page_content for d in docs])

            # 3. Generate grounded answer with conversation history and retrieved context
            response = rag_chain.invoke(
                {
                    "context": context_text,
                    "chat_history": prior_history_str,
                    "question": standalone_query,
                }
            )

            # 4. Render response
            st.markdown(response)

            # 5. Render structured source cards
            extracted_sources = [d.page_content for d in docs]
            render_source_citations(
                sources=extracted_sources,
                original_query=active_query,
                standalone_query=standalone_query if has_prior_turns else "",
            )

            # Save turn to session state
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": response,
                    "sources": extracted_sources,
                    "original_query": active_query,
                    "standalone_query": standalone_query,
                }
            )
            st.rerun()
