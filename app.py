import base64
import os
import re
import time
from typing import List, Optional

import torch
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

# Prevent CPU thread thrashing on cloud container vCPUs (Railway / Docker)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
try:
    torch.set_num_threads(2)
except Exception:
    pass

app = FastAPI(
    title="TPC-Assist Admission Portal",
    description="RAG-Based Admission & Policy Chatbot for Talibon Polytechnic College",
    version="2.1.0",
)


def get_base64_logo(path: str = "tpc_logo.png") -> str:
    """Loads the official TPC seal as a base64 data URI."""
    if not os.path.exists(path):
        return ""
    with open(path, "rb") as f:
        return "data:image/png;base64," + base64.b64encode(f.read()).decode("utf-8")


LOGO_DATA_URI = get_base64_logo()

# --- GLOBAL RAG ENGINE SINGLETON ---
_rag_components = None


def get_rag_components():
    global _rag_components
    if _rag_components is not None:
        return _rag_components

    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    # Pre-warm the PyTorch embedding model in memory so first user query is instant
    embeddings.embed_query("Talibon Polytechnic College admission requirements")

    persist_dir = "chroma_db"
    vectorstore = Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings,
        collection_metadata={"hnsw:space": "cosine"},
    )

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

                splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
                chunks = splitter.split_text(raw_text)
                vectorstore.add_texts(chunks)

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    model_name = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")

    # Configure Gemini for low-latency grounded synthesis (single call per turn, no long retry stalls)
    llm = ChatGoogleGenerativeAI(
        model=model_name,
        temperature=0,
        max_output_tokens=512,
        timeout=25,
        max_retries=1,
    )

    # Strict Grounded RAG Synthesis Chain with Multi-Turn Context (Prompt 1)
    qa_template = """You are an official academic assistant for Talibon Polytechnic College (TPC).
Answer the question strictly based on the provided context below.

Guidelines:
1. Do not use outside knowledge or assume policies not stated in the context.
2. Maintain a professional, polite, and concise tone.
3. If the question is a follow-up referring to earlier turns in the Chat History, resolve it using the Chat History and answer strictly from the Context.
4. If the context does not contain sufficient information to answer the question, state clearly:
   "I cannot find this information in the official TPC manual."

Context:
{context}

Chat History:
{chat_history}

Question: {question}
"""
    qa_prompt = ChatPromptTemplate.from_template(qa_template)
    qa_chain = qa_prompt | llm | StrOutputParser()

    _rag_components = (retriever, qa_chain)
    return _rag_components


class ChatTurn(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    question: str
    history: Optional[List[ChatTurn]] = []


FOLLOWUP_PRONOUN_PATTERN = re.compile(
    r"\b(it|they|them|their|theirs|that|those|these|this|there|he|she|also|too|what about|how about|where do i|when do i|how do i submit)\b",
    re.IGNORECASE,
)


def build_contextualized_query(question: str, history: List[ChatTurn]) -> tuple[str, bool]:
    """
    Instant (<1ms) history-aware query contextualizer.
    Resolves follow-up questions with pronouns or implicit references against recent user turns
    without making a second blocking LLM API call (preventing 429 rate-limit delays).
    """
    if not history:
        return question, False

    user_turns = [t.content.strip() for t in history if t.role == "user" and t.content.strip()]
    if not user_turns:
        return question, False

    last_user_topic = user_turns[-1]
    word_count = len(question.split())
    has_followup_signal = bool(FOLLOWUP_PRONOUN_PATTERN.search(question)) or word_count <= 6

    if not has_followup_signal:
        return question, False

    # Construct standalone retrieval query anchored by the previous inquiry context
    standalone = f"{question} (regarding: {last_user_topic})"
    return standalone, True


def format_chat_history(history: List[ChatTurn], max_turns: int = 4) -> str:
    if not history:
        return "No previous conversation."
    recent = history[-(max_turns * 2) :]
    lines = []
    for turn in recent:
        speaker = "Student" if turn.role == "user" else "TPC-Assist"
        lines.append(f"{speaker}: {turn.content}")
    return "\n".join(lines)


def parse_source_chunk(raw_chunk: str, idx: int) -> dict:
    """Extracts page number and section heading from a retrieved handbook chunk."""
    page_match = re.search(r"---\s*Page\s+([^\n-]+?)\s*---", raw_chunk)
    page_str = f"Handbook Page {page_match.group(1).strip()}" if page_match else f"Handbook Excerpt #{idx}"
    cleaned = re.sub(r"---\s*Page\s+[^\n-]+?\s*---\s*", "", raw_chunk).strip()

    first_line = cleaned.split("\n")[0].strip().lstrip("#").strip()
    if len(first_line) < 65 and "\n" in cleaned:
        title = f"{page_str}: {first_line}"
        body = "\n".join(cleaned.split("\n")[1:]).strip()
    else:
        title = f"{page_str}: Academic Policies & Admission Guidelines"
        body = cleaned

    return {
        "index": idx,
        "title": title,
        "page": page_str,
        "content": body or cleaned,
    }


@app.on_event("startup")
def startup_event():
    try:
        get_rag_components()
    except Exception as e:
        print(f"[WARN] Deferred RAG initialization: {e}")


@app.get("/health")
@app.get("/_stcore/health")
def health_check():
    return JSONResponse({"status": "ok", "service": "tpc-assist-rag"})


@app.post("/api/chat")
def chat_endpoint(req: ChatRequest):
    start_time = time.perf_counter()
    question = (req.question or "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        retriever, qa_chain = get_rag_components()
        history_list = req.history or []
        history_str = format_chat_history(history_list)

        # 1. Instant history-aware query contextualization (<1ms, 0 extra API calls)
        standalone_query, was_reformulated = build_contextualized_query(question, history_list)

        # 2. Local ChromaDB Cosine retrieval (<15ms on pre-warmed CPU embeddings)
        docs = retriever.invoke(standalone_query)
        context_text = "\n\n".join([d.page_content for d in docs])

        # 3. Single Gemini Flash API synthesis call
        answer = qa_chain.invoke(
            {
                "context": context_text,
                "chat_history": history_str,
                "question": question,
            }
        )

        elapsed_sec = round(time.perf_counter() - start_time, 2)
        parsed_sources = [parse_source_chunk(d.page_content, i) for i, d in enumerate(docs, 1)]

        return JSONResponse(
            {
                "answer": answer,
                "original_query": question,
                "standalone_query": standalone_query,
                "was_reformulated": was_reformulated,
                "latency_sec": elapsed_sec,
                "sources": parsed_sources,
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RAG synthesis error: {str(e)}")


@app.get("/", response_class=HTMLResponse)
def serve_portal():
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>TPC-Assist Admission Portal | Talibon Polytechnic College</title>
    <link rel="icon" href="{LOGO_DATA_URI}" />
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap" rel="stylesheet">
    <style>
        :root {{
            --bg-canvas: #EEF2F6;
            --surface-card: #FFFFFF;
            --surface-muted: #F4F7FA;
            --hero-navy: #062D4C;
            --hero-teal: #0B6A58;
            --hero-emerald: #128E63;
            --user-bubble: #1751B0;
            --user-bubble-dark: #103F8C;
            --bot-bubble: #F1F4F9;
            --text-primary: #0F1D2D;
            --text-secondary: #4B5E71;
            --text-muted: #6C7F93;
            --border-subtle: #DCE4EE;
            --border-strong: #8FA6BF;
            --emerald-accent: #0F835B;
            --emerald-highlight: #D4F4E2;
            --gold-accent: #EAB308;
            --radius-xl: 18px;
            --radius-lg: 14px;
            --radius-md: 10px;
            --shadow-card: 0 8px 24px rgba(9, 30, 66, 0.06), 0 1px 3px rgba(9, 30, 66, 0.04);
        }}

        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}

        body {{
            font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
            background-color: var(--bg-canvas);
            color: var(--text-primary);
            min-height: 100vh;
            padding: 20px 24px;
            line-height: 1.5;
        }}

        .portal-shell {{
            max-width: 1360px;
            margin: 0 auto;
            display: flex;
            flex-direction: column;
            gap: 20px;
        }}

        /* --- TOP HERO BANNER WITH 5-STEP ENROLLMENT TRACKER --- */
        .hero-banner {{
            background: linear-gradient(100deg, var(--hero-navy) 0%, #084B66 48%, var(--hero-emerald) 100%);
            border-radius: var(--radius-xl);
            padding: 26px 36px;
            color: #FFFFFF;
            display: flex;
            align-items: center;
            gap: 32px;
            box-shadow: 0 12px 32px rgba(6, 45, 76, 0.18);
            position: relative;
            overflow: hidden;
            border: 1px solid rgba(255, 255, 255, 0.14);
        }}

        .hero-seal-container {{
            flex-shrink: 0;
            width: 112px;
            height: 112px;
            border-radius: 50%;
            background: rgba(255, 255, 255, 0.1);
            padding: 6px;
            box-shadow: 0 6px 20px rgba(0, 0, 0, 0.25);
            display: flex;
            align-items: center;
            justify-content: center;
        }}

        .hero-seal {{
            width: 100%;
            height: 100%;
            object-fit: contain;
            border-radius: 50%;
        }}

        .hero-content {{
            flex: 1;
            min-width: 0;
        }}

        .hero-top-row {{
            display: flex;
            align-items: baseline;
            justify-content: space-between;
            flex-wrap: wrap;
            gap: 12px;
            margin-bottom: 6px;
        }}

        .hero-title {{
            font-size: 2rem;
            font-weight: 400;
            letter-spacing: -0.02em;
            line-height: 1.15;
        }}

        .hero-title strong {{
            font-weight: 800;
        }}

        .hero-status-badges {{
            display: flex;
            gap: 8px;
            flex-wrap: wrap;
        }}

        .hero-badge {{
            background: rgba(255, 255, 255, 0.15);
            backdrop-filter: blur(6px);
            border: 1px solid rgba(255, 255, 255, 0.25);
            padding: 5px 12px;
            border-radius: 999px;
            font-size: 0.76rem;
            font-weight: 600;
            display: inline-flex;
            align-items: center;
            gap: 6px;
        }}

        .hero-subtitle {{
            font-size: 1.02rem;
            font-weight: 500;
            color: rgba(255, 255, 255, 0.9);
            margin-bottom: 18px;
        }}

        /* 5-Step Enrollment Stepper */
        .stepper-track {{
            display: grid;
            grid-template-columns: repeat(5, 1fr);
            position: relative;
            padding-top: 4px;
        }}

        .stepper-line {{
            position: absolute;
            top: 15px;
            left: 10%;
            right: 10%;
            height: 3px;
            background: rgba(255, 255, 255, 0.28);
            z-index: 1;
        }}

        .stepper-line-progress {{
            height: 100%;
            width: 25%;
            background: #FFFFFF;
            transition: width 0.3s ease;
        }}

        .step-item {{
            position: relative;
            z-index: 2;
            display: flex;
            flex-direction: column;
            align-items: center;
            text-align: center;
            cursor: pointer;
            background: none;
            border: none;
            color: #FFFFFF;
            font-family: inherit;
            padding: 0 4px;
            transition: transform 0.15s ease;
        }}

        .step-item:hover {{
            transform: translateY(-2px);
        }}

        .step-dot {{
            width: 24px;
            height: 24px;
            border-radius: 50%;
            background: rgba(255, 255, 255, 0.35);
            border: 3px solid rgba(255, 255, 255, 0.65);
            margin-bottom: 8px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 0.7rem;
            font-weight: 800;
            color: var(--hero-navy);
            transition: all 0.2s ease;
        }}

        .step-item.active .step-dot,
        .step-item.completed .step-dot {{
            background: #FFFFFF;
            border-color: var(--gold-accent);
            box-shadow: 0 0 0 4px rgba(255, 255, 255, 0.2);
        }}

        .step-label-num {{
            font-size: 0.76rem;
            font-weight: 600;
            opacity: 0.9;
        }}

        .step-label-title {{
            font-size: 0.82rem;
            font-weight: 700;
        }}

        /* --- SPLIT-SCREEN TWO-COLUMN WORKSPACE --- */
        .split-workspace {{
            display: grid;
            grid-template-columns: 1fr 1.08fr;
            gap: 22px;
            align-items: stretch;
        }}

        .panel-card {{
            background: var(--surface-card);
            border-radius: var(--radius-xl);
            border: 1px solid var(--border-subtle);
            box-shadow: var(--shadow-card);
            padding: 24px;
            display: flex;
            flex-direction: column;
            min-height: 620px;
            max-height: calc(100vh - 235px);
        }}

        .panel-header {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 16px;
            padding-bottom: 12px;
            border-bottom: 1px solid var(--border-subtle);
        }}

        .panel-title {{
            font-size: 1.18rem;
            font-weight: 700;
            color: var(--text-primary);
            letter-spacing: -0.01em;
        }}

        /* --- LEFT COLUMN: TPC HANDBOOK & PROGRAMS EXPLORER --- */
        .explorer-scroll {{
            overflow-y: auto;
            padding-right: 4px;
            display: flex;
            flex-direction: column;
            gap: 16px;
            flex: 1;
        }}

        .programs-grid {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 12px;
        }}

        .program-card {{
            background: var(--surface-card);
            border: 1.5px solid var(--border-subtle);
            border-radius: var(--radius-lg);
            padding: 14px 12px;
            text-align: left;
            cursor: pointer;
            font-family: inherit;
            transition: all 0.18s ease;
            display: flex;
            flex-direction: column;
            gap: 8px;
        }}

        .program-card:hover,
        .program-card:focus-visible {{
            border-color: var(--hero-navy);
            background: #F7FAFC;
            transform: translateY(-2px);
            box-shadow: 0 6px 14px rgba(6, 45, 76, 0.08);
            outline: none;
        }}

        .program-icon {{
            width: 52px;
            height: 52px;
            border-radius: 12px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 800;
            font-size: 0.95rem;
            color: #FFFFFF;
            box-shadow: 0 4px 10px rgba(0, 0, 0, 0.12);
        }}

        .icon-bsis {{
            background: linear-gradient(135deg, #0A3D7A, #1B65B8);
        }}

        .icon-crim {{
            background: linear-gradient(135deg, #162B5B, #2B4C8C);
            border: 2px solid #E5B842;
        }}

        .icon-agri {{
            background: linear-gradient(135deg, #136B3B, #229954);
        }}

        .program-card-title {{
            font-size: 0.92rem;
            font-weight: 700;
            color: var(--text-primary);
            line-height: 1.25;
        }}

        .program-card-sub {{
            font-size: 0.76rem;
            color: var(--text-muted);
        }}

        /* Quick Handbook Policy Topics Bar */
        .quick-topics-row {{
            display: flex;
            flex-wrap: wrap;
            gap: 7px;
        }}

        .topic-pill {{
            background: var(--surface-muted);
            border: 1px solid var(--border-strong);
            color: var(--text-primary);
            border-radius: 999px;
            padding: 6px 12px;
            font-size: 0.77rem;
            font-weight: 600;
            cursor: pointer;
            font-family: inherit;
            transition: all 0.15s ease;
        }}

        .topic-pill:hover {{
            background: var(--hero-navy);
            color: #FFFFFF;
            border-color: var(--hero-navy);
        }}

        /* Verified Citation Inspector Box */
        .citation-box {{
            background: var(--surface-card);
            border: 1.5px solid var(--border-subtle);
            border-radius: var(--radius-lg);
            padding: 18px;
            display: flex;
            flex-direction: column;
            gap: 12px;
            flex: 1;
        }}

        .citation-box-header {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding-bottom: 10px;
            border-bottom: 1px solid var(--border-subtle);
        }}

        .verified-badge {{
            display: inline-flex;
            align-items: center;
            gap: 8px;
            font-size: 0.95rem;
            font-weight: 700;
            color: var(--text-primary);
        }}

        .verified-shield {{
            width: 22px;
            height: 22px;
            border-radius: 50%;
            background: var(--emerald-accent);
            color: #FFFFFF;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            font-size: 0.75rem;
            font-weight: 800;
        }}

        .citation-tabs {{
            display: flex;
            gap: 6px;
        }}

        .citation-tab {{
            background: var(--surface-muted);
            border: 1px solid var(--border-subtle);
            border-radius: 6px;
            padding: 4px 10px;
            font-size: 0.74rem;
            font-weight: 700;
            color: var(--text-secondary);
            cursor: pointer;
            font-family: inherit;
        }}

        .citation-tab.active {{
            background: var(--emerald-accent);
            color: #FFFFFF;
            border-color: var(--emerald-accent);
        }}

        .reformulation-pill {{
            background: #FEF9C3;
            border: 1px solid #FDE047;
            color: #713F12;
            padding: 8px 12px;
            border-radius: 8px;
            font-size: 0.79rem;
            display: none;
        }}

        .citation-item-title {{
            font-size: 0.9rem;
            font-weight: 700;
            color: var(--text-primary);
            margin-bottom: 6px;
        }}

        .citation-item-body {{
            font-size: 0.86rem;
            color: var(--text-secondary);
            line-height: 1.65;
            white-space: pre-line;
        }}

        .hl-phrase {{
            background-color: var(--emerald-highlight);
            color: #064E3B;
            padding: 1px 4px;
            border-radius: 4px;
            font-weight: 600;
        }}

        /* --- RIGHT COLUMN: INTUITIVE ADMISSION AI CHATBOT CONVERSATION --- */
        .chat-header-actions {{
            display: flex;
            align-items: center;
            gap: 8px;
        }}

        .latency-pill {{
            background: #ECFDF5;
            color: #065F46;
            border: 1px solid #A7F3D0;
            padding: 4px 9px;
            border-radius: 999px;
            font-size: 0.73rem;
            font-weight: 700;
            display: none;
        }}

        .memory-pill {{
            background: #EBF3FE;
            color: var(--user-bubble);
            border: 1px solid #BFDBFE;
            padding: 4px 10px;
            border-radius: 999px;
            font-size: 0.75rem;
            font-weight: 700;
        }}

        .reset-btn {{
            background: var(--surface-muted);
            border: 1px solid var(--border-strong);
            color: var(--text-secondary);
            border-radius: 8px;
            padding: 5px 11px;
            font-size: 0.76rem;
            font-weight: 600;
            cursor: pointer;
            font-family: inherit;
        }}

        .reset-btn:hover {{
            background: #FEE2E2;
            color: #991B1B;
            border-color: #FCA5A5;
        }}

        .chat-feed {{
            flex: 1;
            overflow-y: auto;
            padding: 8px 6px 14px 4px;
            display: flex;
            flex-direction: column;
            gap: 14px;
        }}

        .msg-row {{
            display: flex;
            gap: 12px;
            align-items: flex-start;
        }}

        .msg-row.user {{
            justify-content: flex-end;
        }}

        .bot-avatar {{
            width: 38px;
            height: 38px;
            border-radius: 50%;
            object-fit: contain;
            flex-shrink: 0;
            border: 1.5px solid var(--border-subtle);
            background: #FFFFFF;
            padding: 2px;
        }}

        .msg-bubble {{
            max-width: 82%;
            padding: 13px 16px;
            border-radius: 16px;
            font-size: 0.91rem;
            line-height: 1.58;
            word-wrap: break-word;
        }}

        .msg-row.user .msg-bubble {{
            background: linear-gradient(135deg, var(--user-bubble), var(--user-bubble-dark));
            color: #FFFFFF;
            border-bottom-right-radius: 4px;
            box-shadow: 0 4px 12px rgba(23, 81, 176, 0.2);
        }}

        .msg-row.assistant .msg-bubble {{
            background: var(--bot-bubble);
            color: var(--text-primary);
            border: 1px solid var(--border-subtle);
            border-top-left-radius: 4px;
        }}

        .msg-bubble ul, .msg-bubble ol {{
            margin: 6px 0 6px 20px;
        }}

        .msg-bubble li {{
            margin-bottom: 4px;
        }}

        /* Pill Chat Input Bar matching Option 2 Mockup */
        .chat-input-shell {{
            margin-top: 12px;
            display: flex;
            align-items: center;
            background: var(--surface-muted);
            border: 2px solid var(--hero-teal);
            border-radius: 999px;
            padding: 5px 6px 5px 18px;
            box-shadow: 0 3px 10px rgba(11, 106, 88, 0.08);
        }}

        .chat-input-shell:focus-within {{
            background: #FFFFFF;
            box-shadow: 0 0 0 3px rgba(18, 142, 99, 0.2);
        }}

        .chat-input {{
            flex: 1;
            border: none;
            background: transparent;
            font-family: inherit;
            font-size: 0.92rem;
            color: var(--text-primary);
            outline: none;
            padding: 6px 8px 6px 0;
        }}

        .send-btn {{
            background: linear-gradient(135deg, var(--hero-teal), var(--hero-emerald));
            color: #FFFFFF;
            border: none;
            border-radius: 999px;
            padding: 10px 24px;
            font-family: inherit;
            font-size: 0.88rem;
            font-weight: 700;
            cursor: pointer;
            transition: opacity 0.15s ease, transform 0.1s ease;
        }}

        .send-btn:hover {{
            opacity: 0.94;
        }}

        .send-btn:disabled {{
            opacity: 0.55;
            cursor: not-allowed;
        }}

        @media (forced-colors: active) {{
            .panel-card, .program-card, .citation-box, .msg-bubble, .chat-input-shell {{
                border: 1px solid CanvasText;
            }}
        }}

        @media (max-width: 980px) {{
            .split-workspace {{
                grid-template-columns: 1fr;
            }}
            .panel-card {{
                max-height: none;
                min-height: 480px;
            }}
            .hero-banner {{
                flex-direction: column;
                text-align: center;
                padding: 22px;
            }}
        }}
    </style>
</head>
<body>
    <div class="portal-shell">
        <!-- TOP HERO BANNER WITH 5-STEP ENROLLMENT TRACKER -->
        <header class="hero-banner">
            <div class="hero-seal-container">
                <img src="{LOGO_DATA_URI}" alt="Talibon Polytechnic College Seal" class="hero-seal" />
            </div>
            <div class="hero-content">
                <div class="hero-top-row">
                    <h1 class="hero-title"><strong>TPC-Assist</strong> Admission Portal</h1>
                    <div class="hero-status-badges">
                        <span class="hero-badge">📚 26 Handbook Pages (38 Chunks)</span>
                        <span class="hero-badge">⚡ Fast Single-Pass RAG</span>
                    </div>
                </div>
                <p class="hero-subtitle">5-Step Enrollment Tracker — Click any step to inspect official procedures</p>
                <div class="stepper-track" role="navigation" aria-label="5-Step Enrollment Tracker">
                    <div class="stepper-line">
                        <div class="stepper-line-progress" id="stepperProgress"></div>
                    </div>
                    <button type="button" class="step-item active" onclick="selectStep(1, 'What happens during Step 1: Document Verification at the Registrar Window 1, and what documents are checked?')">
                        <span class="step-dot">✓</span>
                        <span class="step-label-num">Step 1:</span>
                        <span class="step-label-title">Verification</span>
                    </button>
                    <button type="button" class="step-item active" onclick="selectStep(2, 'What is required for Step 2: TPC Admission Test at the Guidance and Testing Center?')">
                        <span class="step-dot">2</span>
                        <span class="step-label-num">Step 2:</span>
                        <span class="step-label-title">Entrance Test</span>
                    </button>
                    <button type="button" class="step-item" onclick="selectStep(3, 'How does Step 3: Academic Advising and Subject Loading work at the Department Dean Office?')">
                        <span class="step-dot">3</span>
                        <span class="step-label-num">Step 3:</span>
                        <span class="step-label-title">Academic Advising</span>
                    </button>
                    <button type="button" class="step-item" onclick="selectStep(4, 'What happens in Step 4: Assessment of Fees at the Accounting and Cashier Office?')">
                        <span class="step-dot">4</span>
                        <span class="step-label-num">Step 4:</span>
                        <span class="step-label-title">Assessment</span>
                    </button>
                    <button type="button" class="step-item" onclick="selectStep(5, 'How do I complete Step 5: Validation and Issuance of the Certificate of Registration (COR)?')">
                        <span class="step-dot">5</span>
                        <span class="step-label-num">Step 5:</span>
                        <span class="step-label-title">COR Release</span>
                    </button>
                </div>
            </div>
        </header>

        <!-- TWO-COLUMN SPLIT WORKSPACE -->
        <main class="split-workspace">
            <!-- LEFT SIDE: TPC HANDBOOK & PROGRAMS EXPLORER -->
            <section class="panel-card" aria-label="TPC Handbook and Programs Explorer">
                <div class="panel-header">
                    <h2 class="panel-title">TPC Handbook &amp; Programs Explorer</h2>
                    <span style="font-size: 0.78rem; color: var(--text-muted); font-weight: 600;">Click to Query</span>
                </div>

                <div class="explorer-scroll">
                    <!-- Visual Program Cards -->
                    <div class="programs-grid">
                        <button type="button" class="program-card" onclick="askQuestion('Tell me about the BS Information Systems (BSIS) program and all degree programs offered at TPC.')">
                            <div class="program-icon icon-bsis">BSIS</div>
                            <div>
                                <div class="program-card-title">BSIS</div>
                                <div class="program-card-sub">Information Systems</div>
                            </div>
                        </button>

                        <button type="button" class="program-card" onclick="askQuestion('Does Talibon Polytechnic College offer BS Criminology, and what are the admission requirements?')">
                            <div class="program-icon icon-crim">🛡️</div>
                            <div>
                                <div class="program-card-title">BS Criminology</div>
                                <div class="program-card-sub">Criminology Course</div>
                            </div>
                        </button>

                        <button type="button" class="program-card" onclick="askQuestion('Does TPC offer BS Agriculture and what other degree courses are available?')">
                            <div class="program-icon icon-agri">🌾</div>
                            <div>
                                <div class="program-card-title">BS Agriculture</div>
                                <div class="program-card-sub">Agriculture Course</div>
                            </div>
                        </button>
                    </div>

                    <!-- Quick Handbook Policy Filter Pills -->
                    <div class="quick-topics-row">
                        <button type="button" class="topic-pill" onclick="askQuestion('What documents do incoming freshmen need to submit for admission?')">📋 Freshman Checklist</button>
                        <button type="button" class="topic-pill" onclick="askQuestion('What are the admission requirements for transferees?')">🔄 Transferee Requirements</button>
                        <button type="button" class="topic-pill" onclick="askQuestion('What is the attendance and tardiness policy at TPC?')">🕒 Attendance &amp; Tardiness</button>
                        <button type="button" class="topic-pill" onclick="askQuestion('How many units can graduating students take during regular and summer terms?')">⚖️ Academic Load Rules</button>
                        <button type="button" class="topic-pill" onclick="askQuestion('How do I apply for a student visa to study in Japan?')">🛡️ Test Out-of-Scope Refusal</button>
                    </div>

                    <!-- Verified Citation Box -->
                    <div class="citation-box">
                        <div class="citation-box-header">
                            <div class="verified-badge">
                                <span class="verified-shield" aria-hidden="true">✓</span>
                                <span>Verified Citation</span>
                            </div>
                            <div class="citation-tabs" id="citationTabs">
                                <button type="button" class="citation-tab active" onclick="switchCitationTab(0)">Chunk #1</button>
                                <button type="button" class="citation-tab" onclick="switchCitationTab(1)">Chunk #2</button>
                                <button type="button" class="citation-tab" onclick="switchCitationTab(2)">Chunk #3</button>
                            </div>
                        </div>

                        <div class="reformulation-pill" id="reformulationBanner"></div>

                        <div id="citationDisplay">
                            <div class="citation-item-title" id="citationTitle">Handbook Page 12: 11. Freshman Admission Requirements</div>
                            <div class="citation-item-body" id="citationBody">
                                Incoming freshmen applying to Talibon Polytechnic College must submit the following verified documents to the Office of the College Registrar: <span class="hl-phrase">Admission Test Result, Form 138/137 (Original Report Card), Certificate of Good Moral Character, PSA Birth Certificate</span>, recent 2x2 ID photos, accomplished Personal Data Sheet, brown envelope, and white folder.
                            </div>
                        </div>
                    </div>
                </div>
            </section>

            <!-- RIGHT SIDE: INTUITIVE ADMISSION AI CHATBOT CONVERSATION -->
            <section class="panel-card" aria-label="Intuitive Admission AI Chatbot Conversation">
                <div class="panel-header">
                    <h2 class="panel-title">Intuitive Admission AI Chatbot Conversation</h2>
                    <div class="chat-header-actions">
                        <span class="latency-pill" id="latencyBadge">⚡ 0.0s</span>
                        <span class="memory-pill" id="memoryBadge">Memory: 0 turns</span>
                        <button type="button" class="reset-btn" onclick="resetConversation()">Reset Chat</button>
                    </div>
                </div>

                <div class="chat-feed" id="chatFeed">
                    <div class="msg-row assistant">
                        <img src="{LOGO_DATA_URI}" alt="TPC Seal" class="bot-avatar" />
                        <div class="msg-bubble">
                            Welcome to the <strong>Talibon Polytechnic College (TPC) Admission Portal</strong>! 👋<br><br>
                            I am grounded strictly in the official 26-page TPC Student Handbook. Ask me about <strong>freshman checklists, transferee credentials, offered degree programs, academic load rules, or the 5-step enrollment process</strong>—or click any card on the left explorer to begin.
                        </div>
                    </div>
                </div>

                <form class="chat-input-shell" onsubmit="handleChatSubmit(event)">
                    <input
                        type="text"
                        id="chatInput"
                        class="chat-input"
                        placeholder="Ask about freshman requirements, courses, or follow-up questions..."
                        autocomplete="off"
                    />
                    <button type="submit" id="sendBtn" class="send-btn">Send</button>
                </form>
            </section>
        </main>
    </div>

    <script>
        const LOGO_URI = "{LOGO_DATA_URI}";
        let chatHistory = [];
        let currentSources = [
            {{
                index: 1,
                title: "Handbook Page 12: 11. Freshman Admission Requirements",
                content: "Required documents: Admission Test Result, Form 138/137, Good Moral Certificate, PSA Birth Certificate, Recent ID photos, Personal Data Sheet, Brown envelope, and White folder."
            }},
            {{
                index: 2,
                title: "Handbook Page 9: 8. Programs Offered",
                content: "BA English Language, BA Political Science, BS Agriculture, BS Accounting Information Systems, BS Information Systems, BECEd, BS Criminology."
            }},
            {{
                index: 3,
                title: "Handbook Page 17: 16. Enrollment Policies",
                content: "Enrollment is valid only during the official registration period. Students become officially enrolled only after receiving a validated Certificate of Registration (COR)."
            }}
        ];

        function formatMarkdown(text) {{
            let escaped = text
                .replace(/&/g, "&amp;")
                .replace(/</g, "&lt;")
                .replace(/>/g, "&gt;");
            escaped = escaped.replace(/\\*\\*(.+?)\\*\\*/g, "<strong>$1</strong>");
            escaped = escaped.replace(/\\*(.+?)\\*/g, "<em>$1</em>");
            escaped = escaped.replace(/\\n/g, "<br>");
            return escaped;
        }}

        function highlightKeyPhrases(text) {{
            let safe = text
                .replace(/&/g, "&amp;")
                .replace(/</g, "&lt;")
                .replace(/>/g, "&gt;");
            const sentences = safe.split(/(?<=\\.)\\s+/);
            if (sentences.length > 1) {{
                sentences[0] = `<span class="hl-phrase">${{sentences[0]}}</span>`;
                return sentences.join(" ");
            }}
            return `<span class="hl-phrase">${{safe}}</span>`;
        }}

        function switchCitationTab(tabIdx) {{
            const tabs = document.querySelectorAll(".citation-tab");
            tabs.forEach((t, i) => t.classList.toggle("active", i === tabIdx));
            const src = currentSources[tabIdx] || currentSources[0];
            if (!src) return;
            document.getElementById("citationTitle").textContent = src.title;
            document.getElementById("citationBody").innerHTML = highlightKeyPhrases(src.content);
        }}

        function selectStep(stepNum, queryText) {{
            const steps = document.querySelectorAll(".step-item");
            steps.forEach((el, idx) => {{
                el.classList.toggle("active", idx + 1 <= stepNum);
                const dot = el.querySelector(".step-dot");
                dot.textContent = (idx + 1 < stepNum) ? "✓" : String(idx + 1);
            }});
            const pct = ((stepNum - 1) / 4) * 100;
            document.getElementById("stepperProgress").style.width = pct + "%";
            askQuestion(queryText);
        }}

        function appendMessage(role, htmlContent) {{
            const feed = document.getElementById("chatFeed");
            const row = document.createElement("div");
            row.className = "msg-row " + role;

            if (role === "assistant") {{
                const avatar = document.createElement("img");
                avatar.src = LOGO_URI;
                avatar.alt = "TPC Seal";
                avatar.className = "bot-avatar";
                row.appendChild(avatar);
            }}

            const bubble = document.createElement("div");
            bubble.className = "msg-bubble";
            bubble.innerHTML = htmlContent;
            row.appendChild(bubble);

            feed.appendChild(row);
            feed.scrollTop = feed.scrollHeight;
            return bubble;
        }}

        async function askQuestion(questionText) {{
            const input = document.getElementById("chatInput");
            const sendBtn = document.getElementById("sendBtn");
            if (!questionText || sendBtn.disabled) return;

            input.value = "";
            sendBtn.disabled = true;

            appendMessage("user", formatMarkdown(questionText));
            const loadingBubble = appendMessage("assistant", "<em>Consulting official TPC Student Handbook...</em>");

            try {{
                const resp = await fetch("/api/chat", {{
                    method: "POST",
                    headers: {{ "Content-Type": "application/json" }},
                    body: JSON.stringify({{
                        question: questionText,
                        history: chatHistory
                    }})
                }});

                const data = await resp.json();
                if (!resp.ok) {{
                    throw new Error(data.detail || "Unable to process request.");
                }}

                loadingBubble.innerHTML = formatMarkdown(data.answer);

                chatHistory.push({{ role: "user", content: questionText }});
                chatHistory.push({{ role: "assistant", content: data.answer }});

                const turnCount = chatHistory.filter(m => m.role === "user").length;
                document.getElementById("memoryBadge").textContent = `Memory: ${{turnCount}} turn${{turnCount === 1 ? "" : "s"}}`;

                if (data.latency_sec !== undefined) {{
                    const latBadge = document.getElementById("latencyBadge");
                    latBadge.style.display = "inline-block";
                    latBadge.textContent = `⚡ ${{data.latency_sec}}s`;
                }}

                const refBanner = document.getElementById("reformulationBanner");
                if (data.was_reformulated && data.standalone_query) {{
                    refBanner.style.display = "block";
                    refBanner.innerHTML = `<strong>🔄 Multi-Turn Context Applied:</strong> Retrieved using <em>"${{data.standalone_query}}"</em>`;
                }} else {{
                    refBanner.style.display = "none";
                }}

                if (data.sources && data.sources.length > 0) {{
                    currentSources = data.sources;
                    switchCitationTab(0);
                }}
            }} catch (err) {{
                loadingBubble.innerHTML = `<span style="color:#991B1B;">⚠️ Error: ${{err.message}}</span>`;
            }} finally {{
                sendBtn.disabled = false;
                document.getElementById("chatFeed").scrollTop = document.getElementById("chatFeed").scrollHeight;
            }}
        }}

        function handleChatSubmit(e) {{
            e.preventDefault();
            const input = document.getElementById("chatInput");
            const q = input.value.trim();
            if (q) {{
                askQuestion(q);
            }}
        }}

        function resetConversation() {{
            chatHistory = [];
            document.getElementById("memoryBadge").textContent = "Memory: 0 turns";
            document.getElementById("latencyBadge").style.display = "none";
            document.getElementById("reformulationBanner").style.display = "none";
            const feed = document.getElementById("chatFeed");
            feed.innerHTML = "";
            appendMessage(
                "assistant",
                "Conversation session reset. Ask a new question about <strong>TPC admission requirements, degree programs, or enrollment policies</strong>, or click any card on the left explorer."
            );
        }}
    </script>
</body>
</html>"""
    return HTMLResponse(content=html_content)


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8501"))
    uvicorn.run(app, host="0.0.0.0", port=port)
