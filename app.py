import os
import re
import threading
import time
from typing import List, Optional

import torch
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
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
    version="2.3.0",
)
app.add_middleware(GZipMiddleware, minimum_size=500)

LOGO_URL = "/logo.png"

# --- GLOBAL RAG ENGINE SINGLETON (Thread-Safe Non-Blocking Init) ---
_rag_components = None
_rag_lock = threading.Lock()
_answer_cache: dict[str, str] = {}
_model_cooldown_until: dict[str, float] = {}

# Prioritized multi-model pool (each model has its own independent Free-Tier daily quota bucket)
MODEL_FAILOVER_POOL = [
    ("gemini-3-flash-preview", {"thinking_budget": 0}),
    ("gemini-3.1-flash-lite", {"thinking_budget": 0}),
    ("gemini-3.5-flash-lite", {"thinking_level": "minimal"}),
    ("gemini-flash-lite-latest", {"thinking_level": "minimal"}),
    ("gemini-3.6-flash", {"thinking_budget": 0}),
    ("gemma-4-31b-it", {}),
]


def get_rag_components():
    global _rag_components
    if _rag_components is not None:
        return _rag_components
    with _rag_lock:
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

        # Strict Grounded RAG Synthesis Prompt with Multi-Turn Context (Prompt 1)
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

        # Build prioritized multi-model chains so 429 quota limits on one model automatically fail over
        qa_chains = []
        for model_name, extra_kwargs in MODEL_FAILOVER_POOL:
            try:
                llm = ChatGoogleGenerativeAI(
                    model=model_name,
                    max_retries=1,
                    **extra_kwargs,
                )
                qa_chains.append((model_name, qa_prompt | llm | StrOutputParser()))
            except Exception:
                continue

        _rag_components = (retriever, qa_chains)
        return _rag_components


def synthesize_extractive_fallback(question: str, docs: list) -> str:
    """
    Deterministic strictly-grounded fallback if all free-tier LLM quotas are simultaneously exhausted.
    Checks domain relevance against retrieved ChromaDB chunks and returns either the official refusal
    or a clean grounded summary from the top handbook chunk.
    """
    q_lower = question.lower()
    out_of_scope_signals = ["visa", "japan", "passport", "flight", "weather", "bitcoin", "stock", "recipe", "hotel", "dormitory"]
    if any(sig in q_lower for sig in out_of_scope_signals) or not docs:
        return "I cannot find this information in the official TPC manual."

    q_words = {
        w
        for w in re.findall(r"[a-z0-9]{3,}", q_lower)
        if w
        not in {
            "what",
            "where",
            "when",
            "how",
            "does",
            "the",
            "for",
            "and",
            "are",
            "can",
            "with",
            "from",
            "that",
            "this",
            "tpc",
            "talibon",
            "polytechnic",
            "college",
        }
    }
    combined_context = " ".join(d.page_content.lower() for d in docs)
    if q_words and not any(w in combined_context for w in q_words):
        return "I cannot find this information in the official TPC manual."

    top_parsed = parse_source_chunk(docs[0].page_content, 1)
    return (
        f"Based on the official TPC Student Handbook (**{top_parsed['title']}**):\n\n"
        f"{top_parsed['content']}"
    )


def invoke_with_failover(qa_chains: list, payload: dict, standalone_query: str, docs: list) -> str:
    """
    Invokes the RAG synthesis prompt using in-memory caching and automatic multi-model rotation.
    """
    cache_key = f"{standalone_query.strip().lower()}||{payload.get('chat_history', '').strip().lower()}"
    if cache_key in _answer_cache:
        return _answer_cache[cache_key]

    now = time.time()
    for model_name, chain in qa_chains:
        if _model_cooldown_until.get(model_name, 0) > now:
            continue
        try:
            answer = chain.invoke(payload)
            if answer and answer.strip():
                _answer_cache[cache_key] = answer.strip()
                return answer.strip()
        except Exception as err:
            err_str = str(err)
            if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str or "404" in err_str:
                # Cool down this specific model for 15 minutes and immediately try the next model
                _model_cooldown_until[model_name] = time.time() + 900
            continue

    # Ultimate safety net if every model in the pool is on cooldown
    return synthesize_extractive_fallback(payload.get("question", standalone_query), docs)


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
    # Warm up PyTorch & ChromaDB asynchronously so HTTP server renders HTML immediately on mobile
    def _warmup():
        try:
            get_rag_components()
        except Exception as e:
            print(f"[WARN] Deferred RAG initialization: {e}")

    threading.Thread(target=_warmup, daemon=True).start()


@app.get("/logo.png")
def serve_logo():
    if os.path.exists("tpc_logo.png"):
        return FileResponse(
            "tpc_logo.png",
            media_type="image/png",
            headers={"Cache-Control": "public, max-age=604800, immutable"},
        )
    raise HTTPException(status_code=404, detail="Logo not found")


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
        retriever, qa_chains = get_rag_components()
        history_list = req.history or []
        history_str = format_chat_history(history_list)

        # 1. Instant history-aware query contextualization (<1ms, 0 extra API calls)
        standalone_query, was_reformulated = build_contextualized_query(question, history_list)

        # 2. Local ChromaDB Cosine retrieval (<15ms on pre-warmed CPU embeddings)
        docs = retriever.invoke(standalone_query)
        context_text = "\n\n".join([d.page_content for d in docs])

        # 3. Cached + Multi-Model Failover RAG synthesis (immune to 429 per-model free-tier limits)
        answer = invoke_with_failover(
            qa_chains=qa_chains,
            payload={
                "context": context_text,
                "chat_history": history_str,
                "question": question,
            },
            standalone_query=standalone_query,
            docs=docs,
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
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no" />
    <title>TPC-Assist Admission Portal | Talibon Polytechnic College</title>
    <link rel="icon" href="{LOGO_URL}" />
    <link rel="preload" as="image" href="{LOGO_URL}" fetchpriority="high" />
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
            overflow-x: hidden;
        }}

        .portal-shell {{
            max-width: 1360px;
            margin: 0 auto;
            display: flex;
            flex-direction: column;
            gap: 18px;
        }}

        /* --- TOP HERO BANNER WITH 5-STEP ENROLLMENT TRACKER --- */
        .hero-banner {{
            background: linear-gradient(100deg, var(--hero-navy) 0%, #084B66 48%, var(--hero-emerald) 100%);
            border-radius: var(--radius-xl);
            padding: 24px 32px;
            color: #FFFFFF;
            display: flex;
            align-items: center;
            gap: 28px;
            box-shadow: 0 12px 32px rgba(6, 45, 76, 0.18);
            position: relative;
            border: 1px solid rgba(255, 255, 255, 0.14);
        }}

        .hero-identity {{
            display: flex;
            align-items: center;
            gap: 24px;
            width: 100%;
        }}

        .hero-seal-container {{
            flex-shrink: 0;
            width: 104px;
            height: 104px;
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
            align-items: center;
            justify-content: space-between;
            flex-wrap: wrap;
            gap: 10px;
            margin-bottom: 4px;
        }}

        .hero-title {{
            font-size: 1.9rem;
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
            font-size: 0.75rem;
            font-weight: 600;
            display: inline-flex;
            align-items: center;
            gap: 5px;
            white-space: nowrap;
        }}

        .hero-subtitle {{
            font-size: 0.96rem;
            font-weight: 500;
            color: rgba(255, 255, 255, 0.88);
            margin-bottom: 16px;
        }}

        /* 5-Step Enrollment Stepper (Desktop) */
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
            margin-bottom: 6px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 0.7rem;
            font-weight: 800;
            color: var(--hero-navy);
            transition: all 0.2s ease;
            flex-shrink: 0;
        }}

        .step-item.active .step-dot,
        .step-item.completed .step-dot {{
            background: #FFFFFF;
            border-color: var(--gold-accent);
            box-shadow: 0 0 0 4px rgba(255, 255, 255, 0.2);
        }}

        .step-label-num {{
            font-size: 0.74rem;
            font-weight: 600;
            opacity: 0.9;
        }}

        .step-label-title {{
            font-size: 0.8rem;
            font-weight: 700;
        }}

        /* --- MOBILE SEGMENTED TAB SWITCHER (Hidden on Desktop) --- */
        .mobile-tab-bar {{
            display: none;
            background: var(--surface-card);
            border: 1px solid var(--border-subtle);
            border-radius: 12px;
            padding: 4px;
            gap: 4px;
            box-shadow: 0 2px 8px rgba(9, 30, 66, 0.05);
        }}

        .mobile-tab-btn {{
            flex: 1;
            border: none;
            background: transparent;
            color: var(--text-secondary);
            font-family: inherit;
            font-size: 0.84rem;
            font-weight: 700;
            padding: 9px 10px;
            border-radius: 9px;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 6px;
            transition: all 0.15s ease;
        }}

        .mobile-tab-btn.active {{
            background: var(--hero-navy);
            color: #FFFFFF;
            box-shadow: 0 2px 6px rgba(6, 45, 76, 0.2);
        }}

        .citation-count-dot {{
            background: var(--emerald-accent);
            color: #FFFFFF;
            font-size: 0.68rem;
            padding: 1px 6px;
            border-radius: 999px;
            font-weight: 800;
        }}

        /* Mobile Quick-Prompt Strip inside Chat Panel (Hidden on Desktop) */
        .mobile-quick-strip {{
            display: none;
            flex-shrink: 0;
            align-items: center;
            overflow-x: auto;
            overflow-y: hidden;
            gap: 6px;
            height: 38px;
            padding: 2px 2px;
            margin-top: 6px;
            scrollbar-width: none;
            -webkit-overflow-scrolling: touch;
        }}

        .mobile-quick-strip::-webkit-scrollbar {{
            display: none;
        }}

        .mobile-quick-strip .topic-pill {{
            flex-shrink: 0;
            display: inline-flex;
            align-items: center;
            height: 30px;
            padding: 0 11px;
            font-size: 0.75rem;
            line-height: 1;
        }}

        .mobile-grounded-label {{
            display: none;
            align-items: center;
            gap: 6px;
            font-size: 0.76rem;
            font-weight: 700;
            color: var(--emerald-accent);
            white-space: nowrap;
        }}

        .grounded-dot {{
            width: 7px;
            height: 7px;
            border-radius: 50%;
            background: var(--emerald-accent);
            box-shadow: 0 0 0 3px rgba(13, 122, 95, 0.16);
            flex-shrink: 0;
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
            padding: 22px;
            display: flex;
            flex-direction: column;
            height: clamp(560px, calc(100vh - 220px), 780px);
        }}

        .panel-header {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 8px;
            margin-bottom: 14px;
            padding-bottom: 12px;
            border-bottom: 1px solid var(--border-subtle);
            flex-shrink: 0;
        }}

        .panel-title {{
            font-size: 1.14rem;
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
            flex: 1 1 auto;
            min-height: 0;
            -webkit-overflow-scrolling: touch;
        }}

        .programs-grid {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 12px;
            flex-shrink: 0;
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
            width: 48px;
            height: 48px;
            border-radius: 12px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 800;
            font-size: 0.9rem;
            color: #FFFFFF;
            box-shadow: 0 4px 10px rgba(0, 0, 0, 0.12);
            flex-shrink: 0;
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
            font-size: 0.9rem;
            font-weight: 700;
            color: var(--text-primary);
            line-height: 1.25;
        }}

        .program-card-sub {{
            font-size: 0.75rem;
            color: var(--text-muted);
        }}

        /* Quick Handbook Policy Topics Bar */
        .quick-topics-row {{
            display: flex;
            flex-wrap: wrap;
            gap: 7px;
            flex-shrink: 0;
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
            white-space: nowrap;
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
            padding: 16px;
            display: flex;
            flex-direction: column;
            gap: 12px;
            flex: 1 0 auto;
        }}

        .citation-box-header {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            flex-wrap: wrap;
            gap: 8px;
            padding-bottom: 10px;
            border-bottom: 1px solid var(--border-subtle);
        }}

        .verified-badge {{
            display: inline-flex;
            align-items: center;
            gap: 8px;
            font-size: 0.92rem;
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
            gap: 5px;
        }}

        .citation-tab {{
            background: var(--surface-muted);
            border: 1px solid var(--border-subtle);
            border-radius: 6px;
            padding: 4px 9px;
            font-size: 0.73rem;
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
            font-size: 0.88rem;
            font-weight: 700;
            color: var(--text-primary);
            margin-bottom: 6px;
        }}

        .citation-item-body {{
            font-size: 0.85rem;
            color: var(--text-secondary);
            line-height: 1.62;
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
            gap: 6px;
            flex-shrink: 0;
        }}

        .latency-pill {{
            background: #ECFDF5;
            color: #065F46;
            border: 1px solid #A7F3D0;
            padding: 4px 8px;
            border-radius: 999px;
            font-size: 0.72rem;
            font-weight: 700;
            display: none;
        }}

        .memory-pill {{
            background: #EBF3FE;
            color: var(--user-bubble);
            border: 1px solid #BFDBFE;
            padding: 4px 9px;
            border-radius: 999px;
            font-size: 0.73rem;
            font-weight: 700;
            white-space: nowrap;
        }}

        .reset-btn {{
            background: var(--surface-muted);
            border: 1px solid var(--border-strong);
            color: var(--text-secondary);
            border-radius: 8px;
            padding: 5px 10px;
            font-size: 0.74rem;
            font-weight: 600;
            cursor: pointer;
            font-family: inherit;
            white-space: nowrap;
        }}

        .reset-btn:hover {{
            background: #FEE2E2;
            color: #991B1B;
            border-color: #FCA5A5;
        }}

        .chat-feed {{
            flex: 1 1 auto;
            min-height: 0;
            overflow-y: auto;
            -webkit-overflow-scrolling: touch;
            padding: 6px 4px 12px 2px;
            display: flex;
            flex-direction: column;
            gap: 12px;
        }}

        .msg-row {{
            display: flex;
            gap: 10px;
            align-items: flex-start;
        }}

        .msg-row.user {{
            justify-content: flex-end;
        }}

        .bot-avatar {{
            width: 34px;
            height: 34px;
            border-radius: 50%;
            object-fit: contain;
            flex-shrink: 0;
            border: 1.5px solid var(--border-subtle);
            background: #FFFFFF;
            padding: 2px;
        }}

        .msg-bubble {{
            max-width: 84%;
            padding: 12px 15px;
            border-radius: 16px;
            font-size: 0.9rem;
            line-height: 1.56;
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

        /* Pill Chat Input Bar */
        .chat-input-shell {{
            margin-top: 8px;
            flex-shrink: 0;
            display: flex;
            align-items: center;
            background: var(--surface-muted);
            border: 2px solid var(--hero-teal);
            border-radius: 999px;
            padding: 4px 5px 4px 16px;
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
            min-width: 0;
        }}

        .send-btn {{
            background: linear-gradient(135deg, var(--hero-teal), var(--hero-emerald));
            color: #FFFFFF;
            border: none;
            border-radius: 999px;
            padding: 9px 20px;
            font-family: inherit;
            font-size: 0.86rem;
            font-weight: 700;
            cursor: pointer;
            flex-shrink: 0;
            transition: opacity 0.15s ease;
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

        /* =====================================================================
           RESPONSIVE MOBILE & TABLET LAYOUT (Clean, Zero-Clutter Mobile View)
           ===================================================================== */
        @media (max-width: 980px) {{
            html, body {{
                height: 100vh;
                height: 100dvh;
                overflow: hidden;
                padding: 0;
                margin: 0;
            }}

            body {{
                padding: 8px 8px 10px 8px;
            }}

            .portal-shell {{
                height: 100%;
                gap: 8px;
                display: flex;
                flex-direction: column;
            }}

            /* Ultra-Clean Compact Mobile Header */
            .hero-banner {{
                padding: 10px 12px;
                border-radius: 14px;
                flex-shrink: 0;
                display: block;
            }}

            .hero-identity {{
                display: grid;
                grid-template-columns: 38px minmax(0, 1fr);
                column-gap: 10px;
                row-gap: 8px;
                align-items: center;
            }}

            .hero-seal-container {{
                width: 38px;
                height: 38px;
                padding: 2px;
                grid-row: 1;
                grid-column: 1;
            }}

            .hero-content {{
                display: contents;
            }}

            .hero-top-row {{
                grid-row: 1;
                grid-column: 2;
                margin-bottom: 0;
            }}

            .hero-title {{
                font-size: 1.05rem;
                line-height: 1.2;
                white-space: nowrap;
                overflow: hidden;
                text-overflow: ellipsis;
            }}

            .hero-status-badges,
            .hero-subtitle {{
                display: none;
            }}

            /* Full-Width Horizontal Swipeable 5-Step Pill Tracker on Mobile */
            .stepper-track {{
                grid-row: 2;
                grid-column: 1 / -1;
                display: flex;
                align-items: center;
                overflow-x: auto;
                overflow-y: hidden;
                gap: 6px;
                padding: 2px 0;
                scrollbar-width: none;
                -webkit-overflow-scrolling: touch;
            }}

            .stepper-track::-webkit-scrollbar {{
                display: none;
            }}

            .stepper-line {{
                display: none;
            }}

            .step-item {{
                flex-direction: row;
                align-items: center;
                gap: 5px;
                background: rgba(255, 255, 255, 0.13);
                border: 1px solid rgba(255, 255, 255, 0.25);
                border-radius: 999px;
                padding: 4px 10px 4px 5px;
                flex-shrink: 0;
                height: 28px;
            }}

            .step-item.active {{
                background: rgba(255, 255, 255, 0.25);
                border-color: var(--gold-accent);
            }}

            .step-dot {{
                width: 18px;
                height: 18px;
                margin-bottom: 0;
                border-width: 1.5px;
                font-size: 0.62rem;
            }}

            .step-label-num {{
                display: none;
            }}

            .step-label-title {{
                font-size: 0.72rem;
                white-space: nowrap;
                line-height: 1;
            }}

            /* Single-Line Segmented Mobile Switcher (Chat vs Handbook) */
            .mobile-tab-bar {{
                display: flex;
                flex-shrink: 0;
                padding: 3px;
                border-radius: 11px;
            }}

            .mobile-tab-btn {{
                font-size: 0.78rem;
                padding: 7px 8px;
                white-space: nowrap;
                border-radius: 8px;
            }}

            .mobile-quick-strip {{
                display: flex;
            }}

            .split-workspace {{
                flex: 1 1 0%;
                min-height: 0;
                display: flex;
                flex-direction: column;
            }}

            .panel-card {{
                padding: 12px;
                border-radius: 14px;
                flex: 1 1 0%;
                min-height: 0;
                max-height: none;
                height: 100%;
            }}

            .panel-card.mobile-hidden {{
                display: none !important;
            }}

            .panel-header {{
                margin-bottom: 8px;
                padding-bottom: 8px;
                flex-shrink: 0;
            }}

            /* Hide duplicate 'Admission AI Assistant' heading inside Chat card on mobile */
            #panelChat .panel-title {{
                display: none;
            }}

            .mobile-grounded-label {{
                display: inline-flex;
            }}

            .panel-title {{
                font-size: 0.92rem;
                white-space: nowrap;
            }}

            .latency-pill,
            .memory-pill {{
                font-size: 0.69rem;
                padding: 3px 7px;
            }}

            .reset-btn {{
                font-size: 0.7rem;
                padding: 4px 8px;
            }}

            .programs-grid {{
                grid-template-columns: 1fr;
                gap: 8px;
            }}

            .program-card {{
                flex-direction: row;
                align-items: center;
                padding: 10px 12px;
                gap: 12px;
            }}

            .program-icon {{
                width: 38px;
                height: 38px;
                font-size: 0.78rem;
            }}

            .bot-avatar {{
                width: 28px;
                height: 28px;
            }}

            .msg-bubble {{
                max-width: 88%;
                font-size: 0.85rem;
                line-height: 1.48;
                padding: 10px 12px;
            }}

            .chat-input-shell {{
                margin-top: 6px;
                padding: 3px 4px 3px 14px;
            }}

            /* 16px font-size prevents iOS Safari from auto-zooming on input focus */
            .chat-input {{
                font-size: 16px;
                padding: 5px 6px 5px 0;
            }}

            .send-btn {{
                padding: 8px 16px;
                font-size: 0.82rem;
            }}
        }}
    </style>
</head>
<body>
    <div class="portal-shell">
        <!-- TOP HERO BANNER WITH 5-STEP ENROLLMENT TRACKER -->
        <header class="hero-banner">
            <div class="hero-identity">
                <div class="hero-seal-container">
                    <img src="{LOGO_URL}" alt="Talibon Polytechnic College Seal" class="hero-seal" width="104" height="104" fetchpriority="high" decoding="async" />
                </div>
                <div class="hero-content">
                    <div class="hero-top-row">
                        <h1 class="hero-title"><strong>TPC-Assist</strong> Admission Portal</h1>
                        <div class="hero-status-badges">
                            <span class="hero-badge">📚 26 Handbook Pages (38 Chunks)</span>
                            <span class="hero-badge">⚡ Fast Single-Pass RAG</span>
                        </div>
                    </div>
                    <p class="hero-subtitle">5-Step Enrollment Tracker — Tap any step to inspect official procedures</p>
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
            </div>
        </header>

        <!-- MOBILE SEGMENTED TAB SWITCHER (Only visible on phones/tablets) -->
        <div class="mobile-tab-bar" role="tablist">
            <button type="button" class="mobile-tab-btn active" id="tabBtnChat" onclick="switchMobileView('chat')">
                💬 AI Chat
            </button>
            <button type="button" class="mobile-tab-btn" id="tabBtnExplorer" onclick="switchMobileView('explorer')">
                📚 Handbook &amp; Sources <span class="citation-count-dot" id="mobileCitationBadge">3</span>
            </button>
        </div>

        <!-- TWO-COLUMN SPLIT WORKSPACE -->
        <main class="split-workspace">
            <!-- LEFT SIDE: TPC HANDBOOK & PROGRAMS EXPLORER -->
            <section class="panel-card mobile-hidden" id="panelExplorer" aria-label="TPC Handbook and Programs Explorer">
                <div class="panel-header">
                    <h2 class="panel-title">TPC Handbook &amp; Programs Explorer</h2>
                    <span style="font-size: 0.76rem; color: var(--text-muted); font-weight: 600;">Tap to Query</span>
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
            <section class="panel-card" id="panelChat" aria-label="Intuitive Admission AI Chatbot Conversation">
                <div class="panel-header">
                    <h2 class="panel-title">Admission AI Assistant</h2>
                    <span class="mobile-grounded-label"><span class="grounded-dot"></span>Handbook Grounded</span>
                    <div class="chat-header-actions">
                        <span class="latency-pill" id="latencyBadge">⚡ 0.0s</span>
                        <span class="memory-pill" id="memoryBadge">0 turns</span>
                        <button type="button" class="reset-btn" onclick="resetConversation()">Reset</button>
                    </div>
                </div>

                <div class="chat-feed" id="chatFeed">
                    <div class="msg-row assistant">
                        <img src="{LOGO_URL}" alt="TPC Seal" class="bot-avatar" width="34" height="34" decoding="async" />
                        <div class="msg-bubble">
                            Welcome to the <strong>Talibon Polytechnic College (TPC) Admission Portal</strong>! 👋<br><br>
                            Ask me about <strong>freshman checklists, transferee requirements, offered degree programs, academic load rules, or the 5-step enrollment process</strong>.
                        </div>
                    </div>
                </div>

                <!-- Mobile Quick-Inquiry Strip placed right above input bar for thumb ergonomics & zero top clutter -->
                <div class="mobile-quick-strip">
                    <button type="button" class="topic-pill" onclick="askQuestion('What documents do incoming freshmen need to submit for admission?')">📋 Freshman Docs</button>
                    <button type="button" class="topic-pill" onclick="askQuestion('What are the admission requirements for transferees?')">🔄 Transferees</button>
                    <button type="button" class="topic-pill" onclick="askQuestion('What degree programs and courses are offered at TPC?')">🎓 7 TPC Programs</button>
                    <button type="button" class="topic-pill" onclick="askQuestion('What is the attendance and tardiness policy at TPC?')">🕒 Attendance Rule</button>
                    <button type="button" class="topic-pill" onclick="askQuestion('How do I apply for a student visa to study in Japan?')">🛡️ Refusal Demo</button>
                </div>

                <form class="chat-input-shell" onsubmit="handleChatSubmit(event)">
                    <input
                        type="text"
                        id="chatInput"
                        class="chat-input"
                        placeholder="Ask about freshman requirements, courses..."
                        autocomplete="off"
                    />
                    <button type="submit" id="sendBtn" class="send-btn">Send</button>
                </form>
            </section>
        </main>
    </div>

    <script>
        const LOGO_URI = "{LOGO_URL}";
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

        function switchMobileView(view) {{
            const panelChat = document.getElementById("panelChat");
            const panelExplorer = document.getElementById("panelExplorer");
            const btnChat = document.getElementById("tabBtnChat");
            const btnExplorer = document.getElementById("tabBtnExplorer");

            if (view === "chat") {{
                panelChat.classList.remove("mobile-hidden");
                panelExplorer.classList.add("mobile-hidden");
                btnChat.classList.add("active");
                btnExplorer.classList.remove("active");
            }} else {{
                panelExplorer.classList.remove("mobile-hidden");
                panelChat.classList.add("mobile-hidden");
                btnExplorer.classList.add("active");
                btnChat.classList.remove("active");
            }}
        }}

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

            // On mobile, automatically switch to the Chat tab so user sees the conversation immediately
            if (window.innerWidth <= 980) {{
                switchMobileView("chat");
            }}

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

                const rawText = await resp.text();
                let data;
                try {{
                    data = JSON.parse(rawText);
                }} catch (_) {{
                    throw new Error(`Server response error (${{resp.status}}): ${{rawText.slice(0, 120)}}`);
                }}
                if (!resp.ok) {{
                    throw new Error(data.detail || "Unable to process request.");
                }}

                loadingBubble.innerHTML = formatMarkdown(data.answer);

                chatHistory.push({{ role: "user", content: questionText }});
                chatHistory.push({{ role: "assistant", content: data.answer }});

                const turnCount = chatHistory.filter(m => m.role === "user").length;
                document.getElementById("memoryBadge").textContent = `${{turnCount}} turn${{turnCount === 1 ? "" : "s"}}`;

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
                    document.getElementById("mobileCitationBadge").textContent = String(data.sources.length);
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
            document.getElementById("memoryBadge").textContent = "0 turns";
            document.getElementById("latencyBadge").style.display = "none";
            document.getElementById("reformulationBanner").style.display = "none";
            const feed = document.getElementById("chatFeed");
            feed.innerHTML = "";
            appendMessage(
                "assistant",
                "Conversation session reset. Ask a new question about <strong>TPC admission requirements, degree programs, or enrollment policies</strong>."
            );
        }}
    </script>
</body>
</html>"""
    return HTMLResponse(
        content=html_content,
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
    )


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8501"))
    uvicorn.run(app, host="0.0.0.0", port=port)
