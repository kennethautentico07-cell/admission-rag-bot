# Project Proposal: Generative AI Capstone

**Course:** IS Professional Elective 4 — Generative AI Systems  
**Instructor:** Jessie A. Melendres  
**Academic Year:** 1st Semester, A.Y. 2026–2027  

---

## 1. Project Identity

* **Project Title:** TPC-Assist: A RAG-Based Admission and Policy Chatbot for Talibon Polytechnic College
* **Chosen Theme:** Document/Knowledge-Base Q&A Assistant
* **Repository Link:** `https://github.com/kennethautentico07-cell/admission-rag-bot` 

---

## 2. Problem Statement

Navigating college admission requirements, enrollment procedures, and institutional policies often presents a major hurdle for incoming freshmen and transferees at Talibon Polytechnic College (TPC). Currently, prospective students must either manually inspect lengthy PDF manuals or queue up at administrative offices to ask repetitive questions. Manual document lookups are slow, prone to oversight, and impose a heavy support burden on administrative staff during peak enrollment cycles. 

Existing non-AI keyword searches often fail when queries do not match exact manual phrasing. Furthermore, standard ungrounded Large Language Models (LLMs) tend to hallucinate inaccurate campus procedures or deadlines. TPC-Assist solves this problem by utilizing Retrieval-Augmented Generation (RAG) to dynamically fetch verified policy context from official TPC handbooks, ensuring immediate, grounded, and accurate responses.

---

## 3. Target Users

* **Primary Users (Applicants & Incoming Freshmen):** Prospective students seeking quick, accurate answers regarding document requirements, entrance examination schedules, and course offerings without reading through dense handbooks.
* **Secondary Users (Enrolled Students & Administrative Staff):** Current students verifying retention policies or academic guidelines, as well as admissions staff who can offload routine, repetitive inquiries to an automated agent.

---

## 4. Dataset Source(s)

To ensure high domain accuracy, the RAG knowledge base is grounded in official institutional documentation:

* **Data Type:** PDF and structured Markdown (`.pdf`, `.md`)
* **Source:** Official Talibon Polytechnic College Admission & Student Handbook (`TPC_Admission_RAG_Knowledge_Base.pdf`)
* **Volume:** 1 comprehensive handbook document (exceeding 25 pages of raw text), converted into cleaned, chunked Markdown (`extracted_enrollment_manual.md`) stored in the `/data` folder.

---

## 5. Success Criteria

1. **Technical Accuracy & Hallucination Prevention:** The system correctly grounds answers in retrieved context chunks for at least 90% of in-scope inquiries and explicitly refuses to answer out-of-scope queries (e.g., student visa requests) using defined system prompts.
2. **Retrieval Performance:** Query retrieval from ChromaDB and response synthesis via Gemini 3.6 Flash completes with an end-to-end latency under 3 seconds.
3. **Source Transparency:** 100% of generated responses display an expandable reference panel showing the exact context chunks retrieved from the vector database.
4. **Usability:** A clean, responsive Streamlit user interface featuring a customized TPC theme, custom logo integration, and multi-turn conversational state tracking.

---

## 6. Team Composition & Role Distribution

| Team Member Name | Primary Responsibility | Assigned Component |
| :--- | :--- | :--- |
| Kenneth Autentico | Lead Developer | Data Pipeline, Text Preprocessing, & Vector Embeddings |
| Kenneth Autentico | AI Orchestrator | RAG Architecture, LangChain Orchestration, & Prompt Engineering |
| Justine AUtentico, Joar Narabe, Edlyn Joy Sobiono | DevOps / UI Engineer | Streamlit Interface, Session Management, & Docker Containerization |


---

## 7. Data Reflection & Sanitation (1/2 Page)

### Messy Data Challenges
The primary source document (`TPC_Admission_RAG_Knowledge_Base.pdf`) contained several formatting inconsistencies typical of administrative institutional manuals:
* **Unstructured Headers & Page Artifacts:** Headers, footers, page numbers, and repeated institutional watermarks disrupted text flow across page boundaries.
* **Non-Standard Lists:** Bulleted requirement lists contained arbitrary whitespace, non-standard unicode characters, and mixed line breaks.
* **Tabular Policy Formats:** Fee structures and course offerings were organized in non-border tables, causing standard PDF text extractors to scramble column alignment into ungrammatical sentences.

### Handling & Sanitation Strategy
1. **Markdown Normalization:** Converted raw PDF outputs into structured Markdown (`data/extracted_enrollment_manual.md`), manually rebuilding table structures into explicit Markdown tables.
2. **Text Sanitation Pipeline:** Implemented regex-based cleanups to strip non-printable ASCII characters, collapse redundant newlines (`\n\n+`), and remove running headers/footers.
3. **Semantic Hierarchy Preservation:** Standardized section headers (`#`, `##`, `###`) to ensure that chunking algorithms retain positional context during vector retrieval.