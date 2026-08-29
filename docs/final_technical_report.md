# Final Technical Report: TPC-Assist RAG System

**Project Title:** TPC-Assist: A RAG-Based Admission and Policy Chatbot  
**Course:** IS Professional Elective 4 — Generative AI Systems  
**Instructor:** Jessie A. Melendres  
**Lead Developer:** Kenneth Crescencio Autentico  
**Institution:** Talibon Polytechnic College (BSIS)  

---

## 1. Executive Summary
TPC-Assist is a Retrieval-Augmented Generation (RAG) application designed to streamline the student admission and enrollment process at Talibon Polytechnic College. By grounding a Large Language Model (LLM) in the official institutional manual, the system provides prospective and current students with immediate, highly accurate, and context-aware answers to policy, requirement, and scheduling queries. This project demonstrates a full end-to-end LLMOps pipeline, from raw data ingestion to a containerized web deployment.

## 2. System Architecture Overview
The application follows a standard RAG architecture to ensure responses are factual and strictly bound to the institution's official guidelines. The pipeline consists of five core stages:
1. **Document Ingestion & Chunking:** Parsing the TPC manual into semantic blocks.
2. **Vector Embedding:** Converting text chunks into numerical vectors.
3. **Vector Indexing & Storage:** Storing embeddings in a local vector database.
4. **Retrieval & Orchestration:** Matching user queries to the most relevant text chunks.
5. **Generative Synthesis & UI:** Passing the context to the LLM to generate a natural language response within a web interface.

## 3. Data Pipeline & Preprocessing
The foundation of the RAG system relies on high-quality, normalized data.
* **Source Material:** The official TPC Admission Handbook (PDF).
* **Text Extraction:** Raw text was extracted and converted into a structured Markdown format (`extracted_enrollment_manual.md`) to preserve hierarchical context (e.g., headers, lists).
* **Chunking Strategy:** We utilized LangChain’s `RecursiveCharacterTextSplitter`. Text was split into manageable chunks of approximately 1,000 characters with a 200-character overlap. This overlap ensures that semantic context is not lost if a sentence is split across two chunks.

## 4. Vector Database & Indexing
To enable semantic search, the chunked text was transformed into embeddings and indexed.
* **Embedding Model:** We selected `all-MiniLM-L6-v2` via HuggingFace. This model was chosen for its optimal balance of speed and semantic accuracy on CPU-bound local environments.
* **Vector Store:** A local instance of **ChromaDB** was implemented.
* **Distance Metric:** The database was configured to use **Cosine Similarity**, which accurately maps the semantic intent of student queries regardless of vector magnitude or chunk length. 

## 5. RAG Orchestration & Prompt Design
The orchestration layer acts as the brain of the application, connecting the database, the memory, and the LLM.
* **Framework:** **LangChain** was selected over LlamaIndex due to its highly composable LCEL (LangChain Expression Language) syntax and robust support for conversational memory.
* **LLM Integration:** Google’s **Gemini 3.6 Flash** was chosen for synthesis due to its rapid inference speeds and strong zero-shot instruction following capabilities.
* **Prompt Engineering Guardrails:** A strict system prompt was engineered to prevent hallucination. The model is explicitly instructed: *"Answer the question strictly based on the provided context below. If the answer cannot be found... state clearly: 'I cannot find this information in the official TPC manual.'"* This effectively eliminated out-of-scope responses during testing.

## 6. Application Interface & User Experience
The user-facing component was built prioritizing a clean-code architecture and a highly accessible interface.
* **Framework:** **Streamlit** was used to develop a reactive, minimalist green-themed web application that aligns with TPC's branding.
* **Session Management:** LangChain's conversational memory was integrated with Streamlit's `st.session_state` to support multi-turn dialogues, allowing users to ask follow-up questions naturally.
* **Source Attribution:** To build trust and transparency, a UI expander was added to the chat interface. Every AI response includes a "View Retrieved Reference Chunks" dropdown, allowing users and administrators to verify the exact paragraph the AI used to generate its answer.

## 7. Containerization & Deployment
To ensure environment consistency and prepare the application for production, the system was containerized.
* **Docker Implementation:** A multi-stage `Dockerfile` was authored using a lightweight `python:3.11-slim` base image. 
* **Dependency Management:** All required libraries (LangChain, Streamlit, HuggingFace, ChromaDB) are strictly version-controlled in `requirements.txt`.
* **Deployment Readiness:** The resulting Docker image isolates the application, making it platform-agnostic and ready for deployment on cloud services like AWS, Google Cloud Run, or on-premise institutional servers.

## 8. Limitations & Lessons Learned
* **Table Extraction:** Standard PDF parsing struggles with complex tables (e.g., specific fee matrices). Future iterations would benefit from specialized OCR tools or unstructured data parsers to handle tabular fee structures more accurately.
* **Fine-Tuning:** While PEFT/LoRA was considered, RAG proved structurally superior for this domain. Academic policies change frequently; updating a vector database is significantly more efficient than retraining a localized model.
* **Scalability:** The current setup uses a local ChromaDB instance. For a full campus-wide rollout, migrating to a managed vector database (like Pinecone or hosted Chroma) and implementing a proper SQL database to log user queries for analytics would be necessary.