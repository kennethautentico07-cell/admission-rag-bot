# Framework Comparison Note: LangChain vs. LlamaIndex

**Project Title:** TPC-Assist: A RAG-Based Admission Chatbot  
**Course:** IS Professional Elective 4 — Generative AI Systems  
**Framework Selected:** LangChain  

---

## 1. Overview
When designing the Retrieval-Augmented Generation (RAG) architecture for TPC-Assist, we evaluated two leading orchestration frameworks: **LangChain** and **LlamaIndex**. Both frameworks facilitate the connection between Large Language Models (LLMs) and external data, but they have fundamentally different design philosophies regarding data routing, indexing, and application flow.

## 2. Approach to Data Routing and Indexing

### LlamaIndex
* **Design Philosophy:** LlamaIndex is fundamentally a "data-first" framework. It excels at ingesting, structuring, and indexing large amounts of data.
* **Indexing & Routing:** It provides highly optimized, out-of-the-box data structures (like tree indices, list indices, and vector indices). Its routing capabilities are deeply tied to query engines, automatically deciding whether to summarize a document, perform a semantic search, or route across multiple data sources using built-in RouterQueryEngines.
* **Strengths:** Extremely fast setup for complex document hierarchies and advanced retrieval techniques.

### LangChain
* **Design Philosophy:** LangChain is an "agent-first" or "pipeline-first" framework. It focuses on the composability of different components (prompts, LLMs, output parsers) using LangChain Expression Language (LCEL).
* **Indexing & Routing:** LangChain leaves vector storage and indexing largely to external databases (like ChromaDB or FAISS). Routing is typically handled explicitly by the developer using conditional chains, prompt routing, or agent tool-selection. 
* **Strengths:** Maximum flexibility, seamless integration of conversational memory, and granular control over the exact flow of data from user input to LLM output.

## 3. Justification for Selecting LangChain

For the TPC-Assist project, **LangChain** was selected as the primary orchestration framework for the following reasons:

1. **Pipeline Composability (LCEL):** Our system requires a strict, linear pipeline: *User Input → ChromaDB Retrieval → Strict System Prompt → Gemini LLM Synthesis*. LangChain's LCEL allows us to define this exact flow clearly in code (`chain = prompt | llm | StrOutputParser()`).
2. **Conversational Memory Integration:** LangChain offers robust, built-in support for conversational memory (`st.session_state` integration), which is essential for our multi-turn chat interface where prospective students might ask follow-up questions.
3. **Explicit Guardrails:** Because institutional policies require strict compliance, we needed granular control over the prompt templates to enforce anti-hallucination guardrails. LangChain's explicit prompt engineering tools provided better transparency than LlamaIndex's abstracted query engines.
4. **Integration with Streamlit:** LangChain's architecture pairs seamlessly with Streamlit's event loop, making it highly suitable for building our customized web UI.

While LlamaIndex would have been superior if we were parsing thousands of complex academic tables, LangChain provided the exact level of control, memory management, and flexibility required for a conversational, policy-based chatbot.