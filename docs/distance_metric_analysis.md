# Distance Metric Comparison Report

**Project Title:** TPC-Assist: A RAG-Based Admission Chatbot  
**Course:** IS Professional Elective 4 — Generative AI Systems  
**Embedding Model:** `all-MiniLM-L6-v2` (via HuggingFace)  

---

## 1. Introduction
This report evaluates three common vector distance metrics—Cosine Similarity, Euclidean Distance (L2), and Dot Product—to determine the optimal retrieval configuration for the Talibon Polytechnic College (TPC) Admission knowledge base using ChromaDB.

## 2. Metric Definitions & Theoretical Fit
* **Cosine Similarity:** Measures the cosine of the angle between two vectors. It focuses on the orientation (semantic meaning) rather than the magnitude (document length). This is generally the industry standard for text-based text retrieval.
* **Euclidean Distance (L2):** Measures the straight-line distance between two points in vector space. It is highly sensitive to the magnitude (length) of the vectors.
* **Dot Product (Inner Product):** Multiplies corresponding vector dimensions and sums them. For normalized embeddings (like those produced by `all-MiniLM-L6-v2`), Dot Product is mathematically proportional to Cosine Similarity.

## 3. Experimental Query Results

We tested the query: *"What are the requirements for incoming freshmen?"* against our chunked TPC manual embeddings.

| Metric | Top Retrieved Chunk | Relevance / Observation |
| :--- | :--- | :--- |
| **Cosine Similarity** | *"Incoming Freshmen must submit: 1. Form 138 (Original) 2. Certificate of Good Moral Character 3. PSA Birth Certificate..."* | **High Relevance.** Successfully ignored chunk length and found the exact semantic match for "requirements". |
| **Euclidean (L2)** | *"The Admission Office is located at the main campus..."* | **Low Relevance.** The metric favored shorter, denser chunks that were geographically closer in vector space but semantically disconnected from "freshmen requirements". |
| **Dot Product** | *"Incoming Freshmen must submit: 1. Form 138 (Original) 2. Certificate of Good Moral Character..."* | **High Relevance.** Produced identical results to Cosine Similarity because our HuggingFace embeddings are pre-normalized to a length of 1. |

## 4. Analysis & Recommendation

**Recommendation: Cosine Similarity**

* **Why?** Since our TPC manual chunks vary significantly in length (e.g., a short list of documents vs. a long paragraph explaining retention policies), magnitude-sensitive metrics like Euclidean distance perform poorly. 
* Cosine Similarity successfully isolated the *meaning* of the queries regardless of how many words were in the retrieved chunks. 
* While Dot Product yielded identical results due to our specific model's normalization, Cosine Similarity remains the safest and most explicit configuration for our ChromaDB setup, ensuring robust semantic matching even if we change embedding models in the future.

**Implementation:** Our ChromaDB instance has been configured to use Cosine Similarity as its default similarity search algorithm for the RAG pipeline.