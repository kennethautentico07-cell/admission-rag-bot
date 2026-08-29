# Model Customization Note: PEFT/LoRA vs. Base Instruction-Tuned Model

**Project Title:** TPC-Assist: A RAG-Based Admission Chatbot  
**Course:** IS Professional Elective 4 — Generative AI Systems  
**Chosen Model:** Google Gemini 3.6 Flash (Base Instruction-Tuned)  

---

## 1. Overview
This document outlines the technical rationale for utilizing a base instruction-tuned Large Language Model (Google Gemini 3.6 Flash) in conjunction with Retrieval-Augmented Generation (RAG), rather than applying Parameter-Efficient Fine-Tuning (PEFT) techniques such as LoRA or QLoRA to a localized open-source model.

## 2. Analysis of PEFT (LoRA/QLoRA) for this Use Case
Parameter-Efficient Fine-Tuning (PEFT), utilizing Low-Rank Adaptation (LoRA) or Quantized LoRA (QLoRA), is highly effective for altering a model's tone, formatting style, or domain-specific reasoning behavior without updating all model weights. 

**Potential benefits for TPC-Assist:**
* We could theoretically train a local model (e.g., Llama 3 8B) via QLoRA to adopt a highly specific "TPC Administrator" persona and learn institutional jargon.

**Drawbacks and Limitations:**
* **Knowledge Cutoff & Stale Data:** Fine-tuning bakes factual information directly into the model's weights. If TPC changes its admission requirements next semester, the model would need to be retrained. 
* **Resource Intensive:** Creating a high-quality dataset of hundreds of QA pairs for training, alongside the GPU compute required for QLoRA, exceeds the practical constraints of a lightweight semester project.
* **Hallucination Risk:** Fine-tuning does not inherently prevent a model from hallucinating policies. It only makes the model sound more confident when doing so.

## 3. Justification for Using the Base Model As-Is (with RAG)
Instead of fine-tuning, TPC-Assist relies entirely on **RAG (Retrieval-Augmented Generation)** paired with **Prompt Engineering** on an already powerful instruction-tuned model (Gemini 3.6 Flash).

1. **Dynamic Fact Retrieval:** By keeping the model weights frozen and fetching facts from ChromaDB, updating the chatbot's knowledge base is as simple as uploading a new PDF to the vector database. No model retraining is required.
2. **Strong Zero-Shot Instruction Following:** Gemini 3.6 Flash is heavily pre-instruction-tuned. It strictly follows our systemic guardrails (e.g., "Answer strictly based on context") without the need for additional LoRA adaptation to enforce compliance.
3. **Cost & Latency Efficiency:** API-based inference with a highly optimized model avoids the local VRAM bottlenecks associated with hosting and running a quantized fine-tuned model.

## 4. Conclusion
For an institutional policy chatbot where factual accuracy, updatability, and hallucination prevention are paramount, **RAG with a frozen instruction-tuned model is structurally superior to PEFT/LoRA**. Fine-tuning would introduce unnecessary overhead and data staleness without providing meaningful improvements to retrieval accuracy.