# Model Customization Note: PEFT (LoRA/QLoRA) vs. Base Instruction-Tuned Model with RAG

**Project Title:** TPC-Assist: A RAG-Based Admission and Policy Chatbot  
**Course:** IS Professional Elective #4 — Generative AI Systems  
**Instructor:** Jessie A. Melendres  
**Chosen Model:** Google Gemini 3.8 Flash (`gemini-3.8-flash`, Base Instruction-Tuned)  

---

## 1. Overview & CILO 3 Context

This technical note analyzes whether Parameter-Efficient Fine-Tuning (**PEFT**) techniques—specifically **LoRA (Low-Rank Adaptation)** and **QLoRA (Quantized LoRA)**—should be applied to an open-source foundation model (such as `Llama-3-8B-Instruct` or `Mistral-7B-Instruct`) for **TPC-Assist**, or whether pairing **Retrieval-Augmented Generation (RAG)** with a frozen instruction-tuned model (**Google Gemini 3.8 Flash**) is architecturally superior.

---

## 2. Technical Mechanics & Memory Constraints of LoRA / QLoRA

Full fine-tuning of an 8-billion-parameter LLM in 16-bit precision ($\text{FP16}/\text{BF16}$) requires storing model weights ($\approx 16\text{ GB}$), gradients ($\approx 16\text{ GB}$), and AdamW optimizer states ($\approx 32\text{ GB}$), totaling **$>64\text{ GB of VRAM}$**—far beyond consumer or student GPU hardware.

* **LoRA (Low-Rank Adaptation):** Freezes the pre-trained weight matrix $W_0 \in \mathbb{R}^{d \times k}$ and injects trainable low-rank decomposition matrices $B \in \mathbb{R}^{d \times r}$ and $A \in \mathbb{R}^{r \times k}$ (where rank $r \ll \min(d, k)$, typically $r = 8$ or $16$, and scaling factor $\alpha = 16$ or $32$):
  $$h = W_0 x + \Delta W x = W_0 x + \frac{\alpha}{r} B A x$$
  This reduces trainable parameters by $>99\%$ (to $\approx 10\text{–}20\text{M}$ parameters).
* **QLoRA (4-Bit Quantized LoRA):** Compresses the frozen base model weights $W_0$ into **4-bit NormalFloat ($\text{NF4}$)** precision using double quantization and paged optimizers, bringing the training VRAM footprint of an 8B model down to **$\approx 6\text{–}8\text{ GB VRAM}$** while computing forward/backward passes in $\text{BF16}$.

### Potential Role of QLoRA in TPC-Assist
If we fine-tuned a local model with QLoRA, the primary target would **not** be memorizing handbook facts, but rather:
1. Teaching a small local model to strictly adhere to the **TPC Registrar advising tone** and refusal format (`"I cannot find this information in the official TPC manual."`).
2. Training a domain-adapted embedding or cross-encoder reranker on TPC-specific query-document pairs.

---

## 3. Why RAG + Frozen Instruction-Tuned Model Outperforms Fine-Tuning for Campus Policies

Despite QLoRA making fine-tuning feasible on a single GPU, fine-tuning is the **wrong mechanism for injecting factual institutional policies**:

| Evaluation Dimension | PEFT / QLoRA Fine-Tuning | Frozen Instruction-Tuned Model + RAG (Chosen) |
| :--- | :--- | :--- |
| **Policy Updatability** | **Static / Stale:** If TPC updates admission checklists or unit limits next semester, adapter weights must be re-trained and re-deployed. | **Instantaneous:** Updating policies only requires updating the PDF/Markdown in `/data` and running `python src/ingest.py` ($<5\text{ seconds}$). |
| **Hallucination Control** | **High Risk:** Fine-tuned weights compress facts lossily and cannot deterministically abstain when asked an out-of-scope question. | **Strict Refusal:** By passing exact ChromaDB chunks at `temperature=0`, the model grounds every statement in verbatim text or abstains. |
| **Source Attribution** | **Opaque:** A fine-tuned model cannot cite which page of the student handbook a generated answer came from. | **100% Auditable:** Every response includes an expandable drawer displaying the exact retrieved handbook chunks. |
| **Deployment Footprint** | **GPU Required:** Serving a 4-bit 8B model still requires $\ge 6\text{ GB VRAM}$ in production container environments. | **CPU Lightweight:** Local CPU embeddings (`all-MiniLM-L6-v2`) + Gemini 3.8 Flash API run inside a minimal CPU Docker container. |

---

## 4. Conclusion

For an institutional admission and policy assistant where **factual accuracy, source verifiability, zero-hallucination compliance, and rapid policy updates** are paramount, **RAG over a frozen instruction-tuned foundation model (`gemini-3.8-flash`) is structurally superior to PEFT/LoRA weight adaptation**.
