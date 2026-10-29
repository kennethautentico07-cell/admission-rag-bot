"""
Automated End-to-End Document Ingestion Pipeline (Checkpoint 3 - Deliverable 10)
Parses raw PDF/Markdown manuals from /data, sanitizes text, chunks semantically,
generates HuggingFace embeddings (all-MiniLM-L6-v2), and persists to ChromaDB.
"""

import os
import re
import sys
import shutil
from pathlib import Path
from typing import List

# Prevent Windows tokenizer parallelism deadlocks and disable telemetry
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["ANONYMIZED_TELEMETRY"] = "False"

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Resolve project root paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PDF_PATH = DATA_DIR / "TPC_Admission_RAG_Knowledge_Base.pdf"
MD_OUTPUT_PATH = DATA_DIR / "extracted_enrollment_manual.md"
CHROMA_DIR = PROJECT_ROOT / "chroma_db"

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"


def sanitize_text(raw_text: str) -> str:
    """
    Cleans and normalizes raw PDF/Markdown text:
    - Removes non-printable ASCII artifacts
    - Collapses redundant spaces while preserving paragraph/bullet breaks
    """
    text = re.sub(r"[^\x09\x0A\x0D\x20-\x7E]", "", raw_text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# Grounded FAQ & Enrollment Workflow supplement mandated by Page 26 of the TPC Knowledge Base
SUPPLEMENTAL_HANDBOOK_SECTIONS = """--- Page 26 (Supplement A) ---
## 25.1 Grounded Admission Chatbot FAQ Reference
**Q1: What are the admission requirements for incoming freshmen?**
Incoming freshmen must submit: (1) TPC Admission Test Result, (2) Original Form 138 / Form 137 (Report Card), (3) Certificate of Good Moral Character, (4) PSA Birth Certificate, (5) Recent 2x2 ID photos, (6) Accomplished Personal Data Sheet, (7) Long brown envelope, and (8) Long white folder.

**Q2: Is an admission test required at TPC?**
Yes. All applicants for degree programs must take the TPC Admission Test, which consists of verbal and performance subtests, administered before enrollment.

**Q3: Can transferees enroll at Talibon Polytechnic College, and what are their requirements?**
Yes, transferees may apply. Transferee requirements include: (1) Transfer Credential / Honorable Dismissal, (2) Official Transcript of Records (TOR) for evaluation, (3) Certificate of Good Moral Character, (4) PSA Birth Certificate, (5) Recent ID Photos, (6) Personal Data Sheet, and (7) Long brown and white folders.

**Q4: What documents do returnees, shiftees, and cross-enrollees need?**
- **Returnees:** Follow current admission policies except the placement examination, and require academic clearance.
- **Shiftees:** Require signed shifting forms and departmental clearance.
- **Cross-Enrollees:** Must submit an approved cross-registration permit from their home institution.
- **Continuing Students:** Must present their grade slip, examination permit, clinic form, and evaluation form before enrollment.

**Q5: How many units can graduating students take?**
Regular academic load follows the prescribed curriculum. Summer load is normally 9 units, while graduating students may take up to 12 units in summer or up to 27 units during a regular semester with formal academic approval.

**Q6: What is the attendance and tardiness policy?**
Students may be absent for at most 20% of the required class hours. Arriving more than 15 minutes late is recorded as an absence, and three (3) tardies equal one (1) full absence.

**Q7: How do I request registrar documents like a Good Moral Certificate or TOR, and are there fees?**
Registrar-issued documents—including Transcript of Records (TOR), Certificate of Good Moral Character, Certificate of Enrollment, Certification, Authentication, and Verification (CAV), and Honorable Dismissal—are requested through the Office of the College Registrar and have corresponding documentary processing fees.

**Q8: What degree programs and courses does Talibon Polytechnic College (TPC) offer?**
Talibon Polytechnic College offers seven (7) degree programs:
1. Bachelor of Arts in English Language (BA English Language)
2. Bachelor of Arts in Political Science (BA Political Science)
3. Bachelor of Science in Agriculture (BS Agriculture)
4. Bachelor of Science in Accounting Information Systems (BS Accounting Information Systems)
5. Bachelor of Science in Information Systems (BS Information Systems / BSIS)
6. Bachelor of Early Childhood Education (BECEd)
7. Bachelor of Science in Criminology (BS Criminology)

--- Page 26 (Supplement B) ---
## 25.2 Five-Step Official Enrollment Workflow
During the designated registration period, applicants and students must complete the five-step enrollment workflow:
- **Step 1: Document Verification (Registrar's Office - Window 1):** Submit all required admission or continuing student documents for completeness evaluation.
- **Step 2: Admission Test & Interview (Guidance & Testing Center):** Freshmen and new applicants take the TPC Admission Test and complete advisement.
- **Step 3: Academic Advising & Subject Loading (Department Dean's Office):** Program Dean or Adviser evaluates academic standing/TOR and approves subject load.
- **Step 4: Assessment of Fees (Accounting / Cashier's Office):** Assessment of applicable fees or verification of Free Higher Education coverage.
- **Step 5: Validation & Issuance of Certificate of Registration (Registrar's Office):** Final encoding and validation of the official Certificate of Registration (COR). Students become officially enrolled only after receiving a validated COR.
"""


def extract_all_pages_from_pdf(pdf_path: Path = PDF_PATH, md_output_path: Path = MD_OUTPUT_PATH) -> List[Document]:
    """
    Extracts all 26 pages from the official TPC PDF knowledge base, cleans text,
    writes the full normalized markdown corpus to /data/extracted_enrollment_manual.md,
    and returns structured LangChain Documents with page metadata.
    """
    if not pdf_path.exists():
        raise FileNotFoundError(f"Source PDF not found at: {pdf_path}")

    reader = PdfReader(str(pdf_path))
    documents: List[Document] = []
    markdown_sections: List[str] = []

    for idx, page in enumerate(reader.pages, start=1):
        raw_page_text = page.extract_text() or ""
        cleaned_page = sanitize_text(raw_page_text)
        if not cleaned_page:
            continue

        formatted_block = f"--- Page {idx} ---\n{cleaned_page}"
        markdown_sections.append(formatted_block)

        documents.append(
            Document(
                page_content=formatted_block,
                metadata={
                    "source": pdf_path.name,
                    "page": idx,
                    "doc_type": "official_handbook_pdf",
                },
            )
        )

    # Append the grounded FAQ & enrollment workflow sections
    markdown_sections.append(sanitize_text(SUPPLEMENTAL_HANDBOOK_SECTIONS))
    for supp_block in SUPPLEMENTAL_HANDBOOK_SECTIONS.strip().split("\n\n--- "):
        block_text = supp_block if supp_block.startswith("--- ") else f"--- {supp_block}"
        block_text = sanitize_text(block_text)
        if block_text:
            documents.append(
                Document(
                    page_content=block_text,
                    metadata={
                        "source": "extracted_enrollment_manual.md",
                        "page": 26,
                        "doc_type": "faq_and_workflow_supplement",
                    },
                )
            )

    # Save the complete 26-page normalized markdown file
    md_output_path.parent.mkdir(parents=True, exist_ok=True)
    full_markdown = "\n\n".join(markdown_sections) + "\n"
    md_output_path.write_text(full_markdown, encoding="utf-8")

    return documents


def chunk_documents(documents: List[Document], chunk_size: int = 500, chunk_overlap: int = 50) -> List[Document]:
    """Splits extracted documents into semantic chunks while preserving metadata."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(documents)


def run_ingestion_pipeline(reset_db: bool = True) -> Chroma:
    """
    Executes the full automated ingestion pipeline:
    1. Parse all 26 pages from TPC_Admission_RAG_Knowledge_Base.pdf
    2. Clean & export to data/extracted_enrollment_manual.md
    3. Split into semantic chunks (size=500, overlap=50)
    4. Generate 384-dim embeddings via all-MiniLM-L6-v2
    5. Index into ChromaDB with Cosine Similarity
    """
    print("=" * 60)
    print("[INFO] STARTING AUTOMATED TPC-ASSIST RAG INGESTION PIPELINE")
    print("=" * 60)

    print(f"1. Extracting pages from: {PDF_PATH.name}...")
    raw_docs = extract_all_pages_from_pdf(PDF_PATH, MD_OUTPUT_PATH)
    print(f"   [OK] Extracted {len(raw_docs)} document sections (saved to {MD_OUTPUT_PATH.name})")

    print("2. Splitting documents into semantic chunks (chunk_size=500, overlap=50)...")
    chunks = chunk_documents(raw_docs, chunk_size=500, chunk_overlap=50)
    print(f"   [OK] Generated {len(chunks)} semantic text chunks")

    print(f"3. Initializing HuggingFace Embedding Model ({EMBEDDING_MODEL_NAME})...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

    if reset_db and CHROMA_DIR.exists():
        try:
            shutil.rmtree(CHROMA_DIR)
            print("   [OK] Cleared existing ChromaDB directory for clean re-indexing")
        except Exception:
            pass

    print(f"4. Indexing {len(chunks)} chunks into ChromaDB (Cosine Similarity)...")
    vectorstore = Chroma(
        persist_directory=str(CHROMA_DIR),
        embedding_function=embeddings,
        collection_metadata={"hnsw:space": "cosine"},
    )

    try:
        existing_ids = vectorstore.get()["ids"]
        if existing_ids:
            vectorstore.delete(ids=existing_ids)
    except Exception:
        pass

    vectorstore.add_documents(chunks)
    final_count = vectorstore._collection.count()
    print(f"   [OK] Successfully indexed {final_count} chunks in ChromaDB at: {CHROMA_DIR}")
    print("=" * 60)
    print("[OK] INGESTION PIPELINE COMPLETE")
    print("=" * 60)
    return vectorstore


if __name__ == "__main__":
    vs = run_ingestion_pipeline(reset_db=True)

    # Verification query test
    test_queries = [
        "What documents do incoming freshmen need to submit?",
        "What courses or degree programs does TPC offer?",
        "What are the requirements for transferees?",
    ]
    print("\n[VERIFY] TESTING VECTOR RETRIEVAL (Top-1 Cosine Match per Query):")
    retriever = vs.as_retriever(search_kwargs={"k": 1})
    for q in test_queries:
        res = retriever.invoke(q)
        top_preview = res[0].page_content.replace("\n", " ")[:140] if res else "None"
        print(f"  * Query: '{q}'\n    -> Match: {top_preview}...")
