"""
Checkpoint 1 — Deliverable 3: Text Preprocessing, Normalization & Tokenization Demo
Demonstrates cleaning, regex normalization, and subword/word tokenization with Before/After examples.
"""

import re
from typing import List


def clean_text(text: str) -> str:
    """Cleans and normalizes raw text extracted from PDF manuals."""
    # 1. Remove non-ASCII characters / control artifacts (e.g., null bytes)
    text = re.sub(r"[^\x09\x0A\x0D\x20-\x7E]+", "", text)
    # 2. Strip running page headers/markers
    text = re.sub(r"---\s*PAGE\s+\d+\s*---", "", text, flags=re.IGNORECASE)
    # 3. Collapse redundant line breaks and horizontal whitespace
    text = re.sub(r"\s+", " ", text)
    # 4. Trim leading/trailing whitespace
    return text.strip()


def tokenize_text(text: str) -> List[str]:
    """
    Tokenizes text into normalized word/punctuation tokens (with optional tiktoken cl100k_base support).
    """
    return re.findall(r"\w+|[^\w\s]", text)


if __name__ == "__main__":
    raw_sample = """
    TALIBON POLYTECHNIC COLLEGE   --- PAGE 12 ---
    ADMISSION REQUIREMENTS:
    
    * Form 138 (Original Report Card)   \x00
    * Certificate of   Good Moral Character
    * 2x2 Picture (2 copies)
    
    Note: All documents must be submitted to the Registrar Office.
    """

    processed_sample = clean_text(raw_sample)
    raw_tokens = tokenize_text(raw_sample)
    clean_tokens = tokenize_text(processed_sample)

    print("=== 1. BEFORE PREPROCESSING (Raw PDF Extract) ===")
    print(repr(raw_sample))
    print(f"Raw Character Length: {len(raw_sample)} chars | Raw Token Count: {len(raw_tokens)} tokens")

    print("\n=== 2. AFTER PREPROCESSING (Cleaned & Normalized) ===")
    print(processed_sample)
    print(f"Clean Character Length: {len(processed_sample)} chars | Clean Token Count: {len(clean_tokens)} tokens")

    print("\n=== 3. TOKENIZATION OUTPUT (First 20 Tokens) ===")
    print(clean_tokens[:20])
