import re

def clean_text(text: str) -> str:
    """Cleans and normalizes raw text extracted from PDF manuals."""
    # 1. Strip redundant line breaks and spaces
    text = re.sub(r'\s+', ' ', text)
    # 2. Remove non-ASCII characters / artifacts
    text = re.compile(r'[^\x00-\x7F]+').sub('', text)
    # 3. Trim leading/trailing whitespace
    return text.strip()

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

    print("=== BEFORE PREPROCESSING ===")
    print(repr(raw_sample))
    print("\n=== AFTER PREPROCESSING ===")
    print(processed_sample)