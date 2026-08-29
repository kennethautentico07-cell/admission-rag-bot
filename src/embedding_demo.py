from langchain_huggingface import HuggingFaceEmbeddings

def run_embedding_demo():
    # Model Choice: all-MiniLM-L6-v2 (384 dimensions)
    # Rationale: Lightweight, optimized for CPU execution, and highly accurate for English semantic search.
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    sample_text = "Incoming freshmen must submit Form 138 and a Good Moral Certificate."
    vector = embedding_model.embed_query(sample_text)
    
    print(f"Sample Text: '{sample_text}'")
    print(f"Generated Vector Dimension: {len(vector)}")
    print(f"Sample Vector Values (First 5 dimensions): {vector[:5]}")

if __name__ == "__main__":
    run_embedding_demo()