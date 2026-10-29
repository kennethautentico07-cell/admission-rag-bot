"""
Checkpoint 2 — Deliverable 7: Chat Completion API Integration Demo
Sends structured grounded prompts to the Google Gemini Chat Completion API
and demonstrates both grounded in-scope answers and strict out-of-scope refusals.
"""

import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

# Ensure GOOGLE_API_KEY is mapped if GEMINI_API_KEY is provided
if not os.getenv("GOOGLE_API_KEY") and os.getenv("GEMINI_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = os.environ["GEMINI_API_KEY"]

SAMPLE_HANDBOOK_CONTEXT = """
--- Page 12: 11. Freshman Admission Requirements ---
Incoming freshmen applying to Talibon Polytechnic College must submit:
1. TPC Admission Test Result
2. Form 138 / Form 137 (Original Report Card)
3. Certificate of Good Moral Character
4. PSA Birth Certificate
5. Recent 2x2 ID photos
6. Accomplished Personal Data Sheet
7. Long brown envelope and long white folder
"""

GROUNDED_PROMPT = ChatPromptTemplate.from_template(
    """You are an official academic assistant for Talibon Polytechnic College (TPC).
Answer the question strictly based on the provided context below.

Guidelines:
1. Do not use outside knowledge or assume policies not stated in the context.
2. Maintain a professional, polite, and concise tone.
3. If the context does not contain sufficient information to answer the question, state clearly:
   "I cannot find this information in the official TPC manual."

Context:
{context}

Question: {question}
"""
)


def run_api_demo():
    models_to_try = [
        ("gemini-3-flash-preview", {"thinking_budget": 0}),
        ("gemini-3.1-flash-lite", {"thinking_budget": 0}),
        ("gemini-3.5-flash-lite", {"thinking_level": "minimal"}),
        ("gemini-3.6-flash", {"thinking_budget": 0}),
    ]

    active_chain = None
    active_model = None
    for model_name, kwargs in models_to_try:
        try:
            llm = ChatGoogleGenerativeAI(model=model_name, max_retries=1, **kwargs)
            active_chain = GROUNDED_PROMPT | llm | StrOutputParser()
            active_model = model_name
            break
        except Exception:
            continue

    if active_chain is None:
        raise RuntimeError("Could not initialize ChatGoogleGenerativeAI. Check GOOGLE_API_KEY in .env.")

    print(f"=== TPC-ASSIST CHAT COMPLETION API DEMO (Model: {active_model}) ===\n")

    in_scope_q = "What documents do incoming freshmen need to submit?"
    print(f"[Test 1 - In-Scope Query]: {in_scope_q}")
    ans1 = active_chain.invoke({"context": SAMPLE_HANDBOOK_CONTEXT, "question": in_scope_q})
    print(f"[Grounded Response]:\n{ans1}\n")

    out_of_scope_q = "How do I apply for a student visa to study in Japan?"
    print(f"[Test 2 - Out-of-Scope Query]: {out_of_scope_q}")
    ans2 = active_chain.invoke({"context": SAMPLE_HANDBOOK_CONTEXT, "question": out_of_scope_q})
    print(f"[Guardrail Refusal Response]:\n{ans2}\n")


if __name__ == "__main__":
    run_api_demo()
