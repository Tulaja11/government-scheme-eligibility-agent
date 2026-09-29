"""
RAG Chain: Retrieves relevant schemes from ChromaDB and uses LLM to check eligibility.
"""

from dotenv import load_dotenv
load_dotenv()

import os
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

CHROMA_PATH = os.path.join(os.path.dirname(__file__), "..", "chroma_db")

ELIGIBILITY_PROMPT = PromptTemplate(
    input_variables=["context", "profile", "question"],
    template="""You are a Government Scheme Eligibility Assistant for Indian citizens.

Given the user's profile and the retrieved scheme information, do the following:

1. Check which schemes the user is eligible for based on their profile.
2. For each scheme, clearly state:
   - ELIGIBLE: if the user clearly meets all criteria
   - POSSIBLY ELIGIBLE: if some criteria match but others are unclear or missing
   - NOT ELIGIBLE: if the user clearly does not meet a criterion
3. Cite the specific eligibility criteria that led to your decision.
4. If the user's profile is incomplete, mention which missing details would help.

USER PROFILE:
{profile}

USER QUESTION:
{question}

RETRIEVED SCHEME INFORMATION:
{context}

Provide a clear, structured response with scheme names, eligibility status, reasoning, and source citations.
Do NOT make up schemes that are not in the retrieved information.
If no schemes match, say so honestly.
"""
)


def get_vectorstore():
    """Load the existing ChromaDB vector store."""
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)


def get_llm():
    """Initialize Groq LLM."""
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not set. Add it to your .env file.")
    return ChatGroq(
        model_name="openai/gpt-oss-20b",
        api_key=api_key,
        temperature=0.1
    )


def format_docs(docs):
    """Format retrieved documents with source info."""
    formatted = []
    for i, doc in enumerate(docs, 1):
        name = doc.metadata.get("scheme_name", "Unknown")
        category = doc.metadata.get("category", "")
        level = doc.metadata.get("level", "")
        formatted.append(
            f"--- Scheme {i}: {name} [{level}] ---\n"
            f"Category: {category}\n{doc.page_content}\n"
        )
    return "\n".join(formatted)


def query_schemes(profile, question, top_k=10):
    """
    Given a user profile and question, retrieve relevant schemes
    and check eligibility using the LLM.
    """
    search_parts = []
    if profile.get("state"):
        search_parts.append(profile["state"])
    if profile.get("category"):
        search_parts.append(f"{profile['category']} category")
    if profile.get("employment"):
        search_parts.append(profile["employment"])
    if profile.get("gender"):
        search_parts.append(profile["gender"])

    search_query = f"{question} {' '.join(search_parts)}"

    vectorstore = get_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})
    docs = retriever.invoke(search_query)

    profile_str = "\n".join(
        f"  {k}: {v}" for k, v in profile.items()
        if v is not None and not k.startswith("_")
    )
    if profile.get("_missing_fields"):
        profile_str += f"\n  Missing info: {', '.join(profile['_missing_fields'])}"

    context = format_docs(docs)
    llm = get_llm()
    prompt = ELIGIBILITY_PROMPT.format(context=context, profile=profile_str, question=question)
    response = llm.invoke(prompt)

    # Extract unique sources with URLs
    sources = {}
    for doc in docs:
        name = doc.metadata.get("scheme_name", "")
        url = doc.metadata.get("source_url", "")
        if name and name not in sources:
            sources[name] = url

    return {
        "answer": response.content,
        "sources": list(sources.keys()),
        "source_urls": sources,
        "profile_used": profile,
        "schemes_searched": len(docs),
    }


if __name__ == "__main__":
    from profile_extractor import extract_profile

    test_input = "I'm a 22 year old OBC woman from Maharashtra, income 3 lakh, unemployed"
    print(f"Input: {test_input}\n")

    profile = extract_profile(test_input)
    print("Extracted Profile:")
    for k, v in profile.items():
        if not k.startswith("_"):
            print(f"  {k}: {v}")

    print("\nSearching for eligible schemes...\n")
    result = query_schemes(profile=profile, question="What government schemes am I eligible for?")

    print("=" * 60)
    print(result["answer"])
    print("=" * 60)
    print(f"\nSchemes referenced:")
    for name, url in result["source_urls"].items():
        print(f"  {name}: {url}")