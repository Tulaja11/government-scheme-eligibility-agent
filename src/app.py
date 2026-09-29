"""
FastAPI app for the Government Scheme Eligibility Agent.
Run: uvicorn src.app:app --reload
"""

from dotenv import load_dotenv
load_dotenv()

import sqlite3
import json
import os
from datetime import datetime
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.profile_extractor import extract_profile
from src.rag_chain import query_schemes

# --- SQLite Setup ---
DB_PATH = os.path.join(os.path.dirname(__file__), "..", "query_logs.db")


def init_db():
    """Create the query logs table if it doesn't exist."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS query_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            input_text TEXT,
            profile_extracted TEXT,
            schemes_found INTEGER,
            top_sources TEXT
        )
    """)
    conn.commit()
    conn.close()


def log_query(input_text, profile, schemes_found, sources):
    """Log each query to SQLite for analytics."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO query_logs (timestamp, input_text, profile_extracted, schemes_found, top_sources) VALUES (?, ?, ?, ?, ?)",
        (
            datetime.now().isoformat(),
            input_text,
            json.dumps(profile, default=str),
            schemes_found,
            json.dumps(sources[:5]),
        ),
    )
    conn.commit()
    conn.close()


# Initialize DB on startup
init_db()

# --- FastAPI App ---
app = FastAPI(
    title="Government Scheme Eligibility Agent",
    description=(
        "An AI-powered agent that helps Indian citizens discover government schemes "
        "they're eligible for. Uses spaCy NER for profile extraction and "
        "LangChain RAG for scheme matching."
    ),
    version="1.0.0",
)


class NaturalLanguageRequest(BaseModel):
    text: str

    class Config:
        json_schema_extra = {
            "example": {
                "text": "I'm a 22 year old OBC woman from Maharashtra, income 3 lakh, unemployed"
            }
        }


class StructuredRequest(BaseModel):
    age: int | None = None
    gender: str | None = None
    category: str | None = None
    state: str | None = None
    income: int | None = None
    employment: str | None = None
    question: str = "What government schemes am I eligible for?"

    class Config:
        json_schema_extra = {
            "example": {
                "age": 22,
                "gender": "Female",
                "category": "OBC",
                "state": "Maharashtra",
                "income": 300000,
                "employment": "Unemployed",
                "question": "What government schemes am I eligible for?"
            }
        }


class EligibilityResponse(BaseModel):
    answer: str
    profile_used: dict
    sources: list[str]
    schemes_searched: int


@app.get("/")
def root():
    return {
        "message": "Government Scheme Eligibility Agent API",
        "docs": "Go to /docs for the interactive Swagger UI",
        "endpoints": {
            "/find-schemes": "POST - Natural language input",
            "/find-schemes-structured": "POST - Structured profile input",
            "/extract-profile": "POST - Test the NER profile extractor",
            "/analytics": "GET - View query logs and stats",
        },
    }


@app.post("/extract-profile")
def extract_profile_endpoint(request: NaturalLanguageRequest):
    """Test the NER profile extractor alone."""
    profile = extract_profile(request.text)
    return {"input_text": request.text, "extracted_profile": profile}


@app.post("/find-schemes", response_model=EligibilityResponse)
def find_schemes_natural(request: NaturalLanguageRequest):
    """Find eligible schemes from natural language input."""
    profile = extract_profile(request.text)

    if profile["_fields_extracted"] < 2:
        raise HTTPException(
            status_code=400,
            detail=f"Could only extract {profile['_fields_extracted']} profile fields. Please provide more details.",
        )

    result = query_schemes(profile=profile, question=request.text)

    # Log to SQLite
    log_query(request.text, profile, result["schemes_searched"], result["sources"])

    return EligibilityResponse(**result)


@app.post("/find-schemes-structured", response_model=EligibilityResponse)
def find_schemes_structured(request: StructuredRequest):
    """Find eligible schemes from structured profile data."""
    profile = {
        "age": request.age,
        "gender": request.gender,
        "category": request.category,
        "state": request.state,
        "income": request.income,
        "employment": request.employment,
    }

    filled = sum(1 for v in profile.values() if v is not None)
    if filled < 2:
        raise HTTPException(status_code=400, detail="Please provide at least 2 profile fields.")

    profile["_fields_extracted"] = filled
    profile["_missing_fields"] = [k for k, v in profile.items() if v is None and not k.startswith("_")]

    result = query_schemes(profile=profile, question=request.question)

    # Log to SQLite
    log_query(str(profile), profile, result["schemes_searched"], result["sources"])

    return EligibilityResponse(**result)


@app.get("/analytics")
def get_analytics():
    """View query logs and usage stats."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.execute("SELECT COUNT(*) FROM query_logs")
    total_queries = cursor.fetchone()[0]

    cursor = conn.execute("SELECT * FROM query_logs ORDER BY id DESC LIMIT 10")
    columns = [desc[0] for desc in cursor.description]
    recent = [dict(zip(columns, row)) for row in cursor.fetchall()]

    conn.close()

    return {
        "total_queries": total_queries,
        "recent_queries": recent,
    }