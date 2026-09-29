"""
Extract user profile from natural language using spaCy NER + rule-based matching.
This is the ML component of the project.

Example input:  "I'm a 22 year old OBC woman from Maharashtra, income 3 lakh, unemployed"
Example output: {"age": 22, "gender": "Female", "category": "OBC", "state": "Maharashtra",
                 "income": 300000, "employment": "Unemployed"}
"""

import spacy
import re

nlp = spacy.load("en_core_web_sm")

# Indian states and UTs
INDIAN_STATES = [
    "andhra pradesh", "arunachal pradesh", "assam", "bihar", "chhattisgarh",
    "goa", "gujarat", "haryana", "himachal pradesh", "jharkhand", "karnataka",
    "kerala", "madhya pradesh", "maharashtra", "manipur", "meghalaya",
    "mizoram", "nagaland", "odisha", "punjab", "rajasthan", "sikkim",
    "tamil nadu", "telangana", "tripura", "uttar pradesh", "uttarakhand",
    "west bengal", "delhi", "jammu and kashmir", "ladakh", "puducherry",
    "chandigarh", "dadra and nagar haveli", "daman and diu", "lakshadweep",
    "andaman and nicobar",
]

# Social categories
CATEGORIES = ["general", "obc", "sc", "st", "ews", "sbc", "nt", "vjnt", "sebc"]

# Gender keywords
GENDER_MAP = {
    "male": "Male", "man": "Male", "boy": "Male", "father": "Male",
    "female": "Female", "woman": "Female", "girl": "Female", "mother": "Female",
    "transgender": "Transgender", "trans": "Transgender",
}

# Employment keywords
EMPLOYMENT_MAP = {
    "unemployed": "Unemployed", "jobless": "Unemployed", "no job": "Unemployed",
    "employed": "Employed", "working": "Employed", "job": "Employed",
    "student": "Student", "studying": "Student", "college": "Student",
    "self-employed": "Self-Employed", "self employed": "Self-Employed",
    "business": "Self-Employed", "freelance": "Self-Employed",
    "farmer": "Farmer", "farming": "Farmer", "agriculture": "Farmer",
    "retired": "Retired",
}


def extract_age(text):
    patterns = [
        r"(\d{1,2})\s*(?:year|yr|yrs)[\s-]*old",
        r"age\s*(?:is|:)?\s*(\d{1,2})",
        r"i(?:'m| am)\s*(\d{1,2})",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            age = int(match.group(1))
            if 1 <= age <= 120:
                return age
    return None


def extract_income(text):
    patterns = [
        r"(\d+(?:\.\d+)?)\s*(?:lakh|lac|lpa|l)\b",
        r"(?:income|salary|earn(?:ing)?)\s*(?:is|:)?\s*(?:rs\.?|₹|inr)?\s*(\d[\d,]*)",
        r"(?:rs\.?|₹|inr)\s*(\d[\d,]*)",
    ]
    for i, pattern in enumerate(patterns):
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            amount = float(match.group(1).replace(",", ""))
            if i == 0:
                return int(amount * 100000)
            return int(amount)
    return None


def extract_state(text):
    text_lower = text.lower()
    for state in INDIAN_STATES:
        if state in text_lower:
            return state.title()
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ == "GPE":
            for state in INDIAN_STATES:
                if ent.text.lower() in state or state in ent.text.lower():
                    return state.title()
    return None


def extract_gender(text):
    text_lower = text.lower()
    for keyword, gender in GENDER_MAP.items():
        if re.search(r"\b" + keyword + r"\b", text_lower):
            return gender
    return None


def extract_category(text):
    text_lower = text.lower()
    for cat in CATEGORIES:
        if re.search(r"\b" + cat + r"\b", text_lower):
            return cat.upper()
    return None


def extract_employment(text):
    text_lower = text.lower()
    for keyword, status in EMPLOYMENT_MAP.items():
        if keyword in text_lower:
            return status
    return None


def extract_profile(text):
    """Main function: Extract complete user profile from natural language."""
    profile = {
        "age": extract_age(text),
        "gender": extract_gender(text),
        "category": extract_category(text),
        "state": extract_state(text),
        "income": extract_income(text),
        "employment": extract_employment(text),
    }
    extracted = sum(1 for v in profile.values() if v is not None)
    profile["_fields_extracted"] = extracted
    profile["_missing_fields"] = [k for k, v in profile.items() if v is None and not k.startswith("_")]
    return profile


# --- Quick test ---
if __name__ == "__main__":
    test_inputs = [
        "I'm a 22 year old OBC woman from Maharashtra, income 3 lakh, unemployed",
        "25 yr old male, SC category, living in Delhi, earning 5 lpa",
        "I am a farmer from Punjab, age 45, general category, income around 2.5 lakh",
        "Student, 19 years old, from Kerala, ST, no income",
    ]

    for text in test_inputs:
        print(f"\nInput: {text}")
        profile = extract_profile(text)
        for key, val in profile.items():
            if not key.startswith("_"):
                print(f"  {key}: {val}")
        print(f"  Extracted: {profile['_fields_extracted']}/6 fields")