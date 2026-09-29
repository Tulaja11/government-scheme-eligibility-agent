"""
Evaluation Harness for the Government Scheme Eligibility Agent.
Tests retrieval accuracy and answer quality against known ground-truth cases.

Run: python src/evaluate.py
"""

from dotenv import load_dotenv
load_dotenv()

import json
import time
from profile_extractor import extract_profile
from rag_chain import query_schemes

# --- 25 Test Cases with Expected Schemes ---
TEST_CASES = [
    {
        "input": "I'm a 22 year old OBC woman from Maharashtra, income 3 lakh, unemployed",
        "expected_profile": {"age": 22, "gender": "Female", "category": "OBC", "state": "Maharashtra"},
        "should_mention": ["Majhi Ladki Bahin", "Swarnima"],
        "category": "young_female_obc",
    },
    {
        "input": "25 year old SC male from Delhi, income 2 lakh, unemployed",
        "expected_profile": {"age": 25, "gender": "Male", "category": "SC", "state": "Delhi"},
        "should_mention": ["NBCFDC", "Stand Up India"],
        "category": "young_male_sc",
    },
    {
        "input": "I am a farmer from Punjab, age 45, general category, income around 2.5 lakh",
        "expected_profile": {"age": 45, "category": "GENERAL", "state": "Punjab", "employment": "Farmer"},
        "should_mention": ["PM Kisan", "Kisan"],
        "category": "farmer",
    },
    {
        "input": "Student, 19 years old, from Kerala, ST category, no income",
        "expected_profile": {"age": 19, "category": "ST", "state": "Kerala", "employment": "Student"},
        "should_mention": ["Scholarship", "Education"],
        "category": "student_st",
    },
    {
        "input": "35 year old OBC woman from Rajasthan, self-employed, income 1.5 lakh",
        "expected_profile": {"age": 35, "gender": "Female", "category": "OBC", "state": "Rajasthan"},
        "should_mention": ["Mudra", "Swarnima"],
        "category": "self_employed_female",
    },
    {
        "input": "I'm a 60 year old retired general category man from Tamil Nadu, pension 4 lakh",
        "expected_profile": {"age": 60, "gender": "Male", "category": "GENERAL", "state": "Tamil Nadu"},
        "should_mention": ["Pension", "Senior"],
        "category": "senior_citizen",
    },
    {
        "input": "28 year old EWS male from Uttar Pradesh, unemployed, income 1 lakh",
        "expected_profile": {"age": 28, "gender": "Male", "category": "EWS", "state": "Uttar Pradesh"},
        "should_mention": ["EWS", "Employment"],
        "category": "ews_male",
    },
    {
        "input": "I'm a 30 year old SC woman from Bihar, income 80000, unemployed",
        "expected_profile": {"age": 30, "gender": "Female", "category": "SC", "state": "Bihar"},
        "should_mention": ["SC", "Women"],
        "category": "sc_female_low_income",
    },
    {
        "input": "Student from Gujarat, 20 years old, OBC, family income 5 lakh",
        "expected_profile": {"age": 20, "category": "OBC", "state": "Gujarat", "employment": "Student"},
        "should_mention": ["Scholarship", "OBC"],
        "category": "student_obc",
    },
    {
        "input": "40 year old ST male farmer from Jharkhand, income 1 lakh",
        "expected_profile": {"age": 40, "gender": "Male", "category": "ST", "state": "Jharkhand"},
        "should_mention": ["Tribal", "Kisan"],
        "category": "tribal_farmer",
    },
    {
        "input": "I'm a 23 year old woman from Maharashtra, SC category, studying in college",
        "expected_profile": {"age": 23, "gender": "Female", "category": "SC", "state": "Maharashtra"},
        "should_mention": ["Scholarship", "SC"],
        "category": "sc_female_student",
    },
    {
        "input": "55 year old general category man from Karnataka, self-employed, income 6 lakh",
        "expected_profile": {"age": 55, "gender": "Male", "category": "GENERAL", "state": "Karnataka"},
        "should_mention": ["Loan", "Self"],
        "category": "senior_self_employed",
    },
    {
        "input": "18 year old ST girl from Odisha, student, family income 50000",
        "expected_profile": {"age": 18, "gender": "Female", "category": "ST", "state": "Odisha"},
        "should_mention": ["Scholarship", "Tribal"],
        "category": "tribal_girl_student",
    },
    {
        "input": "I am a 32 year old OBC male from Madhya Pradesh, unemployed, income 2 lakh",
        "expected_profile": {"age": 32, "gender": "Male", "category": "OBC", "state": "Madhya Pradesh"},
        "should_mention": ["OBC", "Loan"],
        "category": "obc_male_unemployed",
    },
    {
        "input": "27 year old woman from West Bengal, general category, working, income 4 lakh",
        "expected_profile": {"age": 27, "gender": "Female", "category": "GENERAL", "state": "West Bengal"},
        "should_mention": ["Women", "Employment"],
        "category": "working_woman",
    },
    {
        "input": "45 year old SC male from Andhra Pradesh, farmer, income 1.5 lakh",
        "expected_profile": {"age": 45, "gender": "Male", "category": "SC", "state": "Andhra Pradesh"},
        "should_mention": ["Kisan", "SC"],
        "category": "sc_farmer",
    },
    {
        "input": "21 year old EWS female student from Haryana, no income",
        "expected_profile": {"age": 21, "gender": "Female", "category": "EWS", "state": "Haryana"},
        "should_mention": ["EWS", "Scholarship"],
        "category": "ews_female_student",
    },
    {
        "input": "38 year old OBC woman from Chhattisgarh, self-employed, income 3 lakh",
        "expected_profile": {"age": 38, "gender": "Female", "category": "OBC", "state": "Chhattisgarh"},
        "should_mention": ["OBC", "Women"],
        "category": "obc_self_employed_woman",
    },
    {
        "input": "50 year old general category male from Telangana, retired, pension 5 lakh",
        "expected_profile": {"age": 50, "gender": "Male", "category": "GENERAL", "state": "Telangana"},
        "should_mention": ["Pension", "Senior"],
        "category": "retired_general",
    },
    {
        "input": "I'm a 24 year old SC male from Assam, studying engineering, family income 2 lakh",
        "expected_profile": {"age": 24, "gender": "Male", "category": "SC", "state": "Assam"},
        "should_mention": ["Scholarship", "SC"],
        "category": "sc_engineering_student",
    },
    {
        "input": "33 year old ST woman from Meghalaya, unemployed, income 60000",
        "expected_profile": {"age": 33, "gender": "Female", "category": "ST", "state": "Meghalaya"},
        "should_mention": ["Tribal", "Women"],
        "category": "st_woman_low_income",
    },
    {
        "input": "29 year old OBC male from Gujarat, business owner, income 8 lakh",
        "expected_profile": {"age": 29, "gender": "Male", "category": "OBC", "state": "Gujarat"},
        "should_mention": ["Mudra", "OBC"],
        "category": "obc_business_owner",
    },
    {
        "input": "I am a 19 year old girl from Uttar Pradesh, SC category, studying in 12th",
        "expected_profile": {"age": 19, "gender": "Female", "category": "SC", "state": "Uttar Pradesh"},
        "should_mention": ["Scholarship", "SC"],
        "category": "sc_girl_12th",
    },
    {
        "input": "42 year old general category woman from Himachal Pradesh, farmer, income 2 lakh",
        "expected_profile": {"age": 42, "gender": "Female", "category": "GENERAL", "state": "Himachal Pradesh"},
        "should_mention": ["Kisan", "Farmer"],
        "category": "woman_farmer",
    },
    {
        "input": "26 year old ST male from Nagaland, unemployed, income 1 lakh",
        "expected_profile": {"age": 26, "gender": "Male", "category": "ST", "state": "Nagaland"},
        "should_mention": ["Tribal", "Employment"],
        "category": "st_male_unemployed",
    },
]


def evaluate_ner(test_cases):
    """Evaluate NER profile extraction accuracy."""
    print("=" * 60)
    print("EVALUATION 1: NER Profile Extraction Accuracy")
    print("=" * 60)

    total_fields = 0
    correct_fields = 0
    results = []

    for i, case in enumerate(test_cases, 1):
        profile = extract_profile(case["input"])
        expected = case["expected_profile"]

        case_correct = 0
        case_total = len(expected)

        for field, expected_val in expected.items():
            total_fields += 1
            actual_val = profile.get(field)

            if str(actual_val).upper() == str(expected_val).upper():
                correct_fields += 1
                case_correct += 1

        accuracy = (case_correct / case_total) * 100
        status = "✅" if accuracy == 100 else "⚠️" if accuracy >= 50 else "❌"
        results.append({"case": i, "category": case["category"], "accuracy": accuracy, "status": status})
        print(f"  {status} Test {i} [{case['category']}]: {accuracy:.0f}% ({case_correct}/{case_total} fields)")

    overall = (correct_fields / total_fields) * 100
    print(f"\n  Overall NER Accuracy: {overall:.1f}% ({correct_fields}/{total_fields} fields)")
    return overall, results


def evaluate_retrieval(test_cases, max_cases=10):
    """Evaluate RAG retrieval — does the system find relevant schemes?"""
    print("\n" + "=" * 60)
    print("EVALUATION 2: RAG Retrieval Relevance")
    print("=" * 60)

    hits = 0
    total = 0

    for i, case in enumerate(test_cases[:max_cases], 1):
        profile = extract_profile(case["input"])
        result = query_schemes(profile=profile, question=case["input"])

        # Check if any expected keyword appears in the answer
        answer_lower = result["answer"].lower()
        sources_lower = " ".join(result["sources"]).lower()
        combined = answer_lower + " " + sources_lower

        found_any = False
        for keyword in case["should_mention"]:
            if keyword.lower() in combined:
                found_any = True
                break

        total += 1
        if found_any:
            hits += 1
            print(f"  ✅ Test {i} [{case['category']}]: Found relevant scheme")
        else:
            print(f"  ❌ Test {i} [{case['category']}]: No expected scheme found")
            print(f"       Expected keywords: {case['should_mention']}")

        # Small delay to avoid rate limiting
        time.sleep(1)

    accuracy = (hits / total) * 100
    print(f"\n  Retrieval Relevance: {accuracy:.1f}% ({hits}/{total} cases)")
    return accuracy


if __name__ == "__main__":
    print("\n🔍 Running Evaluation Harness...\n")

    # Eval 1: NER accuracy (fast, no API calls)
    ner_accuracy, ner_results = evaluate_ner(TEST_CASES)

    # Eval 2: RAG retrieval (uses API, run on first 10 cases)
    retrieval_accuracy = evaluate_retrieval(TEST_CASES, max_cases=10)

    # Summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"  NER Extraction Accuracy:  {ner_accuracy:.1f}%")
    print(f"  RAG Retrieval Relevance:  {retrieval_accuracy:.1f}%")
    print(f"  Test Cases:               {len(TEST_CASES)} NER / 10 RAG")
    print("=" * 60)

    # Save results
    summary = {
        "ner_accuracy": round(ner_accuracy, 1),
        "retrieval_relevance": round(retrieval_accuracy, 1),
        "total_ner_cases": len(TEST_CASES),
        "total_rag_cases": 10,
        "ner_details": ner_results,
    }
    with open("evaluation_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\n📄 Results saved to evaluation_results.json")