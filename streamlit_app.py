"""
Streamlit Frontend - Government Scheme Eligibility Agent
Run: streamlit run streamlit_app.py
"""

import streamlit as st
import streamlit.components.v1 as components
from src.profile_extractor import extract_profile
from src.rag_chain import query_schemes
from dotenv import load_dotenv
import re

load_dotenv()

# --- Page Config ---
st.set_page_config(
    page_title="Scheme Eligibility Agent",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# --- Global CSS ---
st.markdown("""
<style>
    #MainMenu, footer, header {display: none;}
    .block-container {
        padding: 1rem 2rem 2rem 2rem;
        max-width: 1200px;
    }
    section[data-testid="stSidebar"] {display: none;}

    /* Hero */
    .hero-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 20px;
        padding: 3rem 2rem;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 40px rgba(102, 126, 234, 0.3);
    }
    .hero-box h1 {
        color: white;
        font-size: 2.6rem;
        font-weight: 800;
        margin: 0 0 0.5rem 0;
        letter-spacing: -0.5px;
    }
    .hero-box p {
        color: rgba(255,255,255,0.85);
        font-size: 1.15rem;
        margin: 0;
    }
    .hero-stats {
        display: flex;
        justify-content: center;
        gap: 2rem;
        margin-top: 1.5rem;
    }
    .hero-stat {
        text-align: center;
    }
    .hero-stat .num {
        font-size: 1.5rem;
        font-weight: 700;
        color: white;
    }
    .hero-stat .label {
        font-size: 0.8rem;
        color: rgba(255,255,255,0.7);
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    /* Profile grid */
    .profile-section {
        background: #1a1a2e;
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1.5rem 0;
    }
    .profile-section h3 {
        color: #a78bfa;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin: 0 0 1rem 0;
    }
    .profile-grid {
        display: grid;
        grid-template-columns: repeat(6, 1fr);
        gap: 12px;
    }
    @media (max-width: 768px) {
        .profile-grid { grid-template-columns: repeat(3, 1fr); }
    }
    .p-card {
        background: #16213e;
        border-radius: 12px;
        padding: 16px;
        text-align: center;
        border: 1px solid #2a2a4a;
    }
    .p-card .lbl {
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #8b8ba7;
        margin-bottom: 6px;
    }
    .p-card .val {
        font-size: 1.15rem;
        font-weight: 600;
        color: #e2e8f0;
    }
    .p-card .val.empty {
        color: #4a4a6a;
        font-weight: 400;
        font-size: 0.9rem;
    }

    /* Result cards */
    .results-section {
        margin: 1.5rem 0;
    }
    .results-section > h3 {
        color: #a78bfa;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin: 0 0 1rem 0;
    }
    .result-card {
        background: #1a1a2e;
        border-radius: 14px;
        padding: 20px 24px;
        margin-bottom: 12px;
        border-left: 4px solid #4a4a6a;
        transition: transform 0.2s;
    }
    .result-card:hover {
        transform: translateX(4px);
    }
    .result-card.eligible {
        border-left-color: #22c55e;
    }
    .result-card.possibly {
        border-left-color: #f59e0b;
    }
    .result-card.not-eligible {
        border-left-color: #ef4444;
    }
    .result-card .r-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 10px;
    }
    .result-card .r-name {
        font-size: 1.05rem;
        font-weight: 600;
        color: #e2e8f0;
    }
    .badge {
        display: inline-block;
        padding: 3px 12px;
        border-radius: 20px;
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .badge.eligible { background: #064e3b; color: #6ee7b7; }
    .badge.possibly { background: #78350f; color: #fcd34d; }
    .badge.not-eligible { background: #7f1d1d; color: #fca5a5; }
    .result-card .r-reason {
        font-size: 0.9rem;
        color: #94a3b8;
        line-height: 1.6;
    }

    /* Sources */
    .sources-section {
        background: #1a1a2e;
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1.5rem 0;
    }
    .sources-section h3 {
        color: #a78bfa;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin: 0 0 1rem 0;
    }
    .source-chip {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 8px 16px;
        background: #16213e;
        border-radius: 10px;
        margin: 4px;
        font-size: 0.82rem;
        color: #93c5fd;
        text-decoration: none;
        border: 1px solid #2a2a4a;
        transition: all 0.2s;
    }
    .source-chip:hover {
        background: #1e3a5f;
        color: #bfdbfe;
        text-decoration: none;
        border-color: #3b5998;
    }

    /* Footer badges */
    .footer-badges {
        text-align: center;
        margin-top: 2.5rem;
        padding-top: 1.5rem;
        border-top: 1px solid #2a2a4a;
    }
    .f-badge {
        display: inline-block;
        padding: 5px 14px;
        background: #1a1a2e;
        border-radius: 20px;
        font-size: 0.75rem;
        color: #8b8ba7;
        margin: 3px;
        border: 1px solid #2a2a4a;
    }

    /* Info box */
    .info-box {
        background: #1e1e3f;
        border: 1px solid #3b3b6d;
        border-radius: 12px;
        padding: 12px 16px;
        color: #a78bfa;
        font-size: 0.88rem;
        margin: 1rem 0;
    }

    /* Streamlit overrides */
    .stTextArea textarea {
        background: #1a1a2e;
        border: 1px solid #2a2a4a;
        border-radius: 12px;
        color: #e2e8f0;
        font-size: 1rem;
        padding: 16px;
    }
    .stTextArea textarea:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 2px rgba(102, 126, 234, 0.2);
    }
    .stTextArea textarea::placeholder {
        color: #4a4a6a;
    }
    .stSelectbox > div > div {
        background: #1a1a2e;
        border-color: #2a2a4a;
        color: #e2e8f0;
    }
    .stNumberInput > div > div > input {
        background: #1a1a2e;
        border-color: #2a2a4a;
        color: #e2e8f0;
    }
    label {
        color: #8b8ba7 !important;
        font-size: 0.8rem !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .stSpinner > div {
        border-color: #667eea;
    }
</style>
""", unsafe_allow_html=True)

# --- Hero ---
st.markdown("""
<div class="hero-box">
    <h1>🏛️ Scheme Eligibility Agent</h1>
    <p>Discover government schemes you qualify for — powered by AI</p>
    <div class="hero-stats">
        <div class="hero-stat"><div class="num">3,383</div><div class="label">Schemes</div></div>
        <div class="hero-stat"><div class="num">98%</div><div class="label">NER Accuracy</div></div>
        <div class="hero-stat"><div class="num">80%</div><div class="label">Retrieval Score</div></div>
    </div>
</div>
""", unsafe_allow_html=True)

# --- Session State ---
if "result" not in st.session_state:
    st.session_state.result = None
if "profile" not in st.session_state:
    st.session_state.profile = None


def clear_results():
    st.session_state.result = None
    st.session_state.profile = None


# --- Input Tabs ---
tab1, tab2 = st.tabs(["💬 Describe Yourself", "📋 Fill Form"])

with tab1:
    user_text = st.text_area(
        "describe",
        placeholder="Example: I'm a 22 year old OBC woman from Maharashtra, income 3 lakh, unemployed",
        height=120,
        label_visibility="collapsed",
    )
    col1, col2 = st.columns(2)
    with col1:
        nl_search = st.button("🔍 Find Schemes", type="primary", use_container_width=True, key="nl_search")
    with col2:
        st.button("🗑️ Clear Results", use_container_width=True, on_click=clear_results, key="nl_clear")

    if nl_search and user_text.strip():
        with st.spinner("Analyzing profile & searching schemes..."):
            profile = extract_profile(user_text)
            if profile["_fields_extracted"] < 2:
                st.markdown(f'<div class="info-box">⚠️ Only {profile["_fields_extracted"]} fields found. Add more details like age, state, income, or category.</div>', unsafe_allow_html=True)
            else:
                result = query_schemes(profile=profile, question=user_text)
                st.session_state.profile = profile
                st.session_state.result = result

with tab2:
    c1, c2, c3 = st.columns(3)
    with c1:
        age = st.number_input("Age", min_value=1, max_value=120, value=None, placeholder="e.g. 22")
        gender = st.selectbox("Gender", [None, "Male", "Female", "Transgender"])
    with c2:
        state = st.selectbox("State", [
            None, "Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar",
            "Chhattisgarh", "Delhi", "Goa", "Gujarat", "Haryana",
            "Himachal Pradesh", "Jharkhand", "Karnataka", "Kerala",
            "Madhya Pradesh", "Maharashtra", "Manipur", "Meghalaya",
            "Mizoram", "Nagaland", "Odisha", "Punjab", "Rajasthan",
            "Sikkim", "Tamil Nadu", "Telangana", "Tripura",
            "Uttar Pradesh", "Uttarakhand", "West Bengal",
        ])
        category = st.selectbox("Category", [None, "GENERAL", "OBC", "SC", "ST", "EWS"])
    with c3:
        income = st.number_input("Annual Income (₹)", min_value=0, value=None, placeholder="e.g. 300000")
        employment = st.selectbox("Employment", [
            None, "Unemployed", "Employed", "Student", "Self-Employed", "Farmer", "Retired"
        ])

    col1, col2 = st.columns(2)
    with col1:
        form_search = st.button("🔍 Find Schemes", type="primary", use_container_width=True, key="form_search")
    with col2:
        st.button("🗑️ Clear Results", use_container_width=True, on_click=clear_results, key="form_clear")

    if form_search:
        profile = {
            "age": age, "gender": gender, "category": category,
            "state": state, "income": income, "employment": employment,
        }
        filled = sum(1 for v in profile.values() if v is not None)
        if filled < 2:
            st.markdown('<div class="info-box">⚠️ Please fill at least 2 fields.</div>', unsafe_allow_html=True)
        else:
            profile["_fields_extracted"] = filled
            profile["_missing_fields"] = [k for k, v in profile.items() if v is None and not k.startswith("_")]
            with st.spinner("Searching schemes..."):
                result = query_schemes(profile=profile, question="What government schemes am I eligible for?")
                st.session_state.profile = profile
                st.session_state.result = result


# --- Display Results ---
if st.session_state.profile and st.session_state.result:
    profile = st.session_state.profile
    result = st.session_state.result

    # Profile Cards
    fields = ["age", "gender", "category", "state", "income", "employment"]
    cards_html = '<div class="profile-section"><h3>📌 Your Profile</h3><div class="profile-grid">'
    for field in fields:
        value = profile.get(field)
        if value is not None:
            display = f"₹{value:,}" if field == "income" else str(value)
            cards_html += f'<div class="p-card"><div class="lbl">{field}</div><div class="val">{display}</div></div>'
        else:
            cards_html += f'<div class="p-card"><div class="lbl">{field}</div><div class="val empty">—</div></div>'
    cards_html += '</div></div>'
    st.markdown(cards_html, unsafe_allow_html=True)

    # Missing fields
    missing = profile.get("_missing_fields", [])
    if missing:
        st.markdown(f'<div class="info-box">💡 Adding <b>{", ".join(missing)}</b> would improve accuracy.</div>', unsafe_allow_html=True)

    # Parse and display results as cards
    answer = result["answer"]

    # Try to parse individual scheme results
    scheme_blocks = re.split(r'\n(?=\d+\.|\*\*Scheme|\*\*\d)', answer)
    if len(scheme_blocks) > 1:
        st.markdown('<div class="results-section"><h3>✅ Eligibility Results</h3>', unsafe_allow_html=True)
        for block in scheme_blocks:
            block = block.strip()
            if not block:
                continue

            # Determine status
            if "ELIGIBLE" in block.upper() and "NOT ELIGIBLE" not in block.upper() and "POSSIBLY" not in block.upper():
                card_class = "eligible"
                badge_text = "Eligible"
            elif "POSSIBLY" in block.upper():
                card_class = "possibly"
                badge_text = "Possibly Eligible"
            elif "NOT ELIGIBLE" in block.upper():
                card_class = "not-eligible"
                badge_text = "Not Eligible"
            else:
                card_class = ""
                badge_text = ""

            # Clean up markdown formatting
            clean_block = block.replace("**", "").replace("##", "").strip()

            badge_html = f'<span class="badge {card_class}">{badge_text}</span>' if badge_text else ""
            st.markdown(
                f'<div class="result-card {card_class}">'
                f'{badge_html}'
                f'<div class="r-reason">{clean_block}</div>'
                f'</div>',
                unsafe_allow_html=True
            )
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        # Fallback: show as single block
        st.markdown('<div class="results-section"><h3>✅ Eligibility Results</h3>', unsafe_allow_html=True)
        st.markdown(f'<div class="result-card"><div class="r-reason">{answer}</div></div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Sources with links
    source_urls = result.get("source_urls", {})
    if source_urls:
        sources_html = '<div class="sources-section"><h3>📚 Sources — Click to verify on MyScheme.gov.in</h3><div>'
        for name, url in source_urls.items():
            if url and url.startswith("http"):
                sources_html += f'<a href="{url}" target="_blank" class="source-chip">🔗 {name}</a>'
            else:
                sources_html += f'<span class="source-chip">📄 {name}</span>'
        sources_html += '</div></div>'
        st.markdown(sources_html, unsafe_allow_html=True)

# --- Footer ---
techs = ["Python", "spaCy NER", "LangChain", "ChromaDB", "Groq LLM", "FastAPI", "SQLite", "Streamlit"]
footer_html = '<div class="footer-badges">'
for t in techs:
    footer_html += f'<span class="f-badge">{t}</span>'
footer_html += '</div>'
st.markdown(footer_html, unsafe_allow_html=True)