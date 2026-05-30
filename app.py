"""
app.py  -- Streamlit front-end for the DocTrust System
Run: streamlit run app.py
"""

import html
import streamlit as st

st.set_page_config(
    page_title="DocTrust — Academic Writing Assistance System",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }

:root {
    --bg:      #0d0f14;
    --surface: #151820;
    --border:  #252a35;
    --accent:  #4f8ef7;
    --accent2: #7c5cfc;
    --danger:  #f76f6f;
    --warn:    #f7b955;
    --ok:      #4fcf8e;
    --text:    #dce3f0;
    --muted:   #6b7590;
}

.stApp { background: var(--bg); color: var(--text); }
#MainMenu, footer, header { visibility: hidden; }

.card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.4rem 1.6rem;
    margin-top: 1rem;
}

.result-text {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.88rem;
    line-height: 1.7;
    color: var(--text);
    white-space: pre-wrap;
    word-break: break-word;
}

.badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 600;
    margin-left: 8px;
}
.badge-high { background:#3d1515; color: var(--danger); }
.badge-mod  { background:#3d2e10; color: var(--warn);   }
.badge-low  { background:#10302a; color: var(--ok);     }

.sim-bar-wrap { height: 8px; background: var(--border); border-radius: 4px; margin: 6px 0 12px; }
.sim-bar      { height: 8px; border-radius: 4px; }

div[data-baseweb="tab-list"] {
    gap: 4px !important;
    background: var(--surface) !important;
    border-radius: 10px !important;
    padding: 4px !important;
    border: 1px solid var(--border) !important;
}

div[data-baseweb="tab"] {
    border-radius: 8px !important;
    padding: 8px 20px !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    color: var(--muted) !important;
    border: none !important;
    background: transparent !important;
}

div[data-baseweb="tab"][aria-selected="true"] {
    background: var(--accent) !important;
    color: #fff !important;
}

div[data-baseweb="tab-highlight"] {
    display: none !important;
}

div[data-baseweb="tab-border"] {
    display: none !important;
}

textarea {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    border-radius: 8px !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
}

.stButton > button {
    background: linear-gradient(135deg, var(--accent), var(--accent2));
    color: #fff;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    padding: 0.5rem 1.6rem;
    transition: opacity 0.2s;
}
.stButton > button:hover { opacity: 0.85; }

.stTextArea label, .stSlider label { color: var(--muted) !important; font-size: 0.82rem; }

.section-title {
    font-size: 1.05rem;
    font-weight: 600;
    color: var(--accent);
    margin-bottom: 0.3rem;
    letter-spacing: 0.03em;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div style="text-align:center; padding: 2rem 0 1rem;">
  <h1 style="font-size:2.4rem; font-weight:700; letter-spacing:-0.5px; margin:0;">
    📄 DocTrust
  </h1>
  <p style="color:#6b7590; margin-top:0.3rem; font-size:0.95rem;">
    Summarization · Paraphrasing · Plagiarism Detection
  </p>
</div>
""", unsafe_allow_html=True)

@st.cache_resource(show_spinner=False)
def load_all_models():
    from model_loader import (
        get_summarizer,
        get_paraphrase_tokenizer,
        get_paraphrase_model,
        get_embedder,
    )
    get_summarizer()
    get_paraphrase_tokenizer()
    get_paraphrase_model()
    get_embedder()
    return True

with st.spinner("⚙️  Loading models (first run only — may take ~60 s) …"):
    load_all_models()

def _esc(text: str) -> str:
    return html.escape(str(text))

def preprocess(text: str, max_chars: int = 3000) -> str:
    import re
    text = re.sub(r'\s+', ' ', text).strip()
    return text[:max_chars]

def sim_bar_html(sim: float) -> str:
    pct   = int(sim * 100)
    color = "#f76f6f" if sim >= 0.88 else "#f7b955" if sim >= 0.72 else "#4fcf8e"
    return (
        f'<div class="sim-bar-wrap">'
        f'<div class="sim-bar" style="width:{pct}%; background:{color};"></div>'
        f'</div>'
        f'<span style="font-size:0.78rem; color:#6b7590;">{pct}% similarity</span>'
    )

def risk_badge(risk: str) -> str:
    cls = (
        "badge-high" if "High"     in risk else
        "badge-mod"  if "Moderate" in risk else
        "badge-low"
    )
    return f'<span class="badge {cls}">{_esc(risk)}</span>'

for _key in ("sum_result", "para_result", "para_sim", "plag_result",
             "sum_input_val", "para_input_val", "plag_input_val"):
    if _key not in st.session_state:
        st.session_state[_key] = "" if _key.endswith("_val") else None

tabs = ["📝  Summarize", "🔁  Paraphrase", "🔍  Plagiarism"]

if "active_tab" not in st.session_state:
    st.session_state.active_tab = tabs[0]

def change_tab():
    st.session_state.active_tab = st.session_state.tab_selector

st.radio(
    "",
    tabs,
    horizontal=True,
    index=tabs.index(st.session_state.active_tab),
    key="tab_selector",
    on_change=change_tab
)

if st.session_state.active_tab == "📝  Summarize":
    st.markdown('<div class="section-title">Text Summarization</div>', unsafe_allow_html=True)
    st.caption("Paste any article or paragraph — the model produces a concise abstractive summary.")

    sum_text = st.text_area(
        "Input text",
        height=220,
        placeholder="Paste your text here (min 20 words recommended) …",
        key="sum_input",
        value=st.session_state.sum_input_val,
    )

    if sum_text != st.session_state.sum_input_val:
        st.session_state.sum_input_val = sum_text

    if st.button("Summarize", key="btn_sum"):
        cleaned = preprocess(sum_text)
        if len(cleaned.split()) < 20:
            st.warning("⚠️  Please enter at least 20 words for meaningful summarization.")
        else:
            with st.spinner("Summarizing …"):
                from summarizer import summarize
                st.session_state.sum_result = summarize(cleaned)

    if st.session_state.sum_result:
        st.markdown(f"""
        <div class="card">
          <div class="section-title" style="margin-bottom:0.6rem;">📄 Summary</div>
          <div class="result-text">{_esc(st.session_state.sum_result)}</div>
        </div>
        """, unsafe_allow_html=True)

if st.session_state.active_tab == "🔁  Paraphrase":
    st.markdown('<div class="section-title">Text Paraphrasing</div>', unsafe_allow_html=True)
    st.caption("Rewrites your text while preserving meaning. Similarity score shows semantic closeness.")

    para_text = st.text_area(
        "Input text",
        height=180,
        placeholder="Enter a sentence or short paragraph to paraphrase …",
        key="para_input",
        value=st.session_state.para_input_val,
    )

    if para_text != st.session_state.para_input_val:
        st.session_state.para_input_val = para_text

    if st.button("Paraphrase", key="btn_para"):
        cleaned = preprocess(para_text, max_chars=512)
        if len(cleaned.split()) < 5:
            st.warning("⚠️  Enter at least 5 words.")
        else:
            with st.spinner("Paraphrasing …"):
                from paraphraser import paraphrase
                result, sim = paraphrase(cleaned)
                st.session_state.para_result = result
                st.session_state.para_sim    = sim

    if st.session_state.para_result:
        st.markdown(f"""
        <div class="card">
          <div class="section-title" style="margin-bottom:0.6rem;">🔁 Paraphrase</div>
          <div class="result-text">{_esc(st.session_state.para_result)}</div>
          <div style="margin-top:1rem;">
            {sim_bar_html(st.session_state.para_sim)}
          </div>
        </div>
        """, unsafe_allow_html=True)

if st.session_state.active_tab == "🔍  Plagiarism":
    st.markdown('<div class="section-title">Plagiarism Detection</div>', unsafe_allow_html=True)
    st.caption("Compares your text against a reference corpus using semantic similarity (FAISS + MiniLM).")

    plag_text = st.text_area(
        "Input text",
        height=180,
        placeholder="Enter text to check for plagiarism …",
        key="plag_input",
        value=st.session_state.plag_input_val,
    )

    if plag_text != st.session_state.plag_input_val:
        st.session_state.plag_input_val = plag_text

    top_k = st.slider("Number of matches to show", 1, 5, 3, key="plag_k")

    if st.button("Check", key="btn_plag"):
        cleaned = preprocess(plag_text)
        if len(cleaned.split()) < 5:
            st.warning("⚠️  Enter at least 5 words.")
        else:
            with st.spinner("Checking plagiarism …"):
                from plagiarism import check_plagiarism
                st.session_state.plag_result = check_plagiarism(cleaned, top_k=top_k)

    if st.session_state.plag_result:
        result       = st.session_state.plag_result
        matches      = result["matches"]
        overall_risk = result["overall_risk"]
        max_sim      = result["max_similarity"]

        banner_color = (
            "#3d1515" if "High"     in overall_risk else
            "#3d2e10" if "Moderate" in overall_risk else
            "#10302a"
        )
        text_color = (
            "#f76f6f" if "High"     in overall_risk else
            "#f7b955" if "Moderate" in overall_risk else
            "#4fcf8e"
        )

        st.markdown(f"""
        <div style="background:{banner_color}; border:1px solid {text_color};
                    border-radius:10px; padding:0.9rem 1.2rem; margin:1rem 0;
                    display:flex; justify-content:space-between; align-items:center;">
          <span style="font-weight:700; font-size:1rem; color:{text_color};">
            Overall Document Risk: {_esc(overall_risk)}
          </span>
          <span style="color:{text_color}; font-size:0.85rem;">
            Highest similarity: {int(max_sim * 100)}%
          </span>
        </div>
        """, unsafe_allow_html=True)

        for i, m in enumerate(matches, 1):
            badge    = risk_badge(m["risk_level"])
            sim_html = sim_bar_html(m["similarity_score"])

            query_row = ""
            if "query_sentence" in m:
                query_row = (
                    f'<div style="font-size:0.78rem; color:#6b7590; margin-bottom:0.5rem;">'
                    f'<span style="color:#4f8ef7; font-weight:600;">Your sentence: </span>'
                    f'{_esc(m["query_sentence"])}'
                    f'</div>'
                )

            st.markdown(f"""
            <div class="card">
              <div style="display:flex; align-items:center; margin-bottom:0.5rem;">
                <span style="font-weight:600; color:#dce3f0;">Match #{i}</span>
                {badge}
              </div>
              {query_row}
              <div class="result-text">{_esc(m["matched_text"])}</div>
              <div style="margin-top:0.8rem;">{sim_html}</div>
            </div>
            """, unsafe_allow_html=True)