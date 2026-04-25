import streamlit as st
import numpy as np
import requests
import json
import plotly.graph_objects as go
from knowledge_base import DISEASES, EMERGENCY_RULES
from inference_engine import BayesianTriage
from kb_loader import KnowledgeBaseLoader

st.set_page_config(
    page_title="SmartTriage AI",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── CSS ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=Instrument+Serif:ital@0;1&display=swap');

:root {
    --bg: #0a0f1e;
    --surface: #111827;
    --surface2: #1a2236;
    --accent: #00d4aa;
    --accent2: #7c3aed;
    --danger: #ef4444;
    --warning: #f59e0b;
    --safe: #10b981;
    --text: #e2e8f0;
    --muted: #64748b;
    --border: #1e2d45;
}

html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif;
    background-color: var(--bg);
    color: var(--text);
}
.stApp { background: var(--bg); }
.stApp::before {
    content: '';
    position: fixed; top:0;left:0;right:0;bottom:0;
    background-image:
        linear-gradient(rgba(0,212,170,0.025) 1px, transparent 1px),
        linear-gradient(90deg, rgba(0,212,170,0.025) 1px, transparent 1px);
    background-size: 40px 40px;
    pointer-events: none; z-index: 0;
}

.main-header { text-align:center; padding:1.5rem 0 0.5rem 0; }
.main-header h1 {
    font-family: 'Instrument Serif', serif; font-size:3.2rem; font-weight:400;
    font-style:italic;
    background: linear-gradient(135deg, #00d4aa 0%, #7c3aed 100%);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text;
    margin:0; letter-spacing:-1px;
}
.main-header p { color:var(--muted); font-size:0.85rem; letter-spacing:2px; text-transform:uppercase; }

.card {
    background:var(--surface); border:1px solid var(--border);
    border-radius:16px; padding:1.4rem; margin-bottom:1rem; position:relative; overflow:hidden;
}
.card::before {
    content:''; position:absolute; top:0;left:0;right:0;height:2px;
    background:linear-gradient(90deg,var(--accent),var(--accent2));
}
.card-plain {
    background:var(--surface2); border:1px solid var(--border);
    border-radius:12px; padding:1rem; margin-bottom:0.75rem;
}

.step-indicator { display:flex;align-items:center;gap:8px;color:var(--muted);font-size:0.75rem;letter-spacing:1.5px;text-transform:uppercase;margin-bottom:0.75rem; }
.step-num { width:22px;height:22px;border-radius:50%;background:linear-gradient(135deg,var(--accent),var(--accent2));color:white;display:flex;align-items:center;justify-content:center;font-size:0.7rem;font-weight:700; }

.badge { display:inline-block;padding:2px 10px;border-radius:999px;font-size:0.7rem;font-weight:600;letter-spacing:1px;text-transform:uppercase;border:1px solid;margin:2px; }
.badge-emergency { background:rgba(239,68,68,0.15);border-color:#ef4444;color:#ef4444; }
.badge-moderate  { background:rgba(245,158,11,0.15);border-color:#f59e0b;color:#f59e0b; }
.badge-low       { background:rgba(16,185,129,0.15);border-color:#10b981;color:#10b981; }

.risk-banner { border-radius:16px;padding:1.2rem 2rem;text-align:center;margin-bottom:1.2rem; }
.risk-Emergency { background:linear-gradient(135deg,rgba(239,68,68,0.2),rgba(239,68,68,0.05));border:1px solid rgba(239,68,68,0.4); }
.risk-Moderate  { background:linear-gradient(135deg,rgba(245,158,11,0.2),rgba(245,158,11,0.05));border:1px solid rgba(245,158,11,0.4); }
.risk-Low       { background:linear-gradient(135deg,rgba(16,185,129,0.2),rgba(16,185,129,0.05));border:1px solid rgba(16,185,129,0.4); }

.disease-row { display:flex;align-items:center;gap:10px;padding:10px 0;border-bottom:1px solid var(--border); }
.disease-name { font-weight:600;font-size:0.9rem;min-width:190px; }
.prob-bar-bg { flex:1;height:7px;background:var(--surface2);border-radius:4px;overflow:hidden; }
.prob-bar { height:100%;border-radius:4px;background:linear-gradient(90deg,var(--accent),var(--accent2)); }
.prob-pct { font-weight:700;font-size:0.95rem;min-width:48px;text-align:right;color:var(--accent); }

.explain-item { background:var(--surface2);border-radius:10px;padding:10px 14px;margin:5px 0;font-size:0.85rem;border-left:3px solid var(--accent); }
.explain-ctx  { background:var(--surface2);border-radius:10px;padding:10px 14px;margin:5px 0;font-size:0.85rem;border-left:3px solid var(--accent2); }

.symptom-tag { display:inline-block;background:rgba(0,212,170,0.1);border:1px solid rgba(0,212,170,0.3);color:var(--accent);border-radius:999px;padding:3px 10px;font-size:0.75rem;margin:2px;font-weight:500; }

.stButton > button {
    background:linear-gradient(135deg,#00d4aa,#7c3aed) !important;
    color:white !important; border:none !important; border-radius:12px !important;
    padding:0.65rem 1.5rem !important; font-family:'Space Grotesk',sans-serif !important;
    font-weight:600 !important; font-size:0.95rem !important;
    box-shadow:0 4px 20px rgba(0,212,170,0.2) !important; transition:all 0.2s !important;
}
.stButton > button:hover { transform:translateY(-2px) !important; box-shadow:0 8px 28px rgba(0,212,170,0.35) !important; }

.stTabs [data-baseweb="tab-list"] { background:var(--surface) !important;border-radius:10px !important;gap:3px !important; }
.stTabs [data-baseweb="tab"] { color:var(--muted) !important;font-family:'Space Grotesk',sans-serif !important; }
.stTabs [aria-selected="true"] { color:var(--accent) !important;background:var(--surface2) !important;border-radius:8px !important; }

.stSelectbox label,.stMultiSelect label,.stSlider label,.stRadio label,.stTextArea label {
    color:var(--muted) !important;font-size:0.78rem !important;text-transform:uppercase !important;letter-spacing:1px !important;
}
div[data-testid="stMetricValue"] { color:var(--accent) !important;font-family:'Instrument Serif',serif !important;font-style:italic !important; }

.disclaimer { background:rgba(100,116,139,0.1);border:1px solid rgba(100,116,139,0.2);border-radius:10px;padding:10px 14px;font-size:0.78rem;color:var(--muted);text-align:center;margin-top:0.75rem; }

.api-pill { display:inline-flex;align-items:center;gap:6px;background:rgba(0,212,170,0.08);border:1px solid rgba(0,212,170,0.25);border-radius:999px;padding:4px 14px;font-size:0.78rem;color:var(--accent);font-weight:500;margin-bottom:0.5rem; }
.api-dot { width:7px;height:7px;border-radius:50%;background:var(--accent);animation:pulse 2s infinite; }
@keyframes pulse { 0%,100%{opacity:1}50%{opacity:0.3} }

.sidebar-section { font-size:0.72rem;letter-spacing:1.5px;text-transform:uppercase;color:var(--muted);margin:1rem 0 0.4rem 0;font-weight:600; }

/* Sidebar */
[data-testid="stSidebar"] { background:var(--surface) !important;border-right:1px solid var(--border) !important; }
[data-testid="stSidebar"] .stMarkdown { color:var(--text); }
</style>
""", unsafe_allow_html=True)

# ─── Session state ─────────────────────────────────────────────────────────────
for key, default in [("results", None), ("submitted", False),
                      ("symptoms_used", []), ("use_api", False),
                      ("local_kb", {k: v.copy() for k, v in DISEASES.items()})]:
    if key not in st.session_state:
        st.session_state[key] = default

# ─── All symptoms (from local KB, updated if extended) ─────────────────────
def get_all_symptoms():
    return sorted(set(s for d in st.session_state.local_kb.values() for s in d["symptoms"].keys()))

loader = KnowledgeBaseLoader(st.session_state.local_kb)

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="font-family:'Instrument Serif',serif;font-size:1.6rem;font-style:italic;
                background:linear-gradient(135deg,#00d4aa,#7c3aed);-webkit-background-clip:text;
                -webkit-text-fill-color:transparent;background-clip:text;margin-bottom:0.5rem;">
        SmartTriage AI
    </div>
    """, unsafe_allow_html=True)

    # API mode toggle
    st.markdown('<div class="sidebar-section">Backend Mode</div>', unsafe_allow_html=True)
    use_api = st.toggle("Use Flask API Backend", value=st.session_state.use_api,
                        help="When ON, analysis is sent to Flask API at localhost:5000. When OFF, runs locally in Streamlit.")
    st.session_state.use_api = use_api

    if use_api:
        api_url = st.text_input("API Base URL", value="http://localhost:5000")
        try:
            r = requests.get(f"{api_url}/api/health", timeout=2)
            d = r.json()
            st.markdown(f"""
            <div class="api-pill"><div class="api-dot"></div> API Online · {d['diseases_loaded']} diseases</div>
            """, unsafe_allow_html=True)
        except Exception:
            st.markdown("""
            <div style="background:rgba(239,68,68,0.1);border:1px solid rgba(239,68,68,0.3);
                        border-radius:8px;padding:8px 12px;font-size:0.8rem;color:#ef4444;">
                ⚠️ API offline — run <code>python api.py</code>
            </div>
            """, unsafe_allow_html=True)
            api_url = "http://localhost:5000"
    else:
        api_url = "http://localhost:5000"

    # Knowledge Base Manager
    st.markdown('<div class="sidebar-section">Knowledge Base</div>', unsafe_allow_html=True)
    kb_count = len(st.session_state.local_kb)
    st.markdown(f"""
    <div style="font-size:0.85rem;color:#94a3b8;margin-bottom:0.5rem;">
        {kb_count} diseases loaded
    </div>
    """, unsafe_allow_html=True)

    with st.expander("📂 Expand KB from File"):
        st.markdown("""<div style="font-size:0.8rem;color:#64748b;margin-bottom:8px;">
        Upload a <b>.json</b> or <b>.csv</b> file to add new diseases to the knowledge base at runtime.
        <br><br><b>CSV format:</b><br>
        <code>disease, urgency, prior, description, symptom1, prob1, ...</code>
        </div>""", unsafe_allow_html=True)

        uploaded = st.file_uploader("Choose file", type=["csv", "json"], label_visibility="collapsed")
        if uploaded:
            content = uploaded.read().decode("utf-8")
            if uploaded.name.endswith(".json"):
                try:
                    data = json.loads(content)
                    added, errors = loader.load_from_dict(data)
                except json.JSONDecodeError as e:
                    added, errors = [], [str(e)]
            else:
                added, errors = loader.load_from_csv_string(content)

            if added:
                st.success(f"✅ Added: {', '.join(added)}")
            if errors:
                for e in errors:
                    st.error(f"❌ {e}")
            if added:
                st.rerun()

    with st.expander("➕ Add Disease Manually"):
        new_name = st.text_input("Disease Name")
        new_urgency = st.selectbox("Urgency", ["Low", "Moderate", "Emergency"])
        new_prior = st.slider("Prior Probability", 0.01, 0.30, 0.05, 0.01)
        new_desc = st.text_input("Description (optional)")
        new_syms_raw = st.text_area("Symptoms (one per line: symptom_name, probability)",
                                     placeholder="fever, 0.8\ncough, 0.7\nfatigue, 0.6")
        if st.button("Add Disease", key="add_disease_btn"):
            if new_name and new_syms_raw:
                syms = {}
                for line in new_syms_raw.strip().split("\n"):
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) == 2:
                        try:
                            syms[parts[0].lower().replace(" ", "_")] = float(parts[1])
                        except ValueError:
                            pass
                if syms:
                    st.session_state.local_kb[new_name] = {
                        "prior": new_prior, "urgency": new_urgency,
                        "description": new_desc, "symptoms": syms
                    }
                    st.success(f"✅ '{new_name}' added!")
                    st.rerun()
                else:
                    st.error("Could not parse any symptoms.")
            else:
                st.warning("Please fill in name and symptoms.")

    with st.expander("📋 Disease List"):
        for dname, ddata in st.session_state.local_kb.items():
            urg_colors = {"Emergency": "#ef4444", "Moderate": "#f59e0b", "Low": "#10b981"}
            c = urg_colors.get(ddata["urgency"], "#64748b")
            st.markdown(f"""
            <div style="display:flex;justify-content:space-between;align-items:center;
                        padding:4px 0;border-bottom:1px solid #1e2d45;font-size:0.78rem;">
                <span>{dname}</span>
                <span style="color:{c};font-weight:600;font-size:0.7rem;">{ddata['urgency']}</span>
            </div>""", unsafe_allow_html=True)

    if st.button("🔄 Reset KB to Defaults"):
        st.session_state.local_kb = {k: v.copy() for k, v in DISEASES.items()}
        loader.kb = st.session_state.local_kb
        st.rerun()

# ─── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>SmartTriage AI</h1>
    <p>Bayesian Medical Risk Assessment System</p>
</div>
""", unsafe_allow_html=True)
st.markdown("---")

# ─── Main layout ──────────────────────────────────────────────────────────────
left_col, right_col = st.columns([1, 1.4], gap="large")

with left_col:
    st.markdown('<div class="card"><div class="step-indicator"><div class="step-num">1</div> Patient Context</div>', unsafe_allow_html=True)
    age = st.slider("Age", 1, 100, 30, format="%d years")
    c1, c2 = st.columns(2)
    with c1:
        sex = st.selectbox("Biological Sex", ["Male", "Female", "Other"])
    with c2:
        duration = st.selectbox("Symptom Duration", ["< 24 hours", "1–3 days", "3–7 days", "> 1 week"])
    chronic = st.multiselect("Chronic Conditions", ["Diabetes", "Hypertension", "Heart Disease",
                              "Asthma", "Immunocompromised", "Pregnancy", "Obesity", "None"], default=["None"])
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="card"><div class="step-indicator"><div class="step-num">2</div> Symptom Entry</div>', unsafe_allow_html=True)
    input_mode = st.radio("Input Method", ["Select from list", "Type freely"], horizontal=True)
    if input_mode == "Select from list":
        selected_symptoms = st.multiselect("Choose all that apply", get_all_symptoms(),
                                            placeholder="Start typing to search...")
        free_text = ""
    else:
        free_text = st.text_area("Describe your symptoms",
                                  placeholder="e.g., high fever, sore throat, very tired...", height=90)
        selected_symptoms = []
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="card"><div class="step-indicator"><div class="step-num">3</div> Severity Assessment</div>', unsafe_allow_html=True)
    pain_level = st.slider("Pain / Discomfort Level", 0, 10, 3, help="0 = None, 10 = Worst imaginable")
    additional_flags = st.multiselect("Additional Flags",
        ["Difficulty breathing", "Chest tightness", "Confusion / disorientation",
         "Sudden severe headache", "Loss of consciousness", "Uncontrolled bleeding",
         "High fever > 39°C", "Rapid heart rate", "None of the above"],
        default=["None of the above"])
    st.markdown("</div>", unsafe_allow_html=True)

    analyze_btn = st.button("🔬  Analyze Symptoms", use_container_width=True)

# ─── Text parser ──────────────────────────────────────────────────────────────
def parse_free_text(text):
    matched = []
    text_lower = text.lower()
    for sym in get_all_symptoms():
        readable = sym.replace("_", " ")
        if readable in text_lower or sym in text_lower:
            matched.append(sym)
        else:
            words = readable.split()
            if any(w in text_lower for w in words if len(w) > 4):
                if sym not in matched:
                    matched.append(sym)
    return matched

# ─── Analysis ─────────────────────────────────────────────────────────────────
if analyze_btn:
    if free_text:
        selected_symptoms = parse_free_text(free_text)
    if not selected_symptoms:
        st.warning("⚠️ Please enter at least one symptom.")
    else:
        patient_ctx = {"age": age, "sex": sex, "duration": duration,
                       "chronic": chronic, "pain_level": pain_level, "flags": additional_flags}

        if st.session_state.use_api:
            try:
                resp = requests.post(f"{api_url}/api/analyze",
                    json={"symptoms": selected_symptoms, "patient_context": patient_ctx},
                    timeout=10)
                result = resp.json()
                # Normalize API response to match local format
                for d in result.get("diagnoses", []):
                    d["probability"] = d.get("probability", 0)
                st.session_state.results = result
            except Exception as e:
                st.error(f"API error: {e}. Falling back to local engine.")
                engine = BayesianTriage(st.session_state.local_kb, EMERGENCY_RULES)
                st.session_state.results = engine.analyze(selected_symptoms, patient_ctx)
        else:
            engine = BayesianTriage(st.session_state.local_kb, EMERGENCY_RULES)
            st.session_state.results = engine.analyze(selected_symptoms, patient_ctx)

        st.session_state.submitted = True
        st.session_state.symptoms_used = selected_symptoms

# ─── Results ──────────────────────────────────────────────────────────────────
with right_col:
    if not st.session_state.submitted or st.session_state.results is None:
        mode_label = "Flask API" if st.session_state.use_api else "Local Engine"
        st.markdown(f"""
        <div style="display:flex;flex-direction:column;align-items:center;justify-content:center;
                    height:480px;color:#475569;text-align:center;gap:14px;">
            <div style="font-size:3.5rem;">🩺</div>
            <div style="font-family:'Instrument Serif',serif;font-size:1.7rem;font-style:italic;color:#64748b;">
                Awaiting Assessment
            </div>
            <div style="font-size:0.85rem;max-width:260px;line-height:1.6;">
                Fill in patient context and symptoms, then click <strong>Analyze Symptoms</strong>
            </div>
            <div style="font-size:0.75rem;color:#334155;background:rgba(0,212,170,0.06);
                        border:1px solid rgba(0,212,170,0.15);border-radius:999px;padding:4px 14px;">
                Mode: {mode_label}
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        res = st.session_state.results
        symptoms_used = st.session_state.symptoms_used
        risk = res["risk_level"]
        risk_icons = {"Emergency": "🚨", "Moderate": "⚠️", "Low": "✅"}
        risk_colors = {"Emergency": "#ef4444", "Moderate": "#f59e0b", "Low": "#10b981"}
        risk_actions = {
            "Emergency": "Seek immediate emergency care — call emergency services or go to ER now",
            "Moderate": "Schedule a doctor's appointment within 24–48 hours",
            "Low": "Rest, hydrate, and monitor symptoms. Self-care recommended."
        }

        st.markdown(f"""
        <div class="risk-banner risk-{risk}">
            <div style="font-size:2.2rem;">{risk_icons[risk]}</div>
            <div style="font-size:1.6rem;font-weight:700;color:{risk_colors[risk]};
                        font-family:'Instrument Serif',serif;font-style:italic;margin:4px 0;">
                {risk} Risk
            </div>
            <div style="color:#94a3b8;font-size:0.85rem;max-width:360px;margin:auto;">
                {risk_actions[risk]}
            </div>
        </div>
        """, unsafe_allow_html=True)

        if res.get("emergency_override"):
            st.markdown(f"""
            <div style="background:rgba(239,68,68,0.1);border:1px solid rgba(239,68,68,0.4);
                        border-radius:10px;padding:10px 14px;margin-bottom:0.8rem;font-size:0.85rem;">
                <strong style="color:#ef4444;">⚡ Emergency Rule Triggered:</strong>
                <span style="color:#fca5a5;"> {res['emergency_override']}</span>
            </div>
            """, unsafe_allow_html=True)

        sym_tags = "".join(f'<span class="symptom-tag">{s.replace("_"," ")}</span>' for s in symptoms_used)
        st.markdown(f"""
        <div style="margin-bottom:0.8rem;">
            <div style="color:#64748b;font-size:0.72rem;letter-spacing:1px;text-transform:uppercase;margin-bottom:6px;">
                Symptoms Analyzed
            </div>{sym_tags}
        </div>""", unsafe_allow_html=True)

        tab1, tab2, tab3 = st.tabs(["📊 Diagnosis", "🧠 Reasoning", "📈 Chart"])

        with tab1:
            st.markdown("**Probable Conditions**")
            for diag in res["diagnoses"][:6]:
                pct = diag["probability"] * 100
                urg = diag["urgency"]
                badge_class = f"badge-{urg.lower()}"
                st.markdown(f"""
                <div class="disease-row">
                    <div class="disease-name">{diag['disease']}</div>
                    <div class="prob-bar-bg"><div class="prob-bar" style="width:{int(pct)}%"></div></div>
                    <div class="prob-pct">{pct:.1f}%</div>
                    <span class="badge {badge_class}">{urg}</span>
                </div>""", unsafe_allow_html=True)

            conf = res.get("confidence", 0)
            mode_used = "Flask API" if st.session_state.use_api else "Local Engine"
            st.markdown(f"""
            <div style="margin-top:0.8rem;display:flex;gap:10px;">
                <div style="flex:1;padding:10px 14px;background:var(--surface2);border-radius:10px;">
                    <div style="color:#64748b;font-size:0.72rem;text-transform:uppercase;letter-spacing:1px;">Confidence</div>
                    <div style="font-size:1.3rem;font-weight:700;color:#00d4aa;font-family:'Instrument Serif',serif;font-style:italic;">{conf:.0%}</div>
                </div>
                <div style="flex:1;padding:10px 14px;background:var(--surface2);border-radius:10px;">
                    <div style="color:#64748b;font-size:0.72rem;text-transform:uppercase;letter-spacing:1px;">Engine</div>
                    <div style="font-size:0.85rem;font-weight:600;color:#00d4aa;margin-top:4px;">{mode_used}</div>
                </div>
            </div>""", unsafe_allow_html=True)

        with tab2:
            if res.get("explanations"):
                st.markdown("**Why these results?**")
                for exp in res["explanations"]:
                    st.markdown(f'<div class="explain-item"><span style="color:#00d4aa;">→</span> {exp}</div>', unsafe_allow_html=True)
            if res.get("context_notes"):
                st.markdown("**Context Adjustments**")
                for note in res["context_notes"]:
                    st.markdown(f'<div class="explain-ctx"><span style="color:#7c3aed;">⟳</span> {note}</div>', unsafe_allow_html=True)

        with tab3:
            top5 = res["diagnoses"][:5]
            if top5:
                labels = [d["disease"] for d in top5]
                values = [round(d["probability"] * 100, 1) for d in top5]
                colors = ["#00d4aa", "#7c3aed", "#f59e0b", "#ef4444", "#10b981"]
                fig = go.Figure(go.Bar(
                    x=values, y=labels, orientation='h',
                    marker=dict(color=colors[:len(labels)], line=dict(width=0)),
                    text=[f"{v}%" for v in values], textposition='outside',
                    textfont=dict(color='#e2e8f0', size=11, family='Space Grotesk')
                ))
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#e2e8f0', family='Space Grotesk'),
                    xaxis=dict(gridcolor='rgba(30,45,69,0.8)', title="Probability (%)", color='#64748b'),
                    yaxis=dict(gridcolor='rgba(0,0,0,0)', autorange="reversed", color='#e2e8f0'),
                    margin=dict(l=0, r=60, t=10, b=20), height=260
                )
                st.plotly_chart(fig, use_container_width=True)

        st.markdown("""
        <div class="disclaimer">
            ⚕️ <strong>Medical Disclaimer:</strong> SmartTriage AI is an educational tool only.
            It does <em>not</em> replace professional medical advice, diagnosis, or treatment.
        </div>""", unsafe_allow_html=True)

        if st.button("🔄 New Assessment", use_container_width=True):
            st.session_state.results = None
            st.session_state.submitted = False
            st.rerun()
