# SmartTriage AI — Complete Project
**Group Members:** Amna Rais · Afaf Shahid · Hamna Khalid
**Course:** AI Project 2025 · Namal University

---

## Project Structure

```
smarttriage/
│
├── backend/                      ← Python / Flask
│   ├── knowledge_base.py         ← Disease data + emergency rules (SOURCE OF TRUTH)
│   ├── inference_engine.py       ← BayesianTriage class (core algorithm)
│   ├── kb_loader.py              ← CSV/JSON runtime KB expansion
│   ├── api.py                    ← Flask REST API (9 endpoints)
│   ├── app.py                    ← Streamlit demo UI
│   ├── requirements.txt
│   ├── sample_expansion.csv      ← Test dataset (4 diseases)
│   └── sample_expansion.json     ← Test dataset (3 diseases)
│
└── frontend/                     ← React / Vite
    ├── index.html
    ├── vite.config.js            ← Dev server + proxy to Flask
    └── src/
        ├── main.jsx              ← React entry point
        ├── index.css             ← Global tokens + animations
        ├── knowledgeBase.js      ← MIRRORS knowledge_base.py (frontend copy)
        ├── bayesianEngine.js     ← MIRRORS inference_engine.py (runs in browser)
        ├── api.js                ← All Flask API calls (with error handling)
        ├── App.jsx               ← Main UI component
        └── App.module.css        ← All component styles (CSS Modules)
```

---

## How the Files Connect

```
knowledge_base.py ──────────────────────────────────────────────┐
    ↓ imported by                                               │
inference_engine.py  (BayesianTriage class)                     │ MIRRORED TO
    ↓ imported by                                               │
api.py  (Flask)  ──→  POST /api/analyze  ──→  api.js (React)   │
    ↑ also imports                                              │
kb_loader.py  ←── POST /api/upload/csv  /api/upload/json        │
                                                                │
knowledgeBase.js ◄──────────────────────────────────────────────┘
    ↓ imported by
bayesianEngine.js  ←── offline fallback in App.jsx
    ↓ imported by
App.jsx  ──→  tries api.js first, falls back to bayesianEngine.js
```

### Data Flow on every "Calculate" click:

1. User selects symptoms in React UI
2. `App.jsx` calls `api.runAnalysis(symptoms, patientCtx)` → `POST /api/analyze`
3. Flask `api.py` receives request → calls `BayesianTriage.analyze()` from `inference_engine.py`
4. `inference_engine.py` reads disease data from `knowledge_base.py`
5. Bayesian posterior computed → JSON response sent back
6. React renders `RiskBanner` + `DiagnosisTable` + `XAIPanel`

**If Flask is offline:** Step 2 fails → React falls back to `bayesianEngine.js.analyze()` which runs the same algorithm in the browser using `knowledgeBase.js` data.

---

## File-by-File Reference

### `knowledge_base.py` — The Source of Truth
Contains the complete disease-symptom probability matrix:
- **15 diseases** with `prior` (base probability), `urgency`, `description`, and `symptoms` dict
- Each symptom key maps to P(symptom | disease) — a value from 0 to 1
- **8 emergency override rules** that bypass Bayesian ranking for safety-critical scenarios

**How priors work:**
```python
"Influenza (Flu)": {
    "prior": 0.15,       # 15% base chance in any patient
    "symptoms": {
        "fever": 0.90,   # 90% of flu patients have fever
        "cough": 0.85,   # 85% of flu patients have cough
    }
}
```

### `knowledgeBase.js` — Frontend Mirror of knowledge_base.py
Exact copy of the Python data in JavaScript format. Used by:
1. **Symptom picker UI** — ALL_SYMPTOMS list (80 unique symptoms)
2. **EMERGENCY_SYMPTOMS** — highlighted in red in the picker
3. **bayesianEngine.js** — when running offline

If you add a disease to `knowledge_base.py`, add it here too.

---

### `inference_engine.py` — Core Algorithm
```
P(D|S) = P(S|D) × P(D) / P(S)
```
Three stages:
1. **`_adjust_prior()`** — multiplies base priors by age/sex/chronic/duration factors
2. **`_compute_posterior()`** — joint likelihood × adjusted prior, normalized
3. **`_check_emergency_rules()`** — rule-based safety override
4. **`_build_explanations()`** — human-readable reasoning per result

**Age adjustments example:**
- Patient age > 60 → Pneumonia prior ×2.0, Hypertensive Crisis ×2.5
- Female + age 15–40 → UTI prior ×2.0
- Asthma in chronic conditions → Asthma Attack prior ×3.0

### `bayesianEngine.js` — Browser Port of inference_engine.py
JavaScript implementation of the exact same algorithm. Called when:
- Flask API is offline
- API call returns an error

Results from both sources have the same shape so the React UI renders identically.

---

### `kb_loader.py` — Runtime KB Expansion
Adds diseases to the running knowledge base at runtime without restarting:

**CSV format** (one disease per row):
```
disease,urgency,prior,description,symptom1,prob1,symptom2,prob2,...
Chickenpox,Moderate,0.04,Viral infection causing itchy rash,rash,0.95,fever,0.70
```

**JSON format:**
```json
{
  "Disease Name": {
    "prior": 0.05,
    "urgency": "Moderate",
    "description": "...",
    "symptoms": { "fever": 0.8, "cough": 0.7 }
  }
}
```

Validates: prior in (0,1], urgency in {Emergency/Moderate/Low}, all probabilities in [0,1].

### `api.py` — Flask REST API

| Method | Endpoint | What it does |
|--------|----------|-------------|
| GET | `/api/health` | Status check + disease count |
| POST | `/api/analyze` | Full Bayesian triage (main endpoint) |
| GET | `/api/symptoms` | All symptom keys |
| GET | `/api/diseases` | All diseases with metadata |
| POST | `/api/diseases` | Add disease via JSON body |
| PUT | `/api/diseases/<name>` | Update existing disease |
| POST | `/api/upload/csv` | Bulk upload CSV → kb_loader |
| POST | `/api/upload/json` | Bulk upload JSON → kb_loader |
| GET | `/api/export/json` | Download full KB |
| POST | `/api/reset` | Reset to knowledge_base.py defaults |

### `api.js` — React API Client
Centralised fetch wrapper for all Flask endpoints. Every function returns `{ ok, data?, error? }`. If `ok` is false, `App.jsx` falls back to `bayesianEngine.js`.

### `App.jsx` — React Frontend
Manages all state:
- Patient context (age, sex, duration, chronic conditions, pain level, flags)
- Selected symptoms (Set) + free text input
- Results from either Flask API or browser engine

**Key components:**
- `RiskBanner` — Emergency/Moderate/Low with colour coding
- `DiagnosisTable` — top 6 conditions with animated probability bars
- `XAIPanel` — explanations + context notes in dark card
- `KBManager` — disease list, file upload, export/reset (sidebar)

---

## Running the Project

### Backend (Flask API)
```bash
cd backend
pip install -r requirements.txt
python api.py
# → http://localhost:5000
```

### Streamlit demo (alternative UI)
```bash
cd backend
streamlit run app.py
# → http://localhost:8501
```

### Frontend (React)
```bash
cd frontend
npm install
npm run dev
# → http://localhost:3000
# Vite proxies /api/* to http://localhost:5000
```

Run both simultaneously for full-stack mode. If Flask is not running, the React app falls back to browser-side Bayesian inference automatically.

---

## Expanding the Knowledge Base

### Option 1: Edit knowledge_base.py directly
Add to the `DISEASES` dict. Also add the same entry to `knowledgeBase.js`.

### Option 2: Upload via React UI
Go to sidebar → **Knowledge Base** tab → **Upload dataset** → choose `.csv` or `.json`.

### Option 3: Upload via API
```bash
curl -X POST http://localhost:5000/api/upload/csv \
  -F "file=@sample_expansion.csv"
```

Sample datasets (`sample_expansion.csv` and `sample_expansion.json`) are included — they add 7 more diseases (Chickenpox, Hepatitis B, Sinusitis, Meningitis, Leptospirosis, IBS, Heat Stroke).

---

## Medical Disclaimer
SmartTriage AI is an educational/academic project. It does **not** replace professional medical advice, diagnosis, or treatment. All probabilities are approximations for demonstration purposes. Always consult a qualified healthcare professional for medical decisions.
