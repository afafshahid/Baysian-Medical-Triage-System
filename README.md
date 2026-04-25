# Bayesian Medical Triage System (SmartTriage AI)

SmartTriage AI is an academic full-stack triage assistant that estimates likely conditions from symptoms using a Bayesian inference engine, then classifies urgency as **Emergency**, **Moderate**, or **Low**.

It includes:
- a **Python backend** (Flask API + Streamlit app),
- a **React frontend** (Vite),
- a runtime-expandable **knowledge base** of diseases, symptoms, and emergency rules,
- an offline browser fallback engine when the API is unavailable.

---

## Repository Structure

```text
Bayesian-Medical-Triage-System/
├── README.md
├── smart_triage/                # Python backend + Streamlit UI
│   ├── api.py                   # Flask REST API
│   ├── app.py                   # Streamlit app
│   ├── inference_engine.py      # BayesianTriage algorithm
│   ├── knowledge_base.py        # Disease priors, symptoms, emergency rules
│   ├── kb_loader.py             # CSV/JSON KB expansion utilities
│   ├── requirements.txt
│   ├── sample_expansion.csv
│   └── sample_expansion.json
└── smarttriage-react/           # React frontend
    ├── package.json
    ├── vite.config.js
    ├── src/
    │   ├── App.jsx
    │   ├── api.js
    │   ├── bayesianEngine.js
    │   └── knowledgeBase.js
    └── README.md
```

---

## How the System Works

1. User selects/enters symptoms and patient context (age, sex, duration, chronic conditions, emergency flags).
2. Frontend calls `POST /api/analyze` when backend is online.
3. Backend computes posterior disease probabilities and urgency with Bayesian logic + emergency override rules.
4. Frontend renders top diagnoses, confidence, explanations, and context notes.
5. If API is offline, React falls back to `bayesianEngine.js` and runs inference in-browser.

---

## Core Components

### Backend (`smart_triage`)
- **`inference_engine.py`**: Bayesian posterior computation, prior adjustments, risk scoring, explanation generation.
- **`knowledge_base.py`**: disease metadata (`prior`, `urgency`, `description`, `symptoms`) and emergency rules.
- **`kb_loader.py`**: validates and loads new diseases from JSON/CSV.
- **`api.py`**: REST endpoints for analysis, KB listing/update/upload/export/reset.
- **`app.py`**: Streamlit interface for interactive triage and KB management.

### Frontend (`smarttriage-react`)
- **`App.jsx`**: main dashboard UI and interaction flow.
- **`api.js`**: centralized API client for Flask endpoints.
- **`bayesianEngine.js`**: browser fallback inference engine.
- **`knowledgeBase.js`**: frontend knowledge base mirror used for symptom list and offline mode.

---

## Setup and Run

### 1) Backend setup

```bash
cd smart_triage
python -m pip install -r requirements.txt
```

Run Flask API:
```bash
python api.py
```

Run Streamlit app (optional/alternative UI):
```bash
streamlit run app.py
```

### 2) Frontend setup

```bash
cd smarttriage-react
npm install
npm run dev
```

Vite is configured in `vite.config.js` to serve on `http://localhost:3000` and proxy `/api/*` to `http://localhost:5000`.

---

## API Reference (Flask)

Base URL: `http://localhost:5000`

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/api/health` | Health check and disease count |
| POST | `/api/analyze` | Run triage analysis |
| GET | `/api/symptoms` | List all symptom keys |
| GET | `/api/diseases` | List diseases in runtime KB |
| POST | `/api/diseases` | Add one disease |
| PUT | `/api/diseases/<name>` | Update disease fields |
| POST | `/api/upload/csv` | Upload CSV disease dataset |
| POST | `/api/upload/json` | Upload JSON disease dataset |
| GET | `/api/export/json` | Export runtime KB |
| POST | `/api/reset` | Reset KB to defaults |

---

## Knowledge Base Expansion

You can expand KB at runtime in three ways:
1. Upload CSV/JSON in the React knowledge-base panel.
2. Upload CSV/JSON through API endpoints.
3. Edit `knowledge_base.py` (and mirror important updates in frontend `knowledgeBase.js` for offline mode consistency).

### CSV format
```csv
disease,urgency,prior,description,symptom1,prob1,symptom2,prob2
Example Disease,Moderate,0.05,Example description,fever,0.8,cough,0.7
```

### JSON format
```json
{
  "Example Disease": {
    "prior": 0.05,
    "urgency": "Moderate",
    "description": "Example description",
    "symptoms": {
      "fever": 0.8,
      "cough": 0.7
    }
  }
}
```

---

## Notes and Limitations

- This is an educational AI project and not a clinical-grade medical system.
- Posterior scores are model-based estimates, not diagnoses.
- Emergency outputs should always trigger real clinical escalation.
- Frontend offline mode is useful for resilience, but keep backend/frontend knowledge bases aligned.

---

## Medical Disclaimer

This project is for educational and demonstration purposes only. It does **not** replace professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare professional for medical decisions.
