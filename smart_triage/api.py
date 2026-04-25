"""
SmartTriage AI – Flask REST API Backend
Endpoints for triage analysis, knowledge base management, and dataset upload.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import json, os, csv, io
from knowledge_base import DISEASES, EMERGENCY_RULES
from inference_engine import BayesianTriage
from kb_loader import KnowledgeBaseLoader

app = Flask(__name__)
CORS(app)

# Mutable runtime KB (starts from hardcoded, can be extended)
runtime_kb = {k: v.copy() for k, v in DISEASES.items()}
loader = KnowledgeBaseLoader(runtime_kb)


def get_engine():
    return BayesianTriage(runtime_kb, EMERGENCY_RULES)


# ── Health check ─────────────────────────────────────────────────────────────
@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "diseases_loaded": len(runtime_kb)})


# ── Main triage analysis ──────────────────────────────────────────────────────
@app.route("/api/analyze", methods=["POST"])
def analyze():
    data = request.get_json()
    if not data:
        return jsonify({"error": "JSON body required"}), 400

    symptoms = data.get("symptoms", [])
    patient_ctx = data.get("patient_context", {})

    if not symptoms:
        return jsonify({"error": "At least one symptom required"}), 400

    engine = get_engine()
    result = engine.analyze(symptoms, patient_ctx)

    # Serialize (remove disease_symptoms from output for clean API response)
    clean_diagnoses = []
    for d in result["diagnoses"][:10]:
        clean_diagnoses.append({
            "disease": d["disease"],
            "probability": round(d["probability"], 4),
            "probability_pct": round(d["probability"] * 100, 1),
            "urgency": d["urgency"],
            "description": d.get("description", ""),
        })

    return jsonify({
        "diagnoses": clean_diagnoses,
        "risk_level": result["risk_level"],
        "confidence": round(result["confidence"], 4),
        "emergency_override": result["emergency_override"],
        "explanations": result["explanations"],
        "context_notes": result["context_notes"],
        "symptoms_analyzed": symptoms,
        "diseases_in_kb": len(runtime_kb),
    })


# ── List all symptoms ─────────────────────────────────────────────────────────
@app.route("/api/symptoms", methods=["GET"])
def list_symptoms():
    all_syms = sorted(set(
        s for d in runtime_kb.values() for s in d["symptoms"].keys()
    ))
    return jsonify({"symptoms": all_syms, "count": len(all_syms)})


# ── List all diseases ─────────────────────────────────────────────────────────
@app.route("/api/diseases", methods=["GET"])
def list_diseases():
    result = []
    for name, data in runtime_kb.items():
        result.append({
            "name": name,
            "urgency": data["urgency"],
            "prior": data["prior"],
            "symptom_count": len(data["symptoms"]),
            "description": data.get("description", ""),
        })
    return jsonify({"diseases": result, "count": len(result)})


# ── Add a new disease via JSON ────────────────────────────────────────────────
@app.route("/api/diseases", methods=["POST"])
def add_disease():
    """
    POST body:
    {
      "name": "New Disease",
      "prior": 0.05,
      "urgency": "Moderate",
      "description": "...",
      "symptoms": {"fever": 0.8, "cough": 0.7}
    }
    """
    data = request.get_json()
    if not data:
        return jsonify({"error": "JSON body required"}), 400

    required = ["name", "prior", "urgency", "symptoms"]
    missing = [f for f in required if f not in data]
    if missing:
        return jsonify({"error": f"Missing fields: {missing}"}), 400

    name = data["name"]
    if name in runtime_kb:
        return jsonify({"error": f"Disease '{name}' already exists. Use PUT to update."}), 409

    runtime_kb[name] = {
        "prior": float(data["prior"]),
        "urgency": data["urgency"],
        "description": data.get("description", ""),
        "symptoms": {k: float(v) for k, v in data["symptoms"].items()},
    }
    return jsonify({"message": f"Disease '{name}' added successfully", "total": len(runtime_kb)}), 201


# ── Update existing disease ───────────────────────────────────────────────────
@app.route("/api/diseases/<path:name>", methods=["PUT"])
def update_disease(name):
    if name not in runtime_kb:
        return jsonify({"error": f"Disease '{name}' not found"}), 404
    data = request.get_json()
    if "symptoms" in data:
        runtime_kb[name]["symptoms"].update({k: float(v) for k, v in data["symptoms"].items()})
    for field in ["prior", "urgency", "description"]:
        if field in data:
            runtime_kb[name][field] = data[field]
    return jsonify({"message": f"Disease '{name}' updated"})


# ── Upload CSV dataset to extend KB ──────────────────────────────────────────
@app.route("/api/upload/csv", methods=["POST"])
def upload_csv():
    """
    Expected CSV format:
    disease,urgency,prior,description,symptom1,prob1,symptom2,prob2,...
    """
    if "file" not in request.files:
        return jsonify({"error": "No file provided. Send as multipart/form-data with key 'file'"}), 400

    file = request.files["file"]
    if not file.filename.endswith(".csv"):
        return jsonify({"error": "Only CSV files accepted"}), 400

    content = file.read().decode("utf-8")
    added, errors = loader.load_from_csv_string(content)
    return jsonify({
        "added": added,
        "errors": errors,
        "total_diseases": len(runtime_kb)
    })


# ── Upload JSON dataset to extend KB ─────────────────────────────────────────
@app.route("/api/upload/json", methods=["POST"])
def upload_json():
    """
    Expected JSON: { "DiseaseName": { "prior": 0.05, "urgency": "Low",
                     "description": "...", "symptoms": {"sym": 0.8} }, ... }
    """
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    content = file.read().decode("utf-8")
    try:
        new_diseases = json.loads(content)
    except json.JSONDecodeError as e:
        return jsonify({"error": f"Invalid JSON: {e}"}), 400

    added, errors = loader.load_from_dict(new_diseases)
    return jsonify({
        "added": added,
        "errors": errors,
        "total_diseases": len(runtime_kb)
    })


# ── Export current KB as JSON ─────────────────────────────────────────────────
@app.route("/api/export/json", methods=["GET"])
def export_json():
    return jsonify(runtime_kb)


# ── Reset KB to hardcoded defaults ────────────────────────────────────────────
@app.route("/api/reset", methods=["POST"])
def reset_kb():
    global runtime_kb
    runtime_kb.clear()
    runtime_kb.update({k: v.copy() for k, v in DISEASES.items()})
    loader.kb = runtime_kb
    return jsonify({"message": "Knowledge base reset to defaults", "total": len(runtime_kb)})


if __name__ == "__main__":
    print("\n🩺 SmartTriage Flask API starting...")
    print("   Base URL: http://localhost:5000")
    print("   Endpoints:")
    print("     GET  /api/health")
    print("     POST /api/analyze")
    print("     GET  /api/symptoms")
    print("     GET  /api/diseases")
    print("     POST /api/diseases      (add disease)")
    print("     PUT  /api/diseases/<n>  (update disease)")
    print("     POST /api/upload/csv")
    print("     POST /api/upload/json")
    print("     GET  /api/export/json")
    print("     POST /api/reset\n")
    app.run(debug=True, port=5000)
