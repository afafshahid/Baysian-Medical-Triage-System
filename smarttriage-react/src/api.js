/**
 * api.js
 * ─────────────────────────────────────────────────────────────────────────────
 * Centralised API client for the Flask backend (smart_triage/api.py).
 *
 * ENDPOINTS USED:
 *   GET  /api/health            → check if API is alive
 *   POST /api/analyze           → run Bayesian triage
 *   GET  /api/diseases          → list all diseases in runtime KB
 *   POST /api/diseases          → add a disease (JSON body)
 *   POST /api/upload/csv        → upload CSV to expand KB
 *   POST /api/upload/json       → upload JSON to expand KB
 *   GET  /api/export/json       → download full KB as JSON
 *   POST /api/reset             → reset KB to hardcoded defaults
 *
 * All functions return { ok: boolean, data?, error? }.
 * If the API is unreachable, callers should fall back to bayesianEngine.js.
 * ─────────────────────────────────────────────────────────────────────────────
 */

const BASE = '/api'; // proxied to http://localhost:5000 via vite.config.js

async function request(method, path, body = null, isFormData = false) {
  try {
    const opts = {
      method,
      headers: isFormData ? {} : { 'Content-Type': 'application/json' },
      body: body
        ? isFormData ? body : JSON.stringify(body)
        : undefined,
    };
    const res = await fetch(`${BASE}${path}`, opts);
    const data = await res.json();
    if (!res.ok) return { ok: false, error: data.error || `HTTP ${res.status}` };
    return { ok: true, data };
  } catch (err) {
    return { ok: false, error: err.message || 'Network error' };
  }
}

// ── Health ────────────────────────────────────────────────────────────────────
export async function checkHealth() {
  return request('GET', '/health');
}

// ── Triage analysis ───────────────────────────────────────────────────────────
/**
 * @param {string[]} symptoms        - list of snake_case symptom keys
 * @param {object}  patientContext   - { age, sex, duration, chronic, pain_level, flags }
 * @returns {Promise<{ok, data?, error?}>}
 *
 * data shape (mirrors inference_engine.py output):
 *   diagnoses        → [{ disease, probability, probability_pct, urgency, description }]
 *   risk_level       → "Emergency" | "Moderate" | "Low"
 *   confidence       → float 0–1
 *   emergency_override → string | null
 *   explanations     → string[]
 *   context_notes    → string[]
 */
export async function runAnalysis(symptoms, patientContext) {
  return request('POST', '/analyze', {
    symptoms,
    patient_context: patientContext,
  });
}

// ── Diseases CRUD ─────────────────────────────────────────────────────────────
export async function getDiseases() {
  return request('GET', '/diseases');
}

/**
 * @param {object} disease  - { name, prior, urgency, description, symptoms: {key: prob} }
 */
export async function addDisease(disease) {
  return request('POST', '/diseases', disease);
}

// ── File upload ───────────────────────────────────────────────────────────────
/**
 * Upload a CSV or JSON file to expand the runtime knowledge base.
 * @param {File} file
 */
export async function uploadFile(file) {
  const formData = new FormData();
  formData.append('file', file);
  const ext = file.name.split('.').pop().toLowerCase();
  const endpoint = ext === 'json' ? '/upload/json' : '/upload/csv';
  return request('POST', endpoint, formData, true);
}

// ── Export ────────────────────────────────────────────────────────────────────
export async function exportKB() {
  return request('GET', '/export/json');
}

// ── Reset ─────────────────────────────────────────────────────────────────────
export async function resetKB() {
  return request('POST', '/reset');
}
