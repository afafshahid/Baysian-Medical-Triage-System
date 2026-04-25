/**
 * App.jsx — SmartTriage AI React Frontend
 *
 * Data flow:
 *   knowledgeBase.js  ─→  symptom list, disease metadata (displayed in UI)
 *   api.js            ─→  Flask /api/analyze  (when API online)
 *   bayesianEngine.js ─→  browser Bayes fallback (when API offline)
 *
 * On submit:
 *   1. Try api.runAnalysis(symptoms, ctx)
 *   2. If that fails → bayesianEngine.analyze(symptoms, ctx)
 *   3. Render results from whichever succeeded
 */

import { useState, useEffect, useMemo, useRef } from 'react';
import { ALL_SYMPTOMS, EMERGENCY_SYMPTOMS, DISEASES } from './knowledgeBase.js';
import { analyze as localAnalyze } from './bayesianEngine.js';
import { checkHealth, runAnalysis, uploadFile, getDiseases, resetKB, exportKB } from './api.js';
import styles from './App.module.css';

// ─── Helpers ──────────────────────────────────────────────────────────────────
const fmt = (n) => (n * 100).toFixed(1) + '%';

function toSnake(str) {
  return str.toLowerCase().replace(/\s+/g, '_').replace(/[()\/]/g, '');
}

function labelOf(key) {
  return key.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
}

const URGENCY_COLOR = { Emergency: 'red', Moderate: 'amber', Low: 'teal' };
const CHRONIC_OPTIONS = ['Diabetes', 'Hypertension', 'Heart Disease', 'Asthma', 'Immunocompromised', 'Pregnancy', 'Obesity'];
const DURATION_OPTIONS = ['< 24 hours', '1–3 days', '3–7 days', '> 1 week'];
const FLAGS = [
  'Difficulty breathing', 'Chest tightness', 'Confusion / disorientation',
  'Sudden severe headache', 'Loss of consciousness', 'Uncontrolled bleeding',
  'High fever > 39°C', 'Rapid heart rate',
];

// ─── Sub-components ───────────────────────────────────────────────────────────

function ApiStatus({ online, diseaseCount }) {
  return (
    <div className={`${styles.statusPill} ${online ? styles.statusOnline : styles.statusOffline}`}>
      <span className={styles.statusDot} />
      {online
        ? `API online · ${diseaseCount} diseases`
        : 'API offline · browser mode'}
    </div>
  );
}

function CheckRow({ label, checked, onChange }) {
  return (
    <label className={styles.checkRow}>
      <span className={`${styles.checkBox} ${checked ? styles.checkOn : ''}`} />
      <span className={styles.checkLabel}>{label}</span>
      <input type="checkbox" checked={checked} onChange={onChange} style={{ display: 'none' }} />
    </label>
  );
}

function SymTag({ label, isOn, isEmergency, onClick }) {
  return (
    <button
      onClick={onClick}
      className={`${styles.symTag}
        ${isOn ? styles.symOn : ''}
        ${isEmergency ? styles.symEmergency : ''}
        ${isOn && isEmergency ? styles.symEmergencyOn : ''}`}
    >
      {labelOf(label)}
    </button>
  );
}

function RiskBanner({ risk, emergencyOverride }) {
  const cfg = {
    Emergency: {
      cls: styles.riskEmergency,
      icon: '🚨',
      rec: 'Seek immediate emergency department care — call 1122 / 115 now',
      score: '90+',
    },
    Moderate: {
      cls: styles.riskModerate,
      icon: '⚠️',
      rec: 'Schedule a doctor appointment within 24–48 hours',
      score: '68',
    },
    Low: {
      cls: styles.riskLow,
      icon: '✓',
      rec: 'Home care & symptom monitoring recommended',
      score: '15',
    },
  }[risk];

  return (
    <div className={`${styles.riskBanner} ${cfg.cls} anim-fade-up`}>
      <div className={styles.riskLeft}>
        <div className={styles.riskEyebrow}>Risk classification</div>
        <div className={styles.riskTitle}>{risk} risk</div>
        {emergencyOverride && (
          <div className={styles.riskOverride}>⚡ {emergencyOverride}</div>
        )}
        <div className={styles.riskRec}>{cfg.rec}</div>
      </div>
      <div className={styles.riskScore}>{cfg.score}</div>
    </div>
  );
}

function DiagnosisTable({ diagnoses }) {
  return (
    <div className={`${styles.diagCard} anim-fade-up`}>
      <div className={styles.diagHead}>
        Probable conditions — posterior probability P(D|S)
      </div>
      {diagnoses.slice(0, 6).map((d, i) => {
        const pct = d.probability * 100;
        const urg = d.urgency || d.urgency_level || 'Moderate';
        const color = URGENCY_COLOR[urg] || 'teal';
        return (
          <div key={d.disease} className={styles.diagRow}>
            <div className={styles.diagInfo}>
              <div className={styles.diagName}>{d.disease}</div>
              <div className={styles.diagDesc}>{d.description}</div>
            </div>
            <div className={styles.probBlock}>
              <span className={styles.probNum}>{pct.toFixed(1)}%</span>
              <div className={styles.probBarBg}>
                <div
                  className={`${styles.probBarFill} ${i === 0 ? styles.barTop : styles.barMid}`}
                  style={{ width: `${pct}%`, animation: 'barGrow 0.7s cubic-bezier(.22,1,.36,1) both' }}
                />
              </div>
            </div>
            <span className={`${styles.confBadge} ${styles[`conf${color}`]}`}>{urg}</span>
          </div>
        );
      })}
    </div>
  );
}

function XAIPanel({ explanations, contextNotes, confidence, source, onReset }) {
  return (
    <div className={`${styles.xaiCard} anim-fade-up`}>
      <div className={styles.xaiHead}>
        <svg width="14" height="14" viewBox="0 0 14 14" fill="none" stroke="#9b9a96" strokeWidth="1.4">
          <circle cx="7" cy="7" r="6" /><path d="M7 4v3l2 2" strokeLinecap="round" />
        </svg>
        Inference logic (XAI) · confidence: {(confidence * 100).toFixed(0)}% · engine: {source}
      </div>
      <div className={styles.xaiSections}>
        {explanations.length > 0 && (
          <div>
            <div className={styles.xaiSectionLabel}>Why these results?</div>
            {explanations.map((e, i) => (
              <div key={i} className={styles.xaiItem}>→ {e}</div>
            ))}
          </div>
        )}
        {contextNotes.length > 0 && (
          <div>
            <div className={styles.xaiSectionLabel}>Context adjustments applied</div>
            {contextNotes.map((n, i) => (
              <div key={i} className={`${styles.xaiItem} ${styles.xaiCtx}`}>⟳ {n}</div>
            ))}
          </div>
        )}
      </div>
      <div className={styles.xaiFooter}>
        <button className={styles.resetBtn} onClick={onReset}>↺ New assessment</button>
        <span className={styles.xaiHash}>HASH: 7F2-B09-XAI</span>
      </div>
    </div>
  );
}

// ─── KB Manager Panel ─────────────────────────────────────────────────────────
function KBManager({ apiOnline, onRefresh }) {
  const [tab, setTab] = useState('list');
  const [diseases, setDiseases] = useState([]);
  const [uploadStatus, setUploadStatus] = useState('');
  const fileRef = useRef();

  useEffect(() => {
    if (apiOnline) {
      getDiseases().then(r => {
        if (r.ok) setDiseases(r.data.diseases);
      });
    } else {
      // Show local KB when offline
      setDiseases(Object.entries(DISEASES).map(([name, d]) => ({
        name,
        urgency: d.urgency,
        prior: d.prior,
        symptom_count: Object.keys(d.symptoms).length,
        description: d.description,
      })));
    }
  }, [apiOnline]);

  async function handleUpload(e) {
    const file = e.target.files[0];
    if (!file) return;
    setUploadStatus('Uploading…');
    const r = await uploadFile(file);
    if (r.ok) {
      setUploadStatus(`✓ Added: ${r.data.added.join(', ')} — Total: ${r.data.total_diseases}`);
      onRefresh();
    } else {
      setUploadStatus(`✗ ${r.error}`);
    }
  }

  async function handleExport() {
    const r = await exportKB();
    if (!r.ok) return;
    const blob = new Blob([JSON.stringify(r.data, null, 2)], { type: 'application/json' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = 'smarttriage_kb.json';
    a.click();
  }

  async function handleReset() {
    if (!window.confirm('Reset KB to hardcoded defaults?')) return;
    const r = await resetKB();
    if (r.ok) { setUploadStatus('KB reset.'); onRefresh(); }
  }

  const URGENCY_BADGE = { Emergency: styles.badgeRed, Moderate: styles.badgeAmber, Low: styles.badgeTeal };

  return (
    <div className={styles.kbManager}>
      <div className={styles.kbTabs}>
        {['list', 'upload', 'actions'].map(t => (
          <button
            key={t}
            className={`${styles.kbTab} ${tab === t ? styles.kbTabOn : ''}`}
            onClick={() => setTab(t)}
          >
            {t === 'list' ? `Diseases (${diseases.length})` : t === 'upload' ? 'Upload dataset' : 'Actions'}
          </button>
        ))}
      </div>

      {tab === 'list' && (
        <div className={styles.kbList}>
          {diseases.map(d => (
            <div key={d.name} className={styles.kbRow}>
              <div>
                <div className={styles.kbRowName}>{d.name}</div>
                <div className={styles.kbRowMeta}>{d.symptom_count} symptoms · prior {d.prior}</div>
              </div>
              <span className={`${styles.badge} ${URGENCY_BADGE[d.urgency]}`}>{d.urgency}</span>
            </div>
          ))}
        </div>
      )}

      {tab === 'upload' && (
        <div className={styles.kbUpload}>
          <div className={styles.kbUploadNote}>
            <strong>CSV format:</strong><br />
            <code>disease, urgency, prior, description, symptom1, prob1, ...</code>
            <br /><br />
            <strong>JSON format:</strong><br />
            <code>{`{ "Disease": { "prior": 0.05, "urgency": "Low", "symptoms": { "fever": 0.8 } } }`}</code>
          </div>
          <input type="file" accept=".csv,.json" ref={fileRef} onChange={handleUpload} style={{ display: 'none' }} />
          <button className={styles.uploadBtn} onClick={() => fileRef.current?.click()}>
            Choose CSV or JSON file
          </button>
          {uploadStatus && (
            <div className={`${styles.uploadStatus} ${uploadStatus.startsWith('✓') ? styles.uploadOk : uploadStatus.startsWith('✗') ? styles.uploadErr : ''}`}>
              {uploadStatus}
            </div>
          )}
        </div>
      )}

      {tab === 'actions' && (
        <div className={styles.kbActions}>
          <button className={styles.actionBtn} onClick={handleExport} disabled={!apiOnline}>
            Export KB as JSON
          </button>
          <button className={`${styles.actionBtn} ${styles.actionDanger}`} onClick={handleReset} disabled={!apiOnline}>
            Reset to defaults
          </button>
          {!apiOnline && (
            <div className={styles.offlineNote}>Actions require the Flask API to be running.</div>
          )}
        </div>
      )}
    </div>
  );
}

// ─── Main App ─────────────────────────────────────────────────────────────────
export default function App() {
  // Patient context
  const [age, setAge]           = useState(30);
  const [sex, setSex]           = useState('Male');
  const [duration, setDuration] = useState('1–3 days');
  const [chronic, setChronic]   = useState([]);
  const [painLevel, setPainLevel] = useState(3);
  const [flags, setFlags]       = useState([]);

  // Symptoms
  const [selected, setSelected]   = useState(new Set());
  const [freeText, setFreeText]   = useState('');
  const [symFilter, setSymFilter] = useState('');

  // UI state
  const [results, setResults]     = useState(null);
  const [loading, setLoading]     = useState(false);
  const [apiOnline, setApiOnline] = useState(false);
  const [apiDiseases, setApiDiseases] = useState(0);
  const [sidePanel, setSidePanel] = useState('patient'); // 'patient' | 'kb'

  // Check API on mount
  useEffect(() => {
    checkHealth().then(r => {
      setApiOnline(r.ok);
      if (r.ok) setApiDiseases(r.data?.diseases_loaded ?? 0);
    });
  }, []);

  function toggleChronic(c) {
    setChronic(prev => prev.includes(c) ? prev.filter(x => x !== c) : [...prev, c]);
  }
  function toggleFlag(f) {
    setFlags(prev => prev.includes(f) ? prev.filter(x => x !== f) : [...prev, f]);
  }
  function toggleSym(key) {
    setSelected(prev => {
      const next = new Set(prev);
      next.has(key) ? next.delete(key) : next.add(key);
      return next;
    });
  }

  // Parse free text against ALL_SYMPTOMS
  function parseText(text) {
    const lower = text.toLowerCase();
    const matched = new Set();
    for (const key of ALL_SYMPTOMS) {
      const readable = key.replace(/_/g, ' ');
      if (lower.includes(readable) || lower.includes(key)) matched.add(key);
      else {
        const words = readable.split(' ');
        if (words.some(w => w.length > 4 && lower.includes(w))) matched.add(key);
      }
    }
    return matched;
  }

  const filteredSymptoms = useMemo(() => {
    if (!symFilter) return ALL_SYMPTOMS;
    const f = symFilter.toLowerCase();
    return ALL_SYMPTOMS.filter(k => k.includes(f) || labelOf(k).toLowerCase().includes(f));
  }, [symFilter]);

  const patientCtx = {
    age, sex, duration,
    chronic: chronic.length ? chronic : ['None'],
    pain_level: painLevel,
    flags,
  };

  async function handleAnalyze() {
    let symptoms = new Set(selected);
    if (freeText.trim()) {
      const parsed = parseText(freeText);
      parsed.forEach(s => symptoms.add(s));
    }
    const symList = [...symptoms];
    if (symList.length === 0) return;

    setLoading(true);
    let res;

    if (apiOnline) {
      const r = await runAnalysis(symList, patientCtx);
      if (r.ok) {
        // Normalise API response to same shape as local engine
        res = {
          diagnoses: r.data.diagnoses.map(d => ({
            ...d,
            probability: d.probability,
            diseaseSymptoms: {},
          })),
          riskLevel: r.data.risk_level,
          confidence: r.data.confidence,
          emergencyOverride: r.data.emergency_override,
          explanations: r.data.explanations,
          contextNotes: r.data.context_notes,
          source: 'Flask API',
        };
      } else {
        // Fallback
        res = { ...localAnalyze(symList, patientCtx), source: 'browser (API error)' };
      }
    } else {
      res = { ...localAnalyze(symList, patientCtx), source: 'browser' };
    }

    setResults(res);
    setLoading(false);
  }

  function handleReset() {
    setResults(null);
    setSelected(new Set());
    setFreeText('');
    setFlags([]);
    setPainLevel(3);
  }

  const canSubmit = selected.size > 0 || freeText.trim().length > 0;

  return (
    <div className={styles.shell}>

      {/* ── Sidebar ── */}
      <aside className={styles.sidebar}>
        <div className={styles.logo}>
          <div className={styles.logoMark}>
            <div className={styles.logoSq}>
              <svg viewBox="0 0 16 16" fill="none" stroke="#fff" strokeWidth="2">
                <circle cx="8" cy="8" r="5" /><path d="M5 8h6M8 5v6" strokeLinecap="round" />
              </svg>
            </div>
            <span className={styles.logoName}>SmartTriage</span>
          </div>
          <div className={styles.logoSub}>Bayesian Risk Engine</div>
        </div>

        <div className={styles.sideTabs}>
          <button
            className={`${styles.sideTab} ${sidePanel === 'patient' ? styles.sideTabOn : ''}`}
            onClick={() => setSidePanel('patient')}
          >Patient</button>
          <button
            className={`${styles.sideTab} ${sidePanel === 'kb' ? styles.sideTabOn : ''}`}
            onClick={() => setSidePanel('kb')}
          >Knowledge Base</button>
        </div>

        <div className={styles.sideBody}>
          {sidePanel === 'patient' ? (
            <>
              <div className={styles.fieldGroup}>
                <label className={styles.fieldLabel}>Age</label>
                <input type="number" value={age} min={1} max={110}
                  onChange={e => setAge(parseInt(e.target.value) || 1)}
                  className={styles.input} />
              </div>

              <div className={styles.fieldGroup}>
                <label className={styles.fieldLabel}>Biological sex</label>
                <select value={sex} onChange={e => setSex(e.target.value)} className={styles.input}>
                  <option>Male</option>
                  <option>Female</option>
                  <option>Other</option>
                </select>
              </div>

              <div className={styles.fieldGroup}>
                <label className={styles.fieldLabel}>Symptom duration</label>
                <select value={duration} onChange={e => setDuration(e.target.value)} className={styles.input}>
                  {DURATION_OPTIONS.map(d => <option key={d}>{d}</option>)}
                </select>
              </div>

              <div className={styles.fieldGroup}>
                <label className={styles.fieldLabel}>Pain level: {painLevel}/10</label>
                <input type="range" min={0} max={10} value={painLevel}
                  onChange={e => setPainLevel(+e.target.value)}
                  className={styles.slider} />
              </div>

              <div className={styles.fieldGroup}>
                <label className={styles.fieldLabel}>Chronic conditions</label>
                <div className={styles.checkList}>
                  {CHRONIC_OPTIONS.map(c => (
                    <CheckRow key={c} label={c} checked={chronic.includes(c)}
                      onChange={() => toggleChronic(c)} />
                  ))}
                </div>
              </div>

              <div className={styles.fieldGroup}>
                <label className={styles.fieldLabel}>Emergency flags</label>
                <div className={styles.checkList}>
                  {FLAGS.map(f => (
                    <CheckRow key={f} label={f} checked={flags.includes(f)}
                      onChange={() => toggleFlag(f)} />
                  ))}
                </div>
              </div>
            </>
          ) : (
            <KBManager apiOnline={apiOnline} onRefresh={() => {
              checkHealth().then(r => {
                setApiOnline(r.ok);
                if (r.ok) setApiDiseases(r.data?.diseases_loaded ?? 0);
              });
            }} />
          )}
        </div>

        <div className={styles.sideFooter}>
          <ApiStatus online={apiOnline} diseaseCount={apiOnline ? apiDiseases : Object.keys(DISEASES).length} />
        </div>
      </aside>

      {/* ── Main ── */}
      <main className={styles.main}>
        <header className={styles.topbar}>
          <span className={styles.topbarTitle}>Diagnostic dashboard</span>
          <div className={styles.topbarRight}>
            <span className={styles.topbarSub}>
              {Object.keys(DISEASES).length} core diseases · {ALL_SYMPTOMS.length} symptoms
            </span>
          </div>
        </header>

        <div className={styles.workspace}>

          {/* ── Intake ── */}
          <section className={styles.intake}>

            <div className={styles.intakeSection}>
              <div className={styles.sectionHead}>1. Natural language input</div>
              <textarea
                className={styles.textarea}
                placeholder="Describe symptoms — onset, severity, modifiers… (e.g. 'high fever for 2 days with bad headache')"
                value={freeText}
                onChange={e => setFreeText(e.target.value)}
              />
              {freeText.trim() && (
                <div className={styles.parsedPreview}>
                  Detected: {[...parseText(freeText)].map(s => (
                    <span key={s} className={styles.parsedChip}>{labelOf(s)}</span>
                  ))}
                </div>
              )}
            </div>

            <div className={styles.intakeSection}>
              <div className={styles.sectionHead}>
                2. Symptom picker
                <span className={styles.selectedCount}>{selected.size} selected</span>
              </div>
              <input
                type="text"
                className={styles.symSearch}
                placeholder="Filter symptoms…"
                value={symFilter}
                onChange={e => setSymFilter(e.target.value)}
              />
              <div className={styles.tagCloud}>
                {filteredSymptoms.map(key => (
                  <SymTag
                    key={key}
                    label={key}
                    isOn={selected.has(key)}
                    isEmergency={EMERGENCY_SYMPTOMS.includes(key)}
                    onClick={() => toggleSym(key)}
                  />
                ))}
              </div>
            </div>

            {selected.size > 0 && (
              <div className={styles.intakeSection}>
                <div className={styles.sectionHead}>Selected</div>
                <div className={styles.chipCloud}>
                  {[...selected].map(s => (
                    <span key={s} className={styles.chip}>
                      {labelOf(s)}
                      <button className={styles.chipX} onClick={() => toggleSym(s)}>×</button>
                    </span>
                  ))}
                </div>
              </div>
            )}

            <button
              className={styles.runBtn}
              onClick={handleAnalyze}
              disabled={!canSubmit || loading}
            >
              {loading
                ? <><svg className={`anim-spin ${styles.btnIcon}`} viewBox="0 0 16 16"><circle cx="8" cy="4" r="2" fill="currentColor" /></svg> Processing…</>
                : <><svg className={styles.btnIcon} viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.8"><circle cx="8" cy="8" r="6" /><path d="M5 8h6M8 5v6" strokeLinecap="round" /></svg> Calculate probabilities</>
              }
            </button>
          </section>

          {/* ── Results ── */}
          <section className={styles.results}>
            {!results ? (
              <div className={styles.emptyState}>
                <div className={styles.emptyIcon}>
                  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.2">
                    <circle cx="12" cy="12" r="9" /><path d="M12 7v5l3 3" strokeLinecap="round" />
                  </svg>
                </div>
                <div className={styles.emptyTitle}>Awaiting assessment</div>
                <div className={styles.emptySub}>
                  Select symptoms and click <em>Calculate probabilities</em> to run Bayesian inference.
                </div>
                <div className={styles.emptyFormula}>
                  P(D|S) = P(S|D) · P(D) / P(S)
                </div>
              </div>
            ) : (
              <div className={styles.resultStack}>
                <RiskBanner risk={results.riskLevel} emergencyOverride={results.emergencyOverride} />
                <DiagnosisTable diagnoses={results.diagnoses} />
                <XAIPanel
                  explanations={results.explanations}
                  contextNotes={results.contextNotes}
                  confidence={results.confidence}
                  source={results.source}
                  onReset={handleReset}
                />
                <div className={styles.disclaimer}>
                  ⚕️ SmartTriage AI is an educational tool only. It does not replace professional medical advice.
                  Always consult a qualified healthcare provider.
                </div>
              </div>
            )}
          </section>
        </div>

        <footer className={styles.footer}>
          <span>AI PROJECT | SPRING 2026 · AMNA RAIS (23K-0824) · AFAF SHAHID (23K-0678) · HAMNA KHALID (23K-0700)</span>
          <span>AI Project 2026 · Bayesian Medical Triage</span>
        </footer>
      </main>
    </div>
  );
}
