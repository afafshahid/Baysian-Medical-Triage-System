/**
 * bayesianEngine.js
 * Browser-side Bayesian inference engine — offline fallback when Flask API is unavailable.
 * AUTO-GENERATED to match inference_engine.py logic.
 * 125 diseases · 254 symptoms · 11 emergency rules
 */

import { DISEASES, EMERGENCY_RULES } from './knowledgeBase.js';

// ── Prior adjustment (mirrors _adjust_prior in inference_engine.py) ──────────
function adjustPrior(prior, diseaseName, ctx) {
  const age      = ctx.age      ?? 30;
  const chronic  = ctx.chronic  ?? [];
  const sex      = ctx.sex      ?? '';
  const duration = ctx.duration ?? '';
  let mult = 1.0;

  if (age < 5) {
    if (diseaseName.includes('Common Cold') || diseaseName.includes('Gastroenteritis')) mult *= 1.8;
    if (diseaseName.includes('Influenza'))  mult *= 1.4;
    if (diseaseName.includes('Pneumonia'))  mult *= 1.6;
  } else if (age > 60) {
    if (diseaseName.includes('Pneumonia'))       mult *= 2.0;
    if (diseaseName.includes('Hypertensive'))    mult *= 2.5;
    if (diseaseName.includes('Hypertension'))    mult *= 1.8;
    if (diseaseName.includes('Anemia'))          mult *= 1.5;
    if (diseaseName.includes('Heart Failure'))   mult *= 2.0;
    if (diseaseName.includes('Atrial'))          mult *= 1.8;
    if (diseaseName.includes('Alzheimer'))       mult *= 2.5;
    if (diseaseName.includes('Parkinson'))       mult *= 2.0;
    if (diseaseName.includes('Common Cold'))     mult *= 0.7;
    if (diseaseName.includes('Osteoarthritis'))  mult *= 2.0;
    if (diseaseName.includes('Cataract'))        mult *= 2.5;
    if (diseaseName.includes('Glaucoma'))        mult *= 1.8;
  } else if (age >= 15 && age <= 40) {
    if (diseaseName.includes('Migraine'))                         mult *= 1.5;
    if (diseaseName.includes('UTI') && sex === 'Female')          mult *= 2.0;
    if (diseaseName.includes('PCOS') && sex === 'Female')         mult *= 2.0;
    if (diseaseName.includes('Endometriosis') && sex === 'Female') mult *= 2.0;
    if (diseaseName.includes('Multiple Sclerosis'))                mult *= 1.5;
  }

  if (chronic.includes('Diabetes')) {
    if (diseaseName.includes('UTI'))                    mult *= 1.6;
    if (diseaseName.includes('Anemia'))                  mult *= 1.3;
    if (diseaseName.includes('Kidney'))                  mult *= 1.8;
    if (diseaseName.includes('Diabetic'))                mult *= 2.5;
    if (diseaseName.includes('Peripheral Arterial'))     mult *= 1.6;
    if (diseaseName.includes('Hypoglycemia'))            mult *= 2.0;
  }
  if (chronic.includes('Heart Disease')) {
    if (diseaseName.includes('Hypertensive'))  mult *= 2.0;
    if (diseaseName.includes('Heart Failure'))  mult *= 2.5;
    if (diseaseName.includes('Atrial'))         mult *= 1.8;
    if (diseaseName.includes('Heart Attack'))   mult *= 2.0;
    if (diseaseName.includes('Pericarditis'))   mult *= 1.5;
  }
  if (chronic.includes('Asthma')) {
    if (diseaseName.includes('Asthma')) mult *= 3.0;
    if (diseaseName.includes('COPD'))   mult *= 1.5;
  }
  if (chronic.includes('Hypertension')) {
    if (diseaseName.includes('Stroke'))        mult *= 2.0;
    if (diseaseName.includes('Heart Attack'))  mult *= 1.8;
    if (diseaseName.includes('Kidney'))        mult *= 1.5;
    if (diseaseName.includes('Hypertensive'))  mult *= 2.5;
  }
  if (chronic.includes('Immunocompromised')) {
    if (diseaseName.includes('Pneumonia'))    mult *= 2.5;
    if (diseaseName.includes('COVID'))        mult *= 1.8;
    if (diseaseName.includes('Tuberculosis')) mult *= 2.0;
    if (diseaseName.includes('Sepsis'))       mult *= 2.0;
  }
  if (chronic.includes('Pregnancy')) {
    if (diseaseName.includes('UTI'))           mult *= 1.8;
    if (diseaseName.includes('Anemia'))         mult *= 1.6;
    if (diseaseName.includes('Preeclampsia'))   mult *= 3.0;
    if (diseaseName.includes('Gestational'))    mult *= 3.0;
  }
  if (chronic.includes('Obesity')) {
    if (diseaseName.includes('Diabetes'))            mult *= 1.8;
    if (diseaseName.includes('Sleep Apnea'))          mult *= 2.5;
    if (diseaseName.includes('Heart'))                mult *= 1.5;
    if (diseaseName.includes('Osteoarthritis'))       mult *= 1.6;
    if (diseaseName.includes('Hypertension'))         mult *= 1.5;
    if (diseaseName.includes('Gallstone'))            mult *= 1.8;
  }

  if (duration.includes('> 1 week') || duration.includes('1 week')) {
    if (diseaseName.includes('Common Cold'))   mult *= 0.5;
    if (diseaseName.includes('Typhoid'))       mult *= 1.5;
    if (diseaseName.includes('Anemia'))        mult *= 1.5;
    if (diseaseName.includes('Tuberculosis'))  mult *= 1.8;
    if (diseaseName.includes('Depression'))    mult *= 1.5;
    if (diseaseName.includes('Chronic'))       mult *= 1.4;
  }

  return Math.min(prior * mult, 1.0);
}

// ── Core Bayesian computation ─────────────────────────────────────────────────
function computePosterior(symptoms, ctx) {
  const symNorm = symptoms.map(s => s.toLowerCase().replace(/ /g, '_'));
  const results = [];

  for (const [diseaseName, diseaseData] of Object.entries(DISEASES)) {
    let prior = adjustPrior(diseaseData.prior, diseaseName, ctx);
    let likelihood = 1.0;

    for (const sym of symNorm) {
      const p = diseaseData.symptoms[sym] ?? 0.02;
      likelihood *= p;
    }

    // Soft penalty for absent high-probability symptoms
    for (const [keySym, keyProb] of Object.entries(diseaseData.symptoms)) {
      if (!symNorm.includes(keySym) && keyProb > 0.80) {
        likelihood *= (1 - keyProb * 0.5);
      }
    }

    results.push({
      disease: diseaseName,
      unnormalized: likelihood * prior,
      probability: 0,
      urgency: diseaseData.urgency,
      description: diseaseData.description,
      diseaseSymptoms: diseaseData.symptoms,
    });
  }

  const total = results.reduce((s, r) => s + r.unnormalized, 0) || 1e-10;
  for (const r of results) r.probability = r.unnormalized / total;
  results.sort((a, b) => b.probability - a.probability);
  return results;
}

// ── Emergency rules check ─────────────────────────────────────────────────────
function checkEmergencyRules(symptoms, ctx) {
  const symSet = new Set(symptoms.map(s => s.toLowerCase().replace(/ /g, '_')));
  const age   = ctx.age   ?? 30;
  const flags = ctx.flags ?? [];

  const flagMap = {
    'Difficulty breathing':       'shortness_of_breath',
    'Chest tightness':            'chest_tightness',
    'Confusion / disorientation': 'confusion',
    'Sudden severe headache':     'severe_headache',
    'Loss of consciousness':      'loss_of_consciousness',
    'Uncontrolled bleeding':      'uncontrolled_bleeding',
    'High fever > 39°C':          'fever',
    'Rapid heart rate':           'palpitations',
  };
  for (const flag of flags) {
    if (flagMap[flag]) symSet.add(flagMap[flag]);
  }

  for (const rule of EMERGENCY_RULES) {
    const required = new Set(rule.symptoms);
    if (rule.age_max && age > rule.age_max) continue;
    const triggered = rule.match === 'all'
      ? [...required].every(s => symSet.has(s))
      : [...required].some(s => symSet.has(s));
    if (triggered) return rule.message;
  }

  if ((ctx.pain_level ?? 0) >= 9)
    return 'Extreme pain level reported — immediate medical evaluation recommended.';

  return null;
}

// ── Explanations ──────────────────────────────────────────────────────────────
function buildExplanations(symptoms, results, ctx) {
  const symNorm = symptoms.map(s => s.toLowerCase().replace(/ /g, '_'));
  const explanations = [];
  const top = results[0];

  if (top) {
    const matched = symNorm.filter(s => s in top.diseaseSymptoms);
    if (matched.length > 0) {
      explanations.push(
        `${top.disease} ranked highest: ${matched.slice(0, 3).join(', ')} ` +
        `match this condition's symptom profile (${matched.length} hit${matched.length > 1 ? 's' : ''}).`
      );
    }
  }

  for (const sym of symNorm.slice(0, 4)) {
    const supporting = results.slice(0, 4)
      .filter(r => (r.diseaseSymptoms[sym] ?? 0) > 0.7)
      .map(r => r.disease);
    if (supporting.length > 0) {
      explanations.push(
        `'${sym.replace(/_/g, ' ')}' strongly supports: ${supporting.slice(0, 2).join(', ')}.`
      );
    }
  }

  if (results.length > 1) {
    const r2 = results[1];
    explanations.push(
      `${r2.disease} (${(r2.probability * 100).toFixed(1)}%) also considered due to overlapping symptoms.`
    );
  }

  return explanations.slice(0, 5);
}

// ── Context notes ─────────────────────────────────────────────────────────────
function buildContextNotes(ctx, results) {
  const notes = [];
  const { age = 30, sex = '', chronic = [], duration = '', pain_level = 0 } = ctx;

  if (age < 5)   notes.push(`Age ${age} (infant): elevated risk weighting for respiratory and GI conditions.`);
  if (age > 60)  notes.push(`Age ${age} (elderly): increased prior for pneumonia, cardiac, and degenerative conditions.`);

  if (sex === 'Female' && results.slice(0, 3).some(r => r.disease.includes('UTI')))
    notes.push('Biological sex (Female): UTI prior probability doubled.');
  if (sex === 'Female' && results.slice(0, 3).some(r => r.disease.includes('PCOS')))
    notes.push('Biological sex (Female): PCOS prior applied.');

  for (const c of chronic) {
    if (c !== 'None') notes.push(`Chronic condition '${c}' factored into disease priors.`);
  }

  if (duration.includes('> 1 week') || duration.includes('1 week'))
    notes.push('Duration > 1 week: acute viral conditions deprioritised; chronic conditions weighted up.');

  if (pain_level >= 7)
    notes.push(`High pain level (${pain_level}/10): urgency threshold lowered.`);

  return notes;
}

// ── Main analyze function ─────────────────────────────────────────────────────
export function analyze(symptoms, patientCtx) {
  const diagnoses = computePosterior(symptoms, patientCtx);
  const emergencyOverride = checkEmergencyRules(symptoms, patientCtx);

  let riskLevel;
  if (emergencyOverride) {
    riskLevel = 'Emergency';
  } else {
    const topUrgencies = diagnoses.slice(0, 3).map(d => d.urgency);
    if (topUrgencies.includes('Emergency') && diagnoses[0].probability > 0.20) {
      riskLevel = 'Emergency';
    } else if (topUrgencies.includes('Emergency') || topUrgencies.includes('Moderate')) {
      riskLevel = 'Moderate';
    } else {
      riskLevel = 'Low';
    }
    if ((patientCtx.pain_level ?? 0) >= 8 && riskLevel === 'Low')  riskLevel = 'Moderate';
    if ((patientCtx.duration ?? '').includes('> 1 week') && riskLevel === 'Low') riskLevel = 'Moderate';
  }

  let confidence = 0;
  if (diagnoses.length >= 2) {
    const p1 = diagnoses[0].probability;
    const p2 = diagnoses[1].probability;
    confidence = Math.min(0.95, p1 / (p1 + p2 + 0.01));
  } else if (diagnoses.length === 1) {
    confidence = diagnoses[0].probability;
  }

  return {
    diagnoses,
    riskLevel,
    confidence,
    emergencyOverride,
    explanations: buildExplanations(symptoms, diagnoses, patientCtx),
    contextNotes: buildContextNotes(patientCtx, diagnoses),
    source: 'browser',
  };
}