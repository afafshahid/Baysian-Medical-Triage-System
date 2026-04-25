"""
SmartTriage AI – Bayesian Inference Engine
Computes P(Disease | Symptoms) using Bayes' theorem with patient context personalization.
"""

import numpy as np
from typing import List, Dict, Any


class BayesianTriage:
    def __init__(self, diseases: dict, emergency_rules: list):
        self.diseases = diseases
        self.emergency_rules = emergency_rules

    # ─── Core Bayesian Computation ────────────────────────────────────────────
    def _compute_posterior(self, symptoms: List[str], patient_ctx: dict) -> List[dict]:
        results = []
        for disease_name, disease_data in self.diseases.items():
            prior = disease_data["prior"]
            prior = self._adjust_prior(prior, disease_name, patient_ctx)

            # Compute P(S|D) — joint likelihood of all symptoms given disease
            likelihood = 1.0
            for symptom in symptoms:
                sym_norm = symptom.lower().replace(" ", "_")
                p_sym_given_disease = disease_data["symptoms"].get(sym_norm, 0.02)
                likelihood *= p_sym_given_disease

            # Also penalize for absence of highly expected symptoms for this disease
            # (soft constraint)
            all_disease_syms = disease_data["symptoms"]
            for key_sym, key_prob in all_disease_syms.items():
                if key_sym not in [s.lower().replace(" ", "_") for s in symptoms]:
                    if key_prob > 0.80:
                        likelihood *= (1 - key_prob * 0.5)  # gentle penalty

            # Unnormalized posterior
            unnorm = likelihood * prior
            results.append({
                "disease": disease_name,
                "unnormalized": unnorm,
                "prior": prior,
                "likelihood": likelihood,
                "urgency": disease_data["urgency"],
                "description": disease_data["description"],
            })

        # Normalize: P(D|S) = unnorm / sum(all unnorm)
        total = sum(r["unnormalized"] for r in results)
        if total == 0:
            total = 1e-10
        for r in results:
            r["probability"] = r["unnormalized"] / total

        results.sort(key=lambda x: x["probability"], reverse=True)
        return results

    # ─── Age & Context Prior Adjustments ─────────────────────────────────────
    def _adjust_prior(self, prior: float, disease_name: str, ctx: dict) -> float:
        age = ctx.get("age", 30)
        chronic = ctx.get("chronic", [])
        sex = ctx.get("sex", "")

        multiplier = 1.0

        # Age adjustments
        if age < 5:
            if "Common Cold" in disease_name or "Gastroenteritis" in disease_name:
                multiplier *= 1.8
            if "Influenza" in disease_name:
                multiplier *= 1.4
            if "Pneumonia" in disease_name:
                multiplier *= 1.6

        elif age > 60:
            if "Pneumonia" in disease_name:
                multiplier *= 2.0
            if "Hypertensive" in disease_name:
                multiplier *= 2.5
            if "Anemia" in disease_name:
                multiplier *= 1.5
            if "Common Cold" in disease_name:
                multiplier *= 0.7

        elif 15 <= age <= 40:
            if "Migraine" in disease_name:
                multiplier *= 1.5
            if "UTI" in disease_name and sex == "Female":
                multiplier *= 2.0

        # Chronic condition adjustments
        if "Diabetes" in chronic:
            if "UTI" in disease_name:
                multiplier *= 1.6
            if "Anemia" in disease_name:
                multiplier *= 1.3

        if "Heart Disease" in chronic:
            if "Hypertensive" in disease_name:
                multiplier *= 2.0

        if "Asthma" in chronic:
            if "Asthma" in disease_name:
                multiplier *= 3.0

        if "Immunocompromised" in chronic:
            if "Pneumonia" in disease_name:
                multiplier *= 2.5
            if "COVID" in disease_name:
                multiplier *= 1.8

        if "Pregnancy" in chronic:
            if "UTI" in disease_name:
                multiplier *= 1.8
            if "Anemia" in disease_name:
                multiplier *= 1.6

        # Symptom duration adjustments
        duration = ctx.get("duration", "")
        if "> 1 week" in duration:
            if "Common Cold" in disease_name:
                multiplier *= 0.5  # Cold shouldn't last >1 week
            if "Typhoid" in disease_name or "Anemia" in disease_name:
                multiplier *= 1.5

        return min(prior * multiplier, 1.0)

    # ─── Emergency Rule Check ─────────────────────────────────────────────────
    def _check_emergency_rules(self, symptoms: List[str], patient_ctx: dict):
        sym_set = set(s.lower().replace(" ", "_") for s in symptoms)
        age = patient_ctx.get("age", 30)
        flags = patient_ctx.get("flags", [])

        # Add flag-based symptoms
        flag_map = {
            "Difficulty breathing": "shortness_of_breath",
            "Chest tightness": "chest_tightness",
            "Confusion / disorientation": "confusion",
            "Sudden severe headache": "severe_headache",
            "Loss of consciousness": "loss_of_consciousness",
            "Uncontrolled bleeding": "uncontrolled_bleeding",
            "High fever > 39°C": "fever",
            "Rapid heart rate": "palpitations",
        }
        for flag in flags:
            if flag in flag_map:
                sym_set.add(flag_map[flag])

        for rule in self.emergency_rules:
            required = set(rule["symptoms"])
            age_max = rule.get("age_max", None)
            match_type = rule.get("match", "all")

            if age_max and age > age_max:
                continue

            if match_type == "all":
                triggered = required.issubset(sym_set)
            else:  # any
                triggered = bool(required & sym_set)

            if triggered:
                return rule["message"]

        # Pain level check
        if patient_ctx.get("pain_level", 0) >= 9:
            return "Extreme pain level reported — immediate medical evaluation recommended."

        return None

    # ─── Build Explanations ───────────────────────────────────────────────────
    def _build_explanations(self, symptoms: List[str], results: List[dict],
                            patient_ctx: dict) -> List[str]:
        explanations = []
        sym_norm = [s.lower().replace(" ", "_") for s in symptoms]
        top = results[0] if results else None

        if top:
            matched = [s for s in sym_norm if s in top["disease_symptoms"]]
            if matched:
                explanations.append(
                    f"{top['disease']} was ranked highest because {len(matched)} of your symptoms "
                    f"({', '.join(matched[:3])}) have high likelihood for this condition."
                )

        for sym in sym_norm[:4]:
            matching = [
                r["disease"] for r in results[:4]
                if sym in r.get("disease_symptoms", {}) and r["disease_symptoms"].get(sym, 0) > 0.7
            ]
            if matching:
                explanations.append(
                    f"Symptom '{sym.replace('_', ' ')}' strongly supports: {', '.join(matching[:2])}."
                )

        if len(results) > 1:
            r2 = results[1]
            explanations.append(
                f"{r2['disease']} ({r2['probability']*100:.1f}%) is also considered due to overlapping symptoms."
            )

        return explanations[:5]

    def _build_context_notes(self, patient_ctx: dict, results: List[dict]) -> List[str]:
        notes = []
        age = patient_ctx.get("age", 30)
        sex = patient_ctx.get("sex", "")
        chronic = patient_ctx.get("chronic", [])
        duration = patient_ctx.get("duration", "")

        if age < 5:
            notes.append(f"Age {age} (infant/toddler): higher risk weighting applied for respiratory and GI conditions.")
        elif age > 60:
            notes.append(f"Age {age} (elderly): increased prior probability for pneumonia, hypertension, and anemia.")

        if sex == "Female" and "UTI" in [r["disease"] for r in results[:3]]:
            notes.append("Biological sex (Female): UTI prior probability doubled based on epidemiological data.")

        for c in chronic:
            if c != "None":
                notes.append(f"Chronic condition '{c}' was factored into disease probabilities.")

        if "> 1 week" in duration:
            notes.append("Symptom duration > 1 week: acute viral conditions (cold) deprioritized.")

        pain = patient_ctx.get("pain_level", 0)
        if pain >= 7:
            notes.append(f"High pain level ({pain}/10) noted — increases urgency classification threshold.")

        return notes

    # ─── Main Analysis Entry Point ────────────────────────────────────────────
    def analyze(self, symptoms: List[str], patient_ctx: dict) -> Dict[str, Any]:
        # Add disease_symptoms to results for explanation
        diagnoses = self._compute_posterior(symptoms, patient_ctx)
        for d in diagnoses:
            disease_name = d["disease"]
            d["disease_symptoms"] = self.diseases[disease_name]["symptoms"]

        # Emergency check
        emergency_override = self._check_emergency_rules(symptoms, patient_ctx)

        # Determine overall risk level
        if emergency_override:
            risk_level = "Emergency"
        else:
            top_urgencies = [d["urgency"] for d in diagnoses[:3]]
            if "Emergency" in top_urgencies and diagnoses[0]["probability"] > 0.20:
                risk_level = "Emergency"
            elif "Emergency" in top_urgencies or "Moderate" in top_urgencies:
                risk_level = "Moderate"
            else:
                risk_level = "Low"

            # Pain override
            if patient_ctx.get("pain_level", 0) >= 8:
                if risk_level == "Low":
                    risk_level = "Moderate"

            # Duration override
            if "> 1 week" in patient_ctx.get("duration", "") and risk_level == "Low":
                risk_level = "Moderate"

        # Confidence: based on top probability gap
        if len(diagnoses) >= 2:
            top_p = diagnoses[0]["probability"]
            second_p = diagnoses[1]["probability"]
            confidence = min(0.95, top_p / (top_p + second_p + 0.01))
        else:
            confidence = diagnoses[0]["probability"] if diagnoses else 0.0

        # Explanations
        explanations = self._build_explanations(symptoms, diagnoses, patient_ctx)
        context_notes = self._build_context_notes(patient_ctx, diagnoses)

        return {
            "diagnoses": diagnoses,
            "risk_level": risk_level,
            "confidence": confidence,
            "emergency_override": emergency_override,
            "explanations": explanations,
            "context_notes": context_notes,
        }