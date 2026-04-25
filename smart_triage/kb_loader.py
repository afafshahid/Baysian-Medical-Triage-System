"""
SmartTriage AI – Knowledge Base Loader
Handles runtime expansion of the disease KB from CSV or JSON files.

CSV Format (one disease per row):
  disease,urgency,prior,description,symptom1,prob1,symptom2,prob2,...

JSON Format:
  {
    "Disease Name": {
      "prior": 0.05,
      "urgency": "Moderate",
      "description": "...",
      "symptoms": { "fever": 0.8, "cough": 0.7 }
    }
  }
"""

import csv, io, json
from typing import Tuple, List


VALID_URGENCIES = {"Emergency", "Moderate", "Low"}


class KnowledgeBaseLoader:
    def __init__(self, kb: dict):
        self.kb = kb  # reference to the runtime KB dict

    # ── Validate a single disease entry ──────────────────────────────────────
    def _validate(self, name: str, data: dict) -> str | None:
        """Returns error string or None if valid."""
        if not name or not name.strip():
            return "Disease name cannot be empty"
        if "symptoms" not in data or not data["symptoms"]:
            return f"'{name}': must have at least one symptom"
        if "prior" not in data:
            return f"'{name}': missing 'prior'"
        try:
            p = float(data["prior"])
            if not (0 < p <= 1):
                return f"'{name}': prior must be between 0 and 1"
        except (ValueError, TypeError):
            return f"'{name}': prior must be a number"
        if data.get("urgency") not in VALID_URGENCIES:
            return f"'{name}': urgency must be one of {VALID_URGENCIES}"
        for sym, prob in data["symptoms"].items():
            try:
                pv = float(prob)
                if not (0 <= pv <= 1):
                    return f"'{name}': symptom '{sym}' probability must be 0-1"
            except (ValueError, TypeError):
                return f"'{name}': symptom '{sym}' probability must be a number"
        return None

    # ── Load from dict (used by JSON upload) ─────────────────────────────────
    def load_from_dict(self, data: dict) -> Tuple[List[str], List[str]]:
        added, errors = [], []
        for name, disease_data in data.items():
            err = self._validate(name, disease_data)
            if err:
                errors.append(err)
                continue
            self.kb[name] = {
                "prior": float(disease_data["prior"]),
                "urgency": disease_data["urgency"],
                "description": disease_data.get("description", ""),
                "symptoms": {k: float(v) for k, v in disease_data["symptoms"].items()},
            }
            added.append(name)
        return added, errors

    # ── Load from CSV string ──────────────────────────────────────────────────
    def load_from_csv_string(self, content: str) -> Tuple[List[str], List[str]]:
        """
        Parses CSV with flexible column layout:
        disease | urgency | prior | description | sym1 | prob1 | sym2 | prob2 | ...
        """
        added, errors = [], []
        reader = csv.reader(io.StringIO(content))
        headers = None

        for i, row in enumerate(reader):
            # Skip empty rows
            if not any(cell.strip() for cell in row):
                continue
            # First non-empty row is header if it starts with 'disease'
            if headers is None:
                if row[0].strip().lower() == "disease":
                    headers = [h.strip().lower() for h in row]
                    continue
                else:
                    # No header row — assume positional
                    headers = []

            row = [cell.strip() for cell in row]
            if len(row) < 5:
                errors.append(f"Row {i+1}: too few columns (minimum: disease,urgency,prior,description,sym,prob)")
                continue

            try:
                name = row[0]
                urgency = row[1]
                prior = float(row[2])
                description = row[3]

                # Parse remaining pairs: sym1, prob1, sym2, prob2, ...
                symptoms = {}
                rest = row[4:]
                if len(rest) % 2 != 0:
                    rest = rest[:-1]  # drop trailing odd column
                for j in range(0, len(rest), 2):
                    sym = rest[j].strip().lower().replace(" ", "_")
                    if sym and rest[j+1]:
                        symptoms[sym] = float(rest[j+1])

                data = {"prior": prior, "urgency": urgency,
                        "description": description, "symptoms": symptoms}
                err = self._validate(name, data)
                if err:
                    errors.append(f"Row {i+1}: {err}")
                    continue

                self.kb[name] = {
                    "prior": prior, "urgency": urgency,
                    "description": description, "symptoms": symptoms,
                }
                added.append(name)

            except (ValueError, IndexError) as e:
                errors.append(f"Row {i+1}: parse error — {e}")

        return added, errors

    # ── Load from file path ───────────────────────────────────────────────────
    def load_from_file(self, filepath: str) -> Tuple[List[str], List[str]]:
        """Auto-detects JSON or CSV by file extension."""
        if filepath.endswith(".json"):
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            return self.load_from_dict(data)
        elif filepath.endswith(".csv"):
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
            return self.load_from_csv_string(content)
        else:
            return [], [f"Unsupported file type: {filepath}"]
