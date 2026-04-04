from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Optional

# ---------------------------------------------------------------------------
# Optional heavy NLP – import gracefully so the validator runs on all envs
# ---------------------------------------------------------------------------
_SPACY_NLP = None
_SPACY_MODEL_NAME: Optional[str] = None
_SENTENCE_MODEL = None
_SENTENCE_MODEL_NAME: Optional[str] = None


def _initialise_spacy() -> tuple[Optional[object], Optional[str]]:
    """
    Load the best available NLP model, preferring biomedical models.

    Priority:
      1) en_ner_bc5cdr_md  (scispaCy disease/chemical NER)
      2) en_core_sci_sm    (scispaCy scientific language model)
      3) en_core_web_sm    (general fallback)
    """
    try:
        import spacy
    except Exception:
        return None, None

    for model_name in ("en_ner_bc5cdr_md", "en_core_sci_sm", "en_core_web_sm"):
        try:
            return spacy.load(model_name), model_name
        except Exception:
            continue
    return None, None


def _initialise_sentence_transformer() -> tuple[Optional[object], Optional[str]]:
    """Load sentence-transformer model for semantic similarity."""
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
    except Exception:
        return None, None

    model_name = "all-MiniLM-L6-v2"
    try:
        return SentenceTransformer(model_name), model_name
    except Exception:
        return None, None


_SPACY_NLP, _SPACY_MODEL_NAME = _initialise_spacy()
try:
    from sentence_transformers import util as st_util  # type: ignore
except Exception:
    st_util = None
_SENTENCE_MODEL, _SENTENCE_MODEL_NAME = _initialise_sentence_transformer()


# ---------------------------------------------------------------------------
# Required sections – mirror the 12-section structure in COMMON_PROMPT_MESSAGE
# ---------------------------------------------------------------------------
REQUIRED_SECTIONS: list[str] = [
    "QUICK CLINICAL DECISION",
    "Executive Summary",
    "Clinical Presentation",
    "Imaging Findings",
    "Differential Diagnosis",
    "Pathophysiology",
    "Tumor Grading",
    "Clinical Implications",
    "Management Plan",
    "Next Steps",
    "Educational Pearls",
    "References",
]

# ---------------------------------------------------------------------------
# Medical knowledge rules: per class → expected keywords + urgency levels
# ---------------------------------------------------------------------------
DIAGNOSIS_KNOWLEDGE: dict[str, dict] = {
    "glioma": {
        "synonyms": ["glioma", "glioblastoma", "gbm", "astrocytoma", "oligodendroglioma", "grade iv", "grade iii"],
        "required_terms": ["idh", "mgmt", "who grade", "resection", "temozolomide"],
        "acceptable_urgency": ["stat", "urgent"],
        "expected_grade_range": [2, 3, 4],
        "reference_summary": (
            "Gliomas are primary brain tumors of glial origin. High-grade variants "
            "(Grade III/IV) are aggressive and require urgent surgical/oncological workup. "
            "IDH mutation and MGMT promoter methylation are critical prognostic markers."
        ),
    },
    "meningioma": {
        "synonyms": ["meningioma", "meningeal", "extra-axial", "dural tail"],
        "required_terms": ["dural", "resection", "grade", "surgery", "seizure"],
        "acceptable_urgency": ["stat", "urgent", "routine"],
        "expected_grade_range": [1, 2, 3],
        "reference_summary": (
            "Meningiomas arise from meningothelial cells. Most are WHO Grade I (benign) "
            "but Grade II/III variants recur. Dural tail sign on MRI is characteristic. "
            "Management depends on size, location, and symptoms."
        ),
    },
    "pituitary": {
        "synonyms": ["pituitary", "adenoma", "pituitary adenoma", "macroadenoma", "microadenoma", "sellar"],
        "required_terms": ["hormone", "vision", "sellar", "prolactin", "cortisol"],
        "acceptable_urgency": ["stat", "urgent", "routine"],
        "expected_grade_range": [1],
        "reference_summary": (
            "Pituitary adenomas are commonly benign sellar lesions causing hormonal "
            "dysfunction and visual field defects via optic chiasm compression. "
            "Medical, surgical, and radiation options depend on functional status and size."
        ),
    },
    "notumor": {
        "synonyms": ["no tumor", "notumor", "normal", "no evidence", "no mass", "benign"],
        "required_terms": ["normal", "no evidence", "follow-up", "routine"],
        "acceptable_urgency": ["routine"],
        "expected_grade_range": [],
        "reference_summary": (
            "No tumor detected. Report should confirm absence of mass lesion, "
            "absence of abnormal enhancement, and recommend routine follow-up if clinically indicated."
        ),
    },
}

# Urgency hierarchy for comparison (lower index = higher urgency)
_URGENCY_RANK: dict[str, int] = {"stat": 0, "urgent": 1, "routine": 2}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class SectionResult:
    name: str
    present: bool
    min_length_ok: bool           # section has at least 20 chars of content
    char_count: int = 0


@dataclass
class ValidationIssue:
    tier: str                     # e.g. "Tier1-Structure"
    severity: str                 # "ERROR" | "WARNING" | "INFO"
    message: str


@dataclass
class ValidationReport:
    timestamp: str
    predicted_class: str
    model_confidence: float
    model_name: str
    tier1_structure_score: float             # 0.0 – 1.0
    tier2_diagnostic_consistent: bool
    tier3_medical_plausibility_score: float  # 0.0 – 1.0
    tier4_confidence_aligned: bool
    overall_score: float                     # weighted average 0.0 – 1.0
    overall_pass: bool
    sections: list[SectionResult] = field(default_factory=list)
    issues: list[ValidationIssue] = field(default_factory=list)
    extracted_diagnosis: Optional[str] = None
    extracted_urgency: Optional[str] = None
    extracted_confidence_text: Optional[str] = None
    semantic_similarity: Optional[float] = None   # filled if sentence-transformers available
    spacy_entities: list[str] = field(default_factory=list)
    spacy_model_name: Optional[str] = None
    sentence_model_name: Optional[str] = None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["sections"] = [asdict(s) for s in self.sections]
        d["issues"] = [asdict(i) for i in self.issues]
        return d


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _normalise(text: str) -> str:
    """Lowercase + collapse whitespace."""
    return re.sub(r'\s+', ' ', text.lower()).strip()


def _extract_section_content(text: str, section_name: str) -> str:
    """
    Extract the text under a given section heading.

    Handles two common heading styles produced by the LLM prompt:
      - Plain     : ## Executive Summary
      - Numbered  : ## 1. Executive Summary   (or ## 2.1. Executive Summary)

    Captures content until the next heading or end-of-string.
    """
    # Use string concatenation to avoid f-string treating {1,3} as a tuple
    heading_pattern = re.compile(
        r'^#{1,3}\s+(?:\d+[\.\d]*\s+)?' + re.escape(section_name),
        re.IGNORECASE,
    )
    next_heading_pattern = re.compile(r'^#{1,3}\s')

    lines = text.splitlines()
    start_idx: Optional[int] = None
    for idx, line in enumerate(lines):
        if heading_pattern.match(line.strip()):
            start_idx = idx
            break

    if start_idx is None:
        return ""

    content_lines: list[str] = []
    for line in lines[start_idx + 1:]:
        if next_heading_pattern.match(line):
            break
        content_lines.append(line)

    return "\n".join(content_lines).strip()


def _check_sections(text: str) -> list[SectionResult]:
    results: list[SectionResult] = []
    for section in REQUIRED_SECTIONS:
        content = _extract_section_content(text, section)
        present = bool(content) or section.lower() in _normalise(text)
        char_count = len(content.strip())
        results.append(SectionResult(
            name=section,
            present=present,
            min_length_ok=char_count >= 20,
            char_count=char_count,
        ))
    return results


def _extract_urgency(text: str) -> Optional[str]:
    """Pull the urgency token from the QUICK CLINICAL DECISION section."""
    section = _extract_section_content(text, "QUICK CLINICAL DECISION")
    match = re.search(r'\*\*Urgency:\*?\*?\s*\[?(STAT|Urgent|Routine)\]?', section, re.IGNORECASE)
    if match:
        return match.group(1).lower()
    # Fallback: scan for urgency anywhere in text
    for token in ("stat", "urgent", "routine"):
        if re.search(rf'\b{token}\b', text, re.IGNORECASE):
            return token
    return None


def _extract_confidence_text(text: str) -> Optional[str]:
    """Extract the confidence level label (High / Medium / Low) from LLM text."""
    patterns = [
        r'\*\*Confidence:\*?\*?\s*\[?(High|Medium|Low)[^\]]*\]?',
        r'Confidence[:\s]+\[?(High|Medium|Low)',
        r'(High|Medium|Low)\s+(?:confidence|certainty)',
    ]
    for pat in patterns:
        match = re.search(pat, text, re.IGNORECASE)
        if match:
            return match.group(1).lower()
    return None


def _confidence_text_to_range(label: Optional[str]) -> tuple[float, float]:
    """Map LLM confidence label to numeric range."""
    mapping = {
        "high":   (0.90, 1.00),
        "medium": (0.70, 0.90),
        "low":    (0.00, 0.70),
    }
    return mapping.get(label or "", (0.0, 1.0))


def _find_diagnosis_in_text(text: str, predicted_class: str) -> Optional[str]:
    """Return the first matching diagnosis synonym found in the LLM text."""
    knowledge = DIAGNOSIS_KNOWLEDGE.get(predicted_class, {})
    synonyms = knowledge.get("synonyms", [predicted_class])
    norm = _normalise(text)
    for syn in synonyms:
        if syn.lower() in norm:
            return syn
    return None


def _check_required_medical_terms(text: str, predicted_class: str) -> tuple[float, list[str]]:
    """
    Returns score (0-1) and list of missing required terms.
    """
    knowledge = DIAGNOSIS_KNOWLEDGE.get(predicted_class, {})
    required = knowledge.get("required_terms", [])
    if not required:
        return 1.0, []
    norm = _normalise(text)
    missing = [t for t in required if t.lower() not in norm]
    score = (len(required) - len(missing)) / len(required)
    return score, missing


def _extract_spacy_entities(text: str) -> list[str]:
    """Run spaCy/scispaCy NER if available and return unique entity strings."""
    if _SPACY_NLP is None:
        return []
    doc = _SPACY_NLP(text[:10000])  # limit tokens for speed
    # Include both biomedical and general labels across model variants.
    allowed_labels = {
        "DISEASE",
        "CHEMICAL",
        "CANCER",
        "DISEASE_OR_SYNDROME",
        "SIGN_OR_SYMPTOM",
        "ANATOMICAL_SYSTEM",
        "ORG",
        "GPE",
        "PERSON",
        "NORP",
    }
    return sorted({ent.text for ent in doc.ents if ent.label_ in allowed_labels})


def _semantic_similarity(text: str, predicted_class: str) -> Optional[float]:
    """
    Cosine similarity between LLM text embedding and a reference summary
    for the predicted class.  Requires sentence-transformers.
    """
    if _SENTENCE_MODEL is None or st_util is None:
        return None
    knowledge = DIAGNOSIS_KNOWLEDGE.get(predicted_class, {})
    reference = knowledge.get("reference_summary", "")
    if not reference:
        return None
    emb_text = _SENTENCE_MODEL.encode(text[:2000], convert_to_tensor=True)
    emb_ref = _SENTENCE_MODEL.encode(reference, convert_to_tensor=True)
    sim = float(st_util.cos_sim(emb_text, emb_ref))
    return round(sim, 4)


# ---------------------------------------------------------------------------
# Main validation function
# ---------------------------------------------------------------------------
def validate_llm_report( llm_text: str, predicted_class: str, model_confidence: float, model_name: str = "unknown",pass_threshold: float = 0.65,) -> ValidationReport:
    """
    Validate one LLM-generated clinical report.

    Parameters
    ----------
    llm_text        : full text of the LLM response
    predicted_class : one of 'glioma', 'meningioma', 'pituitary', 'notumor'
    model_confidence: numeric confidence from the DL model (0.0 – 1.0)
    model_name      : name of the DL model used for prediction
    pass_threshold  : minimum overall_score to mark the report as passing
    """
    issues: list[ValidationIssue] = []
    predicted_class = predicted_class.lower().strip()

    # ---------------------------------------------------------
    # Tier 1 – Structural validation
    # ---------------------------------------------------------
    sections = _check_sections(llm_text)
    present_count = sum(1 for s in sections if s.present)
    filled_count = sum(1 for s in sections if s.present and s.min_length_ok)
    tier1_score = filled_count / len(REQUIRED_SECTIONS)

    for s in sections:
        if not s.present:
            issues.append(ValidationIssue(
                tier="Tier1-Structure",
                severity="ERROR",
                message=f"Required section missing: '{s.name}'"
            ))
        elif not s.min_length_ok:
            issues.append(ValidationIssue(
                tier="Tier1-Structure",
                severity="WARNING",
                message=f"Section '{s.name}' is present but very short ({s.char_count} chars) – may be a placeholder"
            ))

    # ---------------------------------------------------------
    # Tier 2 – Diagnostic consistency
    # ---------------------------------------------------------
    extracted_diagnosis = _find_diagnosis_in_text(llm_text, predicted_class)
    extracted_urgency = _extract_urgency(llm_text)

    tier2_consistent = extracted_diagnosis is not None

    if not tier2_consistent:
        issues.append(ValidationIssue(
            tier="Tier2-Diagnostic",
            severity="ERROR",
            message=(
                f"DL model predicted '{predicted_class}' but none of the expected "
                f"synonyms were found in the LLM report. "
                f"Synonyms checked: {DIAGNOSIS_KNOWLEDGE.get(predicted_class, {}).get('synonyms', [])}"
            )
        ))

    # Urgency check
    knowledge = DIAGNOSIS_KNOWLEDGE.get(predicted_class, {})
    acceptable_urgency = knowledge.get("acceptable_urgency", [])

    if extracted_urgency is None:
        issues.append(ValidationIssue(
            tier="Tier2-Diagnostic",
            severity="WARNING",
            message="Could not extract urgency level from QUICK CLINICAL DECISION section"
        ))
    elif extracted_urgency not in acceptable_urgency:
        issues.append(ValidationIssue(
            tier="Tier2-Diagnostic",
            severity="WARNING",
            message=(
                f"Urgency '{extracted_urgency}' is unusual for '{predicted_class}'. "
                f"Expected one of: {acceptable_urgency}"
            )
        ))

    # ---------------------------------------------------------
    # Tier 3 – Medical plausibility
    # ---------------------------------------------------------
    tier3_score, missing_terms = _check_required_medical_terms(llm_text, predicted_class)

    if missing_terms:
        issues.append(ValidationIssue(
            tier="Tier3-Medical",
            severity="WARNING",
            message=(
                f"Expected medical terms not found for '{predicted_class}': {missing_terms}"
            )
        ))

    spacy_entities = _extract_spacy_entities(llm_text)
    semantic_sim = _semantic_similarity(llm_text, predicted_class)

    if semantic_sim is not None and semantic_sim < 0.3:
        issues.append(ValidationIssue(
            tier="Tier3-Medical",
            severity="WARNING",
            message=(
                f"Semantic similarity to reference summary is low ({semantic_sim:.2f}). "
                f"Report may not be about the expected diagnosis."
            )
        ))

    # ---------------------------------------------------------
    # Tier 4 – Confidence alignment
    # ---------------------------------------------------------
    extracted_confidence_text = _extract_confidence_text(llm_text)
    low, high = _confidence_text_to_range(extracted_confidence_text)
    tier4_aligned = low <= model_confidence <= high

    if extracted_confidence_text is None:
        issues.append(ValidationIssue(
            tier="Tier4-Confidence",
            severity="INFO",
            message="Could not extract confidence label (High/Medium/Low) from LLM report"
        ))
    elif not tier4_aligned:
        issues.append(ValidationIssue(
            tier="Tier4-Confidence",
            severity="WARNING",
            message=(
                f"DL model confidence {model_confidence:.1%} ({model_confidence:.2f}) does not "
                f"fall within the range implied by the LLM label '{extracted_confidence_text}' "
                f"(expected {low:.0%}–{high:.0%})."
            )
        ))
    else:
        issues.append(ValidationIssue(
            tier="Tier4-Confidence",
            severity="INFO",
            message=(
                f"Confidence aligned: DL={model_confidence:.1%}, LLM label='{extracted_confidence_text}' "
                f"→ expected range {low:.0%}–{high:.0%}"
            )
        ))

    # ---------------------------------------------------------
    # Overall score  (weighted: structure 30 %, diagnostic 35 %, medical 25 %, conf 10 %)
    # ---------------------------------------------------------
    t2_score = 1.0 if tier2_consistent else 0.0
    t4_score = 1.0 if tier4_aligned else 0.5  # soft penalty only
    overall_score = round(
        0.30 * tier1_score +
        0.35 * t2_score +
        0.25 * tier3_score +
        0.10 * t4_score,
        4
    )
    overall_pass = overall_score >= pass_threshold

    if not overall_pass:
        issues.append(ValidationIssue(
            tier="Overall",
            severity="ERROR",
            message=(
                f"Report FAILED validation. Overall score {overall_score:.2f} is below "
                f"threshold {pass_threshold:.2f}."
            )
        ))
    else:
        issues.append(ValidationIssue(
            tier="Overall",
            severity="INFO",
            message=f"Report PASSED validation. Overall score: {overall_score:.2f}"
        ))

    return ValidationReport(
        timestamp=datetime.now().isoformat(),
        predicted_class=predicted_class,
        model_confidence=model_confidence,
        model_name=model_name,
        tier1_structure_score=round(tier1_score, 4),
        tier2_diagnostic_consistent=tier2_consistent,
        tier3_medical_plausibility_score=round(tier3_score, 4),
        tier4_confidence_aligned=tier4_aligned,
        overall_score=overall_score,
        overall_pass=overall_pass,
        sections=sections,
        issues=issues,
        extracted_diagnosis=extracted_diagnosis,
        extracted_urgency=extracted_urgency,
        extracted_confidence_text=extracted_confidence_text,
        semantic_similarity=semantic_sim,
        spacy_entities=spacy_entities,
        spacy_model_name=_SPACY_MODEL_NAME,
        sentence_model_name=_SENTENCE_MODEL_NAME,
    )


# ---------------------------------------------------------------------------
# Pretty-print helper
# ---------------------------------------------------------------------------
def print_report(report: ValidationReport) -> None:
    """Print a human-readable summary of the validation report."""
    bar = "=" * 60
    print(f"\n{bar}")
    print(f"  LLM OUTPUT VALIDATOR  —  {report.timestamp}")
    print(bar)
    print(f"  Model     : {report.model_name}")
    print(f"  Prediction: {report.predicted_class}  (DL confidence: {report.model_confidence:.1%})")
    print(f"  Diagnosis in LLM : {report.extracted_diagnosis or '-- NOT FOUND --'}")
    print(f"  Urgency in LLM   : {report.extracted_urgency or '-- NOT FOUND --'}")
    print(f"  Confidence label : {report.extracted_confidence_text or '-- NOT FOUND --'}")
    print(f"  spaCy model      : {report.spacy_model_name or '-- NOT LOADED --'}")
    print(f"  ST model         : {report.sentence_model_name or '-- NOT LOADED --'}")
    if report.semantic_similarity is not None:
        print(f"  Semantic sim.    : {report.semantic_similarity:.3f}")
    print(f"\n  Scores:")
    print(f"    Tier 1 Structure      : {report.tier1_structure_score:.0%}")
    print(f"    Tier 2 Diagnostic     : {'PASS' if report.tier2_diagnostic_consistent else 'FAIL'}")
    print(f"    Tier 3 Medical        : {report.tier3_medical_plausibility_score:.0%}")
    print(f"    Tier 4 Confidence     : {'ALIGNED' if report.tier4_confidence_aligned else 'MISMATCH'}")
    print(f"\n  OVERALL : {report.overall_score:.2f}  →  {'✓ PASS' if report.overall_pass else '✗ FAIL'}")

    print(f"\n  Sections ({len(report.sections)} checked):")
    for s in report.sections:
        status = "✓" if s.present and s.min_length_ok else ("~" if s.present else "✗")
        print(f"    [{status}] {s.name}  ({s.char_count} chars)")

    print(f"\n  Issues ({len(report.issues)}):")
    for iss in report.issues:
        icon = {"ERROR": "❌", "WARNING": "⚠️", "INFO": "ℹ️"}.get(iss.severity, " ")
        print(f"    {icon} [{iss.tier}] {iss.message}")

    print(bar + "\n")


# ---------------------------------------------------------------------------
# Save to JSON / CSV
# ---------------------------------------------------------------------------
def save_report(report: ValidationReport, output_dir: str) -> tuple[str, str]:
    """
    Save the validation report as both JSON and CSV.
    Returns (json_path, csv_path).
    """
    import csv

    os.makedirs(output_dir, exist_ok=True)
    ts = report.timestamp.replace(":", "-").replace(".", "-")
    safe_class = re.sub(r'\W+', '_', report.predicted_class)

    json_path = os.path.join(output_dir, f"llm_validation_{safe_class}_{ts}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report.to_dict(), f, indent=2, ensure_ascii=False)

    csv_path = os.path.join(output_dir, "llm_validation_summary.csv")
    write_header = not os.path.exists(csv_path)

    row = {
        "timestamp": report.timestamp,
        "model_name": report.model_name,
        "predicted_class": report.predicted_class,
        "model_confidence": report.model_confidence,
        "tier1_structure_score": report.tier1_structure_score,
        "tier2_diagnostic_consistent": report.tier2_diagnostic_consistent,
        "tier3_medical_plausibility_score": report.tier3_medical_plausibility_score,
        "tier4_confidence_aligned": report.tier4_confidence_aligned,
        "overall_score": report.overall_score,
        "overall_pass": report.overall_pass,
        "extracted_diagnosis": report.extracted_diagnosis,
        "extracted_urgency": report.extracted_urgency,
        "extracted_confidence_text": report.extracted_confidence_text,
        "semantic_similarity": report.semantic_similarity,
        "error_count": sum(1 for i in report.issues if i.severity == "ERROR"),
        "warning_count": sum(1 for i in report.issues if i.severity == "WARNING"),
    }

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    return json_path, csv_path


# ---------------------------------------------------------------------------
# Batch validate from stored ChromaDB / log folder
# ---------------------------------------------------------------------------
def batch_validate_from_folder( log_folder: str, output_dir: str, predicted_class_default: str = "glioma", model_confidence_default: float = 0.80,) -> list[ValidationReport]:
    """
    Validate all .json or .txt files in log_folder that contain LLM responses.

    Each file is expected to contain either:
      - Plain text LLM response, or
      - A JSON object with keys: "llm_text", "predicted_class", "model_confidence", "model_name"

    Results are appended to the shared CSV in output_dir.
    """
    reports: list[ValidationReport] = []

    if not os.path.isdir(log_folder):
        print(f"[batch_validate] Folder not found: {log_folder}")
        return reports

    for fname in sorted(os.listdir(log_folder)):
        if not fname.endswith((".json", ".txt")):
            continue

        fpath = os.path.join(log_folder, fname)
        try:
            with open(fpath, encoding="utf-8") as f:
                raw = f.read()
        except OSError as exc:
            print(f"[batch_validate] Cannot read {fname}: {exc}")
            continue

        # Try to parse as structured JSON first
        llm_text = raw
        predicted_class = predicted_class_default
        model_confidence = model_confidence_default
        model_name = "unknown"

        if fname.endswith(".json"):
            try:
                obj = json.loads(raw)
                llm_text = obj.get("llm_text", obj.get("response", raw))
                predicted_class = obj.get("predicted_class", predicted_class_default)
                model_confidence = float(obj.get("model_confidence", model_confidence_default))
                model_name = obj.get("model_name", "unknown")
            except json.JSONDecodeError:
                pass  # treat as plain text

        report = validate_llm_report(
            llm_text=llm_text,
            predicted_class=predicted_class,
            model_confidence=model_confidence,
            model_name=model_name,
        )
        save_report(report, output_dir)
        reports.append(report)
        status = "PASS" if report.overall_pass else "FAIL"
        print(f"  [{status}] {fname}  score={report.overall_score:.2f}")

    return reports


# ---------------------------------------------------------------------------
# Quick smoke-test  (run as  python llm_output_validator.py)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    LOG_FOLDER = os.path.join(PROJECT_ROOT, "logs")
    OUTPUT_DIR = os.path.join(PROJECT_ROOT, "llm_output_validate", "results")

    _DEMO_TEXT = """
## QUICK CLINICAL DECISION (For Radiologists - 30 seconds)
**Urgency:** Urgent
**Key Finding:** Right temporal high-grade glioma with significant perilesional oedema.
**Confidence:** High 91%
**Action Required:** Neurosurgical referral within 24 hours.
**Red Flags:** Mild midline shift noted; monitor for herniation.

## 1. Executive Summary
A 52-year-old male presents with a 6-week history of progressive headaches and word-finding
difficulty. MRI reveals a right temporal mass with ring enhancement consistent with a
high-grade glioma (GBM). Immediate neurosurgical evaluation is recommended.

## 2. Clinical Presentation
**Patient Profile:** 52M, right-handed.
**Presenting symptoms:** Word-finding difficulty, progressive headaches.
**Relevant history:** No prior malignancy.

## 3. Imaging Findings
Right temporal lobe mass 4.5 cm, heterogeneous ring-enhancement, moderate oedema, midline shift 3mm.
T2/FLAIR hyperintensity extending beyond visible tumour margins suggestive of infiltration.

## 4. Differential Diagnosis
| Diagnosis | Likelihood | Supporting Features | Against | Key Distinguisher |
|-----------|-----------|---------------------|---------|--------------------|
| GBM (Grade IV glioma) | 85% | Ring enhancement, necrosis, oedema | None | Classic appearance |
| Metastasis | 10% | Ring enhancement | Single lesion, no primary known | No systemic malignancy |
| Abscess | 5% | Ring enhancement | No fever | DWI restriction absent |

## 5. Pathophysiology
Glioblastoma (WHO Grade IV) arises from glial cells. IDH-wildtype GBM is the most
aggressive subtype. MGMT promoter methylation influences chemotherapy response.
Ring enhancement correlates with peripheral tumour viability around a necrotic core.

## 6. Tumor Grading & Classification
**WHO Grade:** IV
**Molecular Subtype:** IDH-wildtype, MGMT methylation status pending.
**Prognosis:** Median survival 14–16 months with Stupp protocol (temozolomide + RT).

## 7. Clinical Implications
Symptoms correlate with right temporal mass. Untreated GBM progresses rapidly leading to
cerebral herniation within weeks to months. Maximal safe surgical resection prolongs survival.

## 8. Management Plan
**IMMEDIATE:** Dexamethasone for oedema; neurosurgery referral.
**SHORT-TERM:** Maximal safe resection with intra-operative MRI guidance if available.
**LONG-TERM:** Stupp protocol (concurrent RT + temozolomide), followed by adjuvant temozolomide.
MRI every 3 months surveillance.

## 9. Next Steps
Refer neurosurgery STAT. MR spectroscopy and perfusion imaging recommended.
Tumour board presentation. Molecular testing for IDH and MGMT.

## 10. Educational Pearls
1. GBM ring enhancement = peripheral viable tumour + central necrosis.
2. IDH status determines prognosis and treatment eligibility.
3. MGMT methylation predicts benefit from temozolomide.

## 11. References
NCCN CNS Tumours v2.2024; Stupp et al. NEJM 2005; WHO CNS Classification 2021.

## 12. TECHNICAL APPENDIX
AI Confidence: 91%. Entropy: 0.12. Grad-CAM highlights right temporal core region.
"""

    report = validate_llm_report(
        llm_text=_DEMO_TEXT,
        predicted_class="glioma",
        model_confidence=0.91,
        model_name="propose_balanced",
    )
    print_report(report)
    json_p, csv_p = save_report(report, OUTPUT_DIR)
    print(f"Saved JSON : {json_p}")
    print(f"Saved CSV  : {csv_p}")

    # If there are real log files, batch validate them too
    if os.path.isdir(LOG_FOLDER) and any(
        fname.endswith((".json", ".txt")) for fname in os.listdir(LOG_FOLDER)
    ):
        print(f"\nBatch validating logs from: {LOG_FOLDER}")
        batch_validate_from_folder(LOG_FOLDER, OUTPUT_DIR)
