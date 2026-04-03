"""
DeepEval LLM Output Validator  (Ollama / local-model edition)
==============================================================
Validates LLM-generated brain tumor clinical reports using the DeepEval library
with a **local Ollama model** as the LLM judge.  No API keys required.

Metrics applied:
  1. HallucinationMetric   – detects factual hallucinations vs. medical reference
  2. AnswerRelevancyMetric – checks if the report is relevant to the diagnosis query
  3. BiasMetric            – flags biased language in the clinical report
  4. GEval (Medical)       – custom criterion: medical terminology completeness

Reads  : llm_output_validate/llm_outputs.csv
Writes : llm_output_validate/deepeval_validation_report.txt
         llm_output_validate/results/deepeval_<class>_<timestamp>.json

Requirements
------------
  pip install deepeval requests

  # Ollama must be running locally:
  #   https://ollama.com/download
  ollama serve
  ollama pull llama3.2          # recommended judge model (or mistral, gemma2, etc.)

Usage
-----
  python llm_output_validate/deepeval_llm_validator.py
  python llm_output_validate/deepeval_llm_validator.py --ollama-model mistral
  python llm_output_validate/deepeval_llm_validator.py --max-rows 5
  python llm_output_validate/deepeval_llm_validator.py --ollama-url http://localhost:11434

Recommended judge models (by accuracy for medical text)
---------------------------------------------------------
  llama3.2        – good balance, fast on most machines  (default)
  mistral         – strong instruction following
  gemma2          – clean JSON output, reliable parsing
  qwen2.5         – strong reasoning, good for medical domain
  meditron        – fine-tuned on medical text (best for this project)
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import requests as _requests   # used by OllamaJudge and connectivity check

# ---------------------------------------------------------------------------
# DeepEval import – fail early with a clear install message
# ---------------------------------------------------------------------------
try:
    from deepeval.metrics import (
        AnswerRelevancyMetric,
        BiasMetric,
        GEval,
        HallucinationMetric,
    )
    from deepeval.models.base_model import DeepEvalBaseLLM
    from deepeval.test_case import LLMTestCase, LLMTestCaseParams
    DEEPEVAL_AVAILABLE = True
except ImportError:
    DEEPEVAL_AVAILABLE = False

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent   # llm_output_validate/
INPUT_CSV  = SCRIPT_DIR / "llm_outputs.csv"
RESULTS_DIR = SCRIPT_DIR / "results"
REPORT_TXT  = SCRIPT_DIR / "deepeval_validation_report.txt"

CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]

# ---------------------------------------------------------------------------
# Default settings
# ---------------------------------------------------------------------------
DEFAULT_OLLAMA_MODEL = "llama3.2"
DEFAULT_OLLAMA_URL   = "http://localhost:11434"

DEFAULT_HALLUCINATION_THRESHOLD = 0.5   # score < threshold  → PASS
DEFAULT_RELEVANCY_THRESHOLD     = 0.7   # score >= threshold → PASS
DEFAULT_BIAS_THRESHOLD          = 0.5   # score < threshold  → PASS
DEFAULT_GEVAL_THRESHOLD         = 0.6   # score >= threshold → PASS

# ---------------------------------------------------------------------------
# Medical reference knowledge  (context fed to HallucinationMetric)
# ---------------------------------------------------------------------------
DIAGNOSIS_KNOWLEDGE: dict[str, dict] = {
    "glioma": {
        "reference_context": (
            "Gliomas are primary brain tumors of glial origin. "
            "High-grade variants (WHO Grade III/IV) are aggressive and require urgent surgical "
            "and oncological workup. IDH mutation status and MGMT promoter methylation are "
            "critical prognostic molecular markers. Standard treatment follows the Stupp "
            "protocol: maximal safe surgical resection followed by concurrent radiotherapy "
            "and temozolomide chemotherapy. Glioblastoma (GBM, Grade IV, IDH-wildtype) has "
            "a median survival of 14-16 months with current treatment."
        ),
        "input_prompt": (
            "Generate a structured 12-section clinical radiology report for a brain MRI scan "
            "where the AI model has predicted GLIOMA. Include urgency, executive summary, "
            "clinical presentation, imaging findings, differential diagnosis, pathophysiology, "
            "tumor grading, clinical implications, management plan, next steps, educational "
            "pearls, and references."
        ),
    },
    "meningioma": {
        "reference_context": (
            "Meningiomas arise from meningothelial cells of the arachnoid layer. "
            "Most are WHO Grade I (benign) but Grade II and III variants recur. "
            "The dural tail sign on contrast MRI is characteristic. Management depends on "
            "size, location, symptoms, and WHO grade: observation, surgical resection, "
            "or stereotactic radiosurgery. Seizures may be a presenting symptom."
        ),
        "input_prompt": (
            "Generate a structured 12-section clinical radiology report for a brain MRI scan "
            "where the AI model has predicted MENINGIOMA. Include urgency, executive summary, "
            "clinical presentation, imaging findings, differential diagnosis, pathophysiology, "
            "tumor grading, clinical implications, management plan, next steps, educational "
            "pearls, and references."
        ),
    },
    "pituitary": {
        "reference_context": (
            "Pituitary adenomas are benign sellar lesions causing hormonal dysfunction "
            "(prolactin, cortisol, growth hormone excess or deficiency) and visual field "
            "defects via optic chiasm compression (bitemporal hemianopia). Classified as "
            "microadenoma (<10 mm) or macroadenoma (>=10 mm). Treatment options depend on "
            "functional status: medical therapy (dopamine agonists for prolactinomas), "
            "trans-sphenoidal surgery, or radiation."
        ),
        "input_prompt": (
            "Generate a structured 12-section clinical radiology report for a brain MRI scan "
            "where the AI model has predicted PITUITARY ADENOMA. Include urgency, executive "
            "summary, clinical presentation, imaging findings, differential diagnosis, "
            "pathophysiology, tumor grading, clinical implications, management plan, next "
            "steps, educational pearls, and references."
        ),
    },
    "notumor": {
        "reference_context": (
            "A normal brain MRI shows no focal mass lesion, no abnormal enhancement, "
            "no midline shift, and no evidence of intracranial neoplasm. "
            "Report should clearly confirm absence of tumor and absence of abnormal signal "
            "intensity, and recommend routine clinical follow-up if symptomatically indicated. "
            "No urgent neurosurgical referral is required."
        ),
        "input_prompt": (
            "Generate a structured 12-section clinical radiology report for a brain MRI scan "
            "where the AI model has predicted NO TUMOR (normal brain). Include urgency, "
            "executive summary, clinical presentation, imaging findings, differential "
            "diagnosis, pathophysiology, tumor grading (N/A), clinical implications, "
            "management plan, next steps, educational pearls, and references."
        ),
    },
}


# ---------------------------------------------------------------------------
# OllamaJudge  – wraps local Ollama as DeepEval's LLM judge
# ---------------------------------------------------------------------------

class OllamaJudge(DeepEvalBaseLLM):
    """
    Adapts a locally running Ollama model to DeepEval's LLM judge interface.

    DeepEval sends prompts expecting JSON-formatted scoring responses.
    Ollama's /api/generate endpoint is called with format='json' when
    a structured schema is requested to improve JSON compliance.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_OLLAMA_MODEL,
        base_url:   str = DEFAULT_OLLAMA_URL,
        timeout:    int = 180,
    ) -> None:
        self.model_name = model_name
        self.base_url   = base_url.rstrip("/")
        self.timeout    = timeout

    # DeepEval requires this method
    def load_model(self):
        return self.model_name

    def generate(self, prompt: str, schema=None) -> str:
        """
        Call Ollama synchronously.
        schema is a Pydantic BaseModel class when DeepEval needs structured JSON.
        """
        payload: dict = {
            "model":  self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.0},
        }
        # Request JSON mode when schema is expected (improves local model compliance)
        if schema is not None:
            payload["format"] = "json"

        try:
            resp = _requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=self.timeout,
            )
            resp.raise_for_status()
            return resp.json().get("response", "")
        except _requests.exceptions.Timeout:
            raise RuntimeError(
                f"Ollama timed out after {self.timeout}s. "
                "Try a smaller model (e.g. --ollama-model gemma2) "
                "or increase --ollama-timeout."
            )
        except _requests.exceptions.ConnectionError:
            raise RuntimeError(
                f"Cannot connect to Ollama at {self.base_url}. "
                "Make sure Ollama is running: 'ollama serve'"
            )

    # DeepEval also calls the async variant; we delegate to sync here
    async def a_generate(self, prompt: str, schema=None) -> str:
        return self.generate(prompt, schema)

    def get_model_name(self) -> str:
        return f"ollama/{self.model_name}"


# ---------------------------------------------------------------------------
# Ollama connectivity helpers
# ---------------------------------------------------------------------------

def _check_ollama_running(base_url: str) -> bool:
    """Return True if Ollama server is reachable."""
    try:
        resp = _requests.get(f"{base_url}/api/tags", timeout=5)
        return resp.status_code == 200
    except Exception:
        return False


def _list_ollama_models(base_url: str) -> list[str]:
    """Return list of pulled model names from Ollama."""
    try:
        resp = _requests.get(f"{base_url}/api/tags", timeout=5)
        if resp.status_code == 200:
            return [m.get("name", "") for m in resp.json().get("models", [])]
    except Exception:
        pass
    return []


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class DeepEvalSampleResult:
    timestamp:          str
    image_filename:     str
    true_class:         str
    predicted_class:    str
    model_name:         str
    model_confidence:   float

    hallucination_score:  Optional[float] = None
    hallucination_pass:   Optional[bool]  = None
    hallucination_reason: Optional[str]   = None

    relevancy_score:    Optional[float] = None
    relevancy_pass:     Optional[bool]  = None
    relevancy_reason:   Optional[str]   = None

    bias_score:         Optional[float] = None
    bias_pass:          Optional[bool]  = None
    bias_reason:        Optional[str]   = None

    geval_score:        Optional[float] = None
    geval_pass:         Optional[bool]  = None
    geval_reason:       Optional[str]   = None

    metrics_run:        int            = 0
    metrics_passed:     int            = 0
    overall_pass:       bool           = False
    error:              Optional[str]  = None

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Build DeepEval metrics  (all wired to the local Ollama judge)
# ---------------------------------------------------------------------------

def _build_metrics(
    judge:                  OllamaJudge,
    hallucination_threshold: float,
    relevancy_threshold:     float,
    bias_threshold:          float,
    geval_threshold:         float,
) -> dict:
    return {
        "hallucination": HallucinationMetric(
            threshold=hallucination_threshold,
            model=judge,
            async_mode=False,
        ),
        "relevancy": AnswerRelevancyMetric(
            threshold=relevancy_threshold,
            model=judge,
            async_mode=False,
        ),
        "bias": BiasMetric(
            threshold=bias_threshold,
            model=judge,
            async_mode=False,
        ),
        "geval": GEval(
            name="Medical Terminology Completeness",
            criteria=(
                "Evaluate whether the clinical report uses correct and complete medical "
                "terminology appropriate for the diagnosed brain tumor type. "
                "Score 1.0 = fully appropriate medical language with all relevant terms. "
                "Score 0.0 = missing or incorrect medical terminology."
            ),
            evaluation_params=[
                LLMTestCaseParams.INPUT,
                LLMTestCaseParams.ACTUAL_OUTPUT,
            ],
            threshold=geval_threshold,
            model=judge,
            async_mode=False,
        ),
    }


# ---------------------------------------------------------------------------
# Validate one sample
# ---------------------------------------------------------------------------

def _run_metric(metric, test_case: LLMTestCase, name: str):
    """
    Run one DeepEval metric safely.
    Returns (score, pass, reason) or (None, None, error_message).
    Local models occasionally produce non-JSON output; we catch those errors
    and mark the metric as errored rather than crashing the whole benchmark.
    """
    try:
        metric.measure(test_case)
        return (
            round(metric.score, 4),
            metric.is_successful(),
            getattr(metric, "reason", None),
        )
    except json.JSONDecodeError as exc:
        return None, None, f"JSON parse error ({name}): {exc}"
    except Exception as exc:
        return None, None, f"Error ({name}): {exc}"


def _validate_one(row: dict, metrics: dict) -> DeepEvalSampleResult:
    true_class  = row.get("true_class",      "unknown").lower().strip()
    pred_class  = row.get("predicted_class", "unknown").lower().strip()
    model_name  = row.get("model_name",      "unknown")
    fname       = row.get("image_filename",  "unknown")
    llm_text    = row.get("llm_output",      "").strip()

    try:
        confidence = float(row.get("confidence", 0.0))
    except ValueError:
        confidence = 0.0

    result = DeepEvalSampleResult(
        timestamp=datetime.now().isoformat(),
        image_filename=fname,
        true_class=true_class,
        predicted_class=pred_class,
        model_name=model_name,
        model_confidence=confidence,
    )

    if not llm_text:
        result.error = "llm_output is empty – skipped"
        return result

    knowledge    = DIAGNOSIS_KNOWLEDGE.get(true_class, DIAGNOSIS_KNOWLEDGE["glioma"])
    input_prompt = knowledge["input_prompt"]
    ref_context  = knowledge["reference_context"]

    test_case = LLMTestCase(
        input=input_prompt,
        actual_output=llm_text,
        context=[ref_context],       # used by HallucinationMetric
        expected_output=ref_context, # used by AnswerRelevancyMetric
    )

    metrics_run = metrics_passed = 0

    # 1. Hallucination
    s, p, r = _run_metric(metrics["hallucination"], test_case, "hallucination")
    result.hallucination_score, result.hallucination_pass, result.hallucination_reason = s, p, r
    if s is not None:
        metrics_run += 1
        if p:
            metrics_passed += 1

    # 2. Answer Relevancy
    s, p, r = _run_metric(metrics["relevancy"], test_case, "relevancy")
    result.relevancy_score, result.relevancy_pass, result.relevancy_reason = s, p, r
    if s is not None:
        metrics_run += 1
        if p:
            metrics_passed += 1

    # 3. Bias
    s, p, r = _run_metric(metrics["bias"], test_case, "bias")
    result.bias_score, result.bias_pass, result.bias_reason = s, p, r
    if s is not None:
        metrics_run += 1
        if p:
            metrics_passed += 1

    # 4. GEval
    s, p, r = _run_metric(metrics["geval"], test_case, "geval")
    result.geval_score, result.geval_pass, result.geval_reason = s, p, r
    if s is not None:
        metrics_run += 1
        if p:
            metrics_passed += 1

    result.metrics_run    = metrics_run
    result.metrics_passed = metrics_passed
    result.overall_pass   = (metrics_run > 0) and (metrics_passed == metrics_run)
    return result


# ---------------------------------------------------------------------------
# Save individual JSON
# ---------------------------------------------------------------------------

def _save_json(sample: DeepEvalSampleResult) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts       = sample.timestamp.replace(":", "-").replace(".", "-")
    safe_cls = sample.true_class.replace(" ", "_")
    out_path = RESULTS_DIR / f"deepeval_{safe_cls}_{ts}.json"
    out_path.write_text(
        json.dumps(sample.to_dict(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def _avg(seq) -> float:
    clean = [v for v in seq if v is not None]
    return round(sum(clean) / len(clean), 4) if clean else 0.0

def _pct(bools) -> float:
    clean = [v for v in bools if v is not None]
    return round(sum(1 for v in clean if v) / len(clean), 4) if clean else 0.0

def _compute_stats(samples: list[DeepEvalSampleResult]) -> dict:
    valid = [s for s in samples if s.error is None and s.metrics_run > 0]
    n     = len(valid)
    stats: dict = {
        "total":   len(samples),
        "valid":   n,
        "skipped": len(samples) - n,
        "passed":  sum(1 for s in valid if s.overall_pass),
        "hallucination_pass_rate": _pct([s.hallucination_pass for s in valid]),
        "relevancy_pass_rate":     _pct([s.relevancy_pass     for s in valid]),
        "bias_pass_rate":          _pct([s.bias_pass          for s in valid]),
        "geval_pass_rate":         _pct([s.geval_pass         for s in valid]),
        "avg_hallucination_score": _avg([s.hallucination_score for s in valid]),
        "avg_relevancy_score":     _avg([s.relevancy_score     for s in valid]),
        "avg_bias_score":          _avg([s.bias_score          for s in valid]),
        "avg_geval_score":         _avg([s.geval_score         for s in valid]),
        "per_class": {},
    }
    for cls in CLASSES:
        grp = [s for s in valid if s.true_class == cls]
        if not grp:
            continue
        m = len(grp)
        stats["per_class"][cls] = {
            "n":       m,
            "passed":  sum(1 for s in grp if s.overall_pass),
            "hallucination_pass_rate": _pct([s.hallucination_pass for s in grp]),
            "relevancy_pass_rate":     _pct([s.relevancy_pass     for s in grp]),
            "bias_pass_rate":          _pct([s.bias_pass          for s in grp]),
            "geval_pass_rate":         _pct([s.geval_pass         for s in grp]),
            "avg_hallucination":       _avg([s.hallucination_score for s in grp]),
            "avg_relevancy":           _avg([s.relevancy_score     for s in grp]),
            "avg_bias":                _avg([s.bias_score          for s in grp]),
            "avg_geval":               _avg([s.geval_score         for s in grp]),
        }
    return stats


# ---------------------------------------------------------------------------
# Report writer
# ---------------------------------------------------------------------------

def _write_report(
    stats: dict,
    samples: list[DeepEvalSampleResult],
    thresholds: dict,
    judge_model: str,
) -> None:
    BAR  = "=" * 72
    THIN = "-" * 72
    lines: list[str] = []

    def w(s: str = "") -> None:
        lines.append(s)

    w(BAR)
    w("  DEEPEVAL LLM OUTPUT VALIDATION REPORT  (local Ollama judge)")
    w(f"  Generated  : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    w(f"  Input CSV  : llm_outputs.csv")
    w(f"  Judge model: {judge_model}  (running via Ollama — no API key required)")
    w(f"  Results    : llm_output_validate/results/deepeval_*.json")
    w(BAR)
    w()
    w("METHODOLOGY")
    w(THIN)
    w()
    w("  DeepEval evaluates clinical reports using a local Ollama model as judge.")
    w("  No internet connection or API key is required.")
    w()
    w("  Metric 1 — HallucinationMetric")
    w("  --------------------------------")
    w("  Checks if the report contains facts that contradict the reference medical")
    w("  context for the predicted tumor class.  Lower score = fewer hallucinations.")
    w(f"  Threshold : score < {thresholds['hallucination']:.1f} → PASS")
    w()
    w("  Metric 2 — AnswerRelevancyMetric")
    w("  ---------------------------------")
    w("  Measures whether the report is relevant to the input diagnosis prompt.")
    w(f"  Threshold : score >= {thresholds['relevancy']:.1f} → PASS")
    w()
    w("  Metric 3 — BiasMetric")
    w("  ----------------------")
    w("  Flags biased language, demographic assumptions, or unjustified certainty.")
    w(f"  Threshold : score < {thresholds['bias']:.1f} → PASS")
    w()
    w("  Metric 4 — GEval (Medical Terminology Completeness)")
    w("  -----------------------------------------------------")
    w("  Custom LLM-as-judge: rates whether the report uses correct medical")
    w("  terminology for the specific tumor class.")
    w(f"  Threshold : score >= {thresholds['geval']:.1f} → PASS")
    w()
    w("  NOTE: Local models (7B–13B) may occasionally produce non-JSON output.")
    w("  Those metric calls are marked as errored and excluded from pass rates.")
    w("  Use a larger model (qwen2.5, gemma2) for more stable JSON output.")
    w()

    # Overall summary
    w(BAR)
    w("  OVERALL BENCHMARK SUMMARY")
    w(BAR)
    n = stats.get("valid", 0)
    if n:
        passed = stats["passed"]
        w(f"  Total samples   : {stats['total']}")
        w(f"  Skipped/Errored : {stats['skipped']}")
        w(f"  Validated       : {n}")
        w(f"  Overall PASS    : {passed}  ({passed/n:.1%})")
        w()
        w("  Metric Pass Rates:")
        w(f"    Hallucination (< {thresholds['hallucination']:.1f}) : "
          f"{stats['hallucination_pass_rate']:.1%}   "
          f"avg score = {stats['avg_hallucination_score']:.4f}")
        w(f"    Relevancy    (>={thresholds['relevancy']:.1f}) : "
          f"{stats['relevancy_pass_rate']:.1%}   "
          f"avg score = {stats['avg_relevancy_score']:.4f}")
        w(f"    Bias         (< {thresholds['bias']:.1f}) : "
          f"{stats['bias_pass_rate']:.1%}   "
          f"avg score = {stats['avg_bias_score']:.4f}")
        w(f"    GEval        (>={thresholds['geval']:.1f}) : "
          f"{stats['geval_pass_rate']:.1%}   "
          f"avg score = {stats['avg_geval_score']:.4f}")
    else:
        w("  No validated samples.")
    w()

    # Per-class table
    w(BAR)
    w("  PER-CLASS BREAKDOWN")
    w(BAR)
    cls_stats = stats.get("per_class", {})
    if cls_stats:
        w(f"  {'Class':<12} {'N':>3}  {'Pass':>4}  {'Pass%':>5}  "
          f"{'Halluc%':>7}  {'Relev%':>6}  {'Bias%':>5}  {'GEval%':>6}")
        w("  " + THIN[2:])
        for cls in CLASSES:
            if cls not in cls_stats:
                continue
            c = cls_stats[cls]
            pp = c["passed"] / c["n"] if c["n"] else 0.0
            w(f"  {cls:<12} {c['n']:>3}  {c['passed']:>4}  {pp:>4.0%}   "
              f"{c['hallucination_pass_rate']:>6.0%}   "
              f"{c['relevancy_pass_rate']:>5.0%}   "
              f"{c['bias_pass_rate']:>4.0%}   "
              f"{c['geval_pass_rate']:>5.0%}")
    else:
        w("  No per-class data available.")
    w()

    # Individual results
    w(BAR)
    w("  INDIVIDUAL SAMPLE RESULTS")
    w(BAR)
    w()
    valid_s   = [s for s in samples if s.error is None and s.metrics_run > 0]
    skipped_s = [s for s in samples if s.error is not None or s.metrics_run == 0]

    for s in valid_s:
        status  = "PASS" if s.overall_pass else "FAIL"
        h_score = f"{s.hallucination_score:.3f}" if s.hallucination_score is not None else " N/A"
        r_score = f"{s.relevancy_score:.3f}"     if s.relevancy_score     is not None else " N/A"
        b_score = f"{s.bias_score:.3f}"          if s.bias_score          is not None else " N/A"
        g_score = f"{s.geval_score:.3f}"         if s.geval_score         is not None else " N/A"
        h_pass  = "Y" if s.hallucination_pass else ("N" if s.hallucination_pass is False else "?")
        r_pass  = "Y" if s.relevancy_pass     else ("N" if s.relevancy_pass     is False else "?")
        b_pass  = "Y" if s.bias_pass          else ("N" if s.bias_pass          is False else "?")
        g_pass  = "Y" if s.geval_pass         else ("N" if s.geval_pass         is False else "?")

        w(f"  [{status}] {s.image_filename}")
        w(f"         True class : {s.true_class.upper():<12}  Predicted : {s.predicted_class.upper()}")
        w(f"         DL conf    : {s.model_confidence:.1%}  |  "
          f"Halluc:{h_score}[{h_pass}]  "
          f"Relev:{r_score}[{r_pass}]  "
          f"Bias:{b_score}[{b_pass}]  "
          f"GEval:{g_score}[{g_pass}]")
        for attr, label in [
            ("hallucination_reason", "Halluc"),
            ("relevancy_reason",     "Relev "),
            ("bias_reason",          "Bias  "),
            ("geval_reason",         "GEval "),
        ]:
            reason = getattr(s, attr)
            passed = getattr(s, attr.replace("_reason", "_pass"))
            if reason and (passed is False or "Error" in reason):
                w(f"         [{label}] {reason[:130].replace(chr(10), ' ')}")
        w()

    for s in skipped_s:
        w(f"  [SKIP] {s.image_filename}  ({s.true_class.upper()}) — {s.error}")
    if skipped_s:
        w()

    # Interpretation guide
    w(BAR)
    w("  INTERPRETATION GUIDE")
    w(BAR)
    w()
    w("  Metric              Pass direction    Target for clinical use")
    w("  " + THIN[2:])
    w("  Hallucination     lower is better    < 0.3  (minimal hallucination)")
    w("  Answer Relevancy  higher is better   > 0.7  (strongly on-topic)")
    w("  Bias              lower is better    < 0.3  (minimal bias)")
    w("  GEval (Medical)   higher is better   > 0.7  (complete terminology)")
    w()
    w("  HOW THIS DIFFERS FROM THE 4-TIER RULE-BASED VALIDATOR")
    w("  -------------------------------------------------------")
    w("  4-tier validator  : checks structural sections, keyword presence, regex")
    w("  DeepEval (this)   : semantic understanding — hallucination, bias, relevance")
    w("  A report can pass all 4 tiers and still hallucinate (correct words, wrong facts).")
    w("  Use both validators together for complete quality assessment in the thesis.")
    w()
    w("  JUDGE MODEL QUALITY NOTE")
    w("  -------------------------")
    w("  Smaller Ollama models (7B) may produce inconsistent scores.")
    w("  For most reliable results use: meditron, qwen2.5, or gemma2.")
    w("  Human expert review is ALWAYS required before any clinical use.")
    w(BAR)

    REPORT_TXT.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(
    csv_path:                Path  = INPUT_CSV,
    ollama_model:            str   = DEFAULT_OLLAMA_MODEL,
    ollama_url:              str   = DEFAULT_OLLAMA_URL,
    ollama_timeout:          int   = 180,
    hallucination_threshold: float = DEFAULT_HALLUCINATION_THRESHOLD,
    relevancy_threshold:     float = DEFAULT_RELEVANCY_THRESHOLD,
    bias_threshold:          float = DEFAULT_BIAS_THRESHOLD,
    geval_threshold:         float = DEFAULT_GEVAL_THRESHOLD,
    max_rows:                Optional[int] = None,
) -> None:
    print("\n" + "=" * 60)
    print("  DEEPEVAL LLM OUTPUT VALIDATOR  (Ollama / no API key)")
    print("=" * 60)
    print(f"  Input        : {csv_path}")
    print(f"  Judge model  : {ollama_model}  @ {ollama_url}")
    print(f"  Thresholds   : Halluc<{hallucination_threshold}  "
          f"Relev>={relevancy_threshold}  Bias<{bias_threshold}  "
          f"GEval>={geval_threshold}")
    print("=" * 60 + "\n")

    # --- Pre-flight checks ---
    if not DEEPEVAL_AVAILABLE:
        print("[ERROR] DeepEval is not installed.")
        print("  Run:  pip install deepeval")
        sys.exit(1)

    print(f"  Checking Ollama at {ollama_url} ... ", end="", flush=True)
    if not _check_ollama_running(ollama_url):
        print("NOT RUNNING\n")
        print("[ERROR] Cannot connect to Ollama.")
        print(f"  Make sure Ollama is running:  ollama serve")
        print(f"  Then pull the judge model:    ollama pull {ollama_model}")
        sys.exit(1)
    print("OK")

    available_models = _list_ollama_models(ollama_url)
    model_present = any(ollama_model in m for m in available_models)
    if not model_present:
        print(f"\n[WARNING] Model '{ollama_model}' not found in Ollama.")
        print(f"  Available models: {available_models or '(none pulled yet)'}")
        print(f"  Pull it with:  ollama pull {ollama_model}")
        print(f"  Continuing anyway — Ollama will attempt to pull on first use.\n")
    else:
        print(f"  Judge model '{ollama_model}' is available.")

    if not csv_path.exists():
        print(f"\n[ERROR] CSV not found: {csv_path}")
        print("  Run collect_llm_outputs.py first.")
        sys.exit(1)

    # --- Load data ---
    with open(csv_path, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if max_rows:
        rows = rows[:max_rows]
    print(f"\n  Loaded {len(rows)} rows from CSV\n")

    # --- Build judge + metrics ---
    judge = OllamaJudge(
        model_name=ollama_model,
        base_url=ollama_url,
        timeout=ollama_timeout,
    )
    thresholds = {
        "hallucination": hallucination_threshold,
        "relevancy":     relevancy_threshold,
        "bias":          bias_threshold,
        "geval":         geval_threshold,
    }
    metrics = _build_metrics(
        judge=judge,
        hallucination_threshold=hallucination_threshold,
        relevancy_threshold=relevancy_threshold,
        bias_threshold=bias_threshold,
        geval_threshold=geval_threshold,
    )

    # --- Validation loop ---
    samples: list[DeepEvalSampleResult] = []
    total = len(rows)

    for i, row in enumerate(rows, start=1):
        fname = row.get("image_filename", f"row_{i}")
        cls   = row.get("true_class", "unknown")
        print(f"  [{i:02d}/{total}] {fname} ({cls}) ... ", end="", flush=True)

        sample = _validate_one(row, metrics)

        if sample.error:
            print(f"SKIP  ({sample.error})")
        else:
            status = "PASS" if sample.overall_pass else "FAIL"
            parts = []
            if sample.hallucination_score is not None:
                parts.append(f"H:{sample.hallucination_score:.2f}")
            if sample.relevancy_score is not None:
                parts.append(f"R:{sample.relevancy_score:.2f}")
            if sample.bias_score is not None:
                parts.append(f"B:{sample.bias_score:.2f}")
            if sample.geval_score is not None:
                parts.append(f"G:{sample.geval_score:.2f}")
            print(f"{status}  {'  '.join(parts) or 'all metrics errored'}")
            _save_json(sample)

        samples.append(sample)

    # --- Stats + report ---
    stats = _compute_stats(samples)
    _write_report(stats, samples, thresholds, judge_model=judge.get_model_name())

    valid  = stats.get("valid", 0)
    passed = stats.get("passed", 0)
    print("\n" + "=" * 60)
    print(f"  Validated    : {valid}/{total} rows")
    if valid:
        print(f"  Pass rate    : {passed}/{valid}  ({passed/valid:.1%})")
        print(f"  Hallucination pass : {stats['hallucination_pass_rate']:.1%}  "
              f"avg={stats['avg_hallucination_score']:.3f}")
        print(f"  Relevancy pass     : {stats['relevancy_pass_rate']:.1%}  "
              f"avg={stats['avg_relevancy_score']:.3f}")
        print(f"  Bias pass          : {stats['bias_pass_rate']:.1%}  "
              f"avg={stats['avg_bias_score']:.3f}")
        print(f"  GEval pass         : {stats['geval_pass_rate']:.1%}  "
              f"avg={stats['avg_geval_score']:.3f}")
    print(f"  Report       : {REPORT_TXT}")
    print(f"  JSON files   : {RESULTS_DIR}/deepeval_*.json")
    print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="DeepEval LLM validator using a local Ollama judge (no API key)."
    )
    parser.add_argument("--csv",           type=Path, default=INPUT_CSV,
                        help=f"Path to llm_outputs.csv (default: {INPUT_CSV})")
    parser.add_argument("--ollama-model",  type=str, default=DEFAULT_OLLAMA_MODEL,
                        help=f"Ollama model to use as judge (default: {DEFAULT_OLLAMA_MODEL})")
    parser.add_argument("--ollama-url",    type=str, default=DEFAULT_OLLAMA_URL,
                        help=f"Ollama base URL (default: {DEFAULT_OLLAMA_URL})")
    parser.add_argument("--ollama-timeout", type=int, default=180,
                        help="Seconds to wait per Ollama call (default: 180)")
    parser.add_argument("--hallucination-threshold", type=float,
                        default=DEFAULT_HALLUCINATION_THRESHOLD)
    parser.add_argument("--relevancy-threshold",     type=float,
                        default=DEFAULT_RELEVANCY_THRESHOLD)
    parser.add_argument("--bias-threshold",          type=float,
                        default=DEFAULT_BIAS_THRESHOLD)
    parser.add_argument("--geval-threshold",         type=float,
                        default=DEFAULT_GEVAL_THRESHOLD)
    parser.add_argument("--max-rows", type=int, default=None,
                        help="Limit to first N rows (useful for quick testing)")
    args = parser.parse_args()

    main(
        csv_path=args.csv,
        ollama_model=args.ollama_model,
        ollama_url=args.ollama_url,
        ollama_timeout=args.ollama_timeout,
        hallucination_threshold=args.hallucination_threshold,
        relevancy_threshold=args.relevancy_threshold,
        bias_threshold=args.bias_threshold,
        geval_threshold=args.geval_threshold,
        max_rows=args.max_rows,
    )
