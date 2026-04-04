"""
LLM Validation Benchmark
=========================
Reads llm_outputs.csv (produced by collect_llm_outputs.py),
runs the four-tier LLM output validator on every row,
and writes a detailed benchmark report to:

    llm_output_validate/llm_validation_benchmark.txt

Each individual validation JSON is also saved under:
    llm_output_validate/results/

Usage
-----
  python llm_output_validate/run_llm_validation_benchmark.py
  python llm_output_validate/run_llm_validation_benchmark.py --csv llm_outputs.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Ensure project root is importable so the validator can be found
# ---------------------------------------------------------------------------
SCRIPT_DIR   = Path(__file__).resolve().parent   # llm_output_validate/
PROJECT_ROOT = SCRIPT_DIR.parent                 # brain_tumor_app/

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from llm_output_validate.llm_output_validator import (   # noqa: E402
    ValidationReport,
    save_report,
    validate_llm_report,
)

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------
INPUT_CSV      = SCRIPT_DIR / "llm_outputs.csv"
RESULTS_DIR    = SCRIPT_DIR / "results"
BENCHMARK_TXT  = SCRIPT_DIR / "llm_validation_benchmark.txt"
PASS_THRESHOLD = 0.65

CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]

# Tier weights (must match llm_output_validator.py)
TIER_WEIGHTS = {"T1": 0.30, "T2": 0.35, "T3": 0.25, "T4": 0.10}

# ---------------------------------------------------------------------------
# CSV reading
# ---------------------------------------------------------------------------

def _read_csv(csv_path: Path) -> list[dict]:
    with open(csv_path, encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


# ---------------------------------------------------------------------------
# Validation loop
# ---------------------------------------------------------------------------

def _run_validation(
    rows: list[dict],
) -> list[tuple[dict, Optional[ValidationReport]]]:
    results: list[tuple[dict, Optional[ValidationReport]]] = []
    total = len(rows)

    for i, row in enumerate(rows, start=1):
        cls      = row.get("true_class", "unknown").lower().strip()
        model    = row.get("model_name", "unknown")
        fname    = row.get("image_filename", f"row_{i}")
        llm_text = row.get("llm_output", "").strip()

        try:
            confidence = float(row.get("confidence", "0.0"))
        except ValueError:
            confidence = 0.0

        print(f"  [{i:02d}/{total}] {fname} ({cls})  conf={confidence:.1%} ... ",
              end="", flush=True)

        if not llm_text:
            print("SKIP (empty llm_output)")
            results.append((row, None))
            continue

        report = validate_llm_report(
            llm_text=llm_text,
            predicted_class=cls,
            model_confidence=confidence,
            model_name=model,
            pass_threshold=PASS_THRESHOLD,
        )
        save_report(report, str(RESULTS_DIR))

        status = "PASS" if report.overall_pass else "FAIL"
        print(f"{status}  score={report.overall_score:.3f}")
        results.append((row, report))

    return results


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def _compute_stats(
    results: list[tuple[dict, Optional[ValidationReport]]],
) -> dict:
    valid = [(row, r) for row, r in results if r is not None]
    if not valid:
        return {}

    all_r = [r for _, r in valid]
    n     = len(all_r)

    def avg(seq) -> float:
        return sum(seq) / len(seq) if seq else 0.0

    stats: dict = {
        "total"       : n,
        "skipped"     : len(results) - n,
        "passed"      : sum(1 for r in all_r if r.overall_pass),
        "avg_overall" : avg([r.overall_score               for r in all_r]),
        "avg_t1"      : avg([r.tier1_structure_score       for r in all_r]),
        "avg_t2"      : avg([1.0 if r.tier2_diagnostic_consistent    else 0.0 for r in all_r]),
        "avg_t3"      : avg([r.tier3_medical_plausibility_score      for r in all_r]),
        "avg_t4"      : avg([1.0 if r.tier4_confidence_aligned       else 0.0 for r in all_r]),
        "per_class"   : {},
    }
    stats["failed"] = n - stats["passed"]

    for cls in CLASSES:
        cls_pairs = [(row, r) for row, r in valid
                     if row.get("true_class", "").lower() == cls]
        if not cls_pairs:
            continue
        cls_r = [r for _, r in cls_pairs]
        m = len(cls_r)
        stats["per_class"][cls] = {
            "n"       : m,
            "passed"  : sum(1 for r in cls_r if r.overall_pass),
            "avg_overall" : avg([r.overall_score for r in cls_r]),
            "avg_t1"  : avg([r.tier1_structure_score for r in cls_r]),
            "avg_t2"  : avg([1.0 if r.tier2_diagnostic_consistent else 0.0 for r in cls_r]),
            "avg_t3"  : avg([r.tier3_medical_plausibility_score   for r in cls_r]),
            "avg_t4"  : avg([1.0 if r.tier4_confidence_aligned    else 0.0 for r in cls_r]),
        }

    return stats


# ---------------------------------------------------------------------------
# Benchmark .txt writer
# ---------------------------------------------------------------------------

def _write_benchmark_txt(
    stats: dict,
    results: list[tuple[dict, Optional[ValidationReport]]],
    output_path: Path,
) -> None:
    BAR  = "=" * 72
    THIN = "-" * 72
    lines: list[str] = []

    def w(s: str = "") -> None:
        lines.append(s)

    # ------------------------------------------------------------------ header
    w(BAR)
    w("  LLM OUTPUT VALIDATION BENCHMARK REPORT")
    w(f"  Generated  : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    w(f"  Input CSV  : llm_outputs.csv")
    w(f"  Results    : llm_output_validate/results/")
    w(f"  Threshold  : {PASS_THRESHOLD:.0%}  (minimum overall score to PASS)")
    w(BAR)
    w()

    # ------------------------------------------------------------ methodology
    w("METHODOLOGY")
    w(THIN)
    w()
    w("  How this validator works")
    w("  ------------------------")
    w("  Each LLM-generated clinical report is automatically checked against")
    w("  four independent tiers. No human reviewer is required for this pass.")
    w("  The validator catches structural, diagnostic, medical, and confidence")
    w("  issues — but CANNOT judge prose quality or subtle clinical reasoning.")
    w()
    w("  Tier 1 — Structural Completeness  (weight 30%)")
    w("  ------------------------------------------------")
    w("  All 12 required sections must be present and contain >= 20 characters.")
    w("  Required sections:")
    w("    • QUICK CLINICAL DECISION    • Executive Summary")
    w("    • Clinical Presentation      • Imaging Findings")
    w("    • Differential Diagnosis     • Pathophysiology")
    w("    • Tumor Grading              • Clinical Implications")
    w("    • Management Plan            • Next Steps")
    w("    • Educational Pearls         • References")
    w()
    w("  Tier 2 — Diagnostic Consistency  (weight 35%)")
    w("  ------------------------------------------------")
    w("  The LLM text must mention the DL model's predicted class using at")
    w("  least one accepted synonym.  Urgency level must match expected range.")
    w()
    w("  Class         Accepted synonyms                      Urgency")
    w("  glioma        glioma/glioblastoma/GBM/astrocytoma    STAT or Urgent")
    w("  meningioma    meningioma/meningeal/dural tail         STAT/Urgent/Routine")
    w("  pituitary     pituitary/adenoma/sellar/macroadenoma   STAT/Urgent/Routine")
    w("  notumor       no tumor/normal/no evidence             Routine only")
    w()
    w("  Tier 3 — Medical Plausibility  (weight 25%)")
    w("  ---------------------------------------------")
    w("  Class-specific medical terms must appear in the report.")
    w("  If sentence-transformers is installed, cosine similarity to a")
    w("  reference summary is also measured (warn if < 0.30).")
    w()
    w("  Class        Required terms")
    w("  glioma       IDH, MGMT, WHO grade, resection, temozolomide")
    w("  meningioma   dural, resection, grade, surgery, seizure")
    w("  pituitary    hormone, vision, sellar, prolactin, cortisol")
    w("  notumor      normal, no evidence, follow-up, routine")
    w()
    w("  Tier 4 — Confidence Alignment  (weight 10%)")
    w("  ---------------------------------------------")
    w("  LLM confidence label extracted and compared to DL numeric score.")
    w("  High   → 90–100%   |   Medium → 70–90%   |   Low → 0–70%")
    w()
    w("  Overall Score = 0.30 × T1 + 0.35 × T2 + 0.25 × T3 + 0.10 × T4")
    w(f"  PASS if Overall Score >= {PASS_THRESHOLD:.0%}")
    w()

    # --------------------------------------------------------- overall summary
    w(BAR)
    w("  OVERALL BENCHMARK SUMMARY")
    w(BAR)
    if stats:
        failed_pct = (stats["failed"] / stats["total"]) if stats["total"] else 0
        w(f"  Total samples  : {stats['total']}")
        w(f"  Skipped        : {stats['skipped']}  (empty llm_output)")
        w(f"  Passed         : {stats['passed']}  ({stats['passed']/stats['total']:.1%})")
        w(f"  Failed         : {stats['failed']}  ({failed_pct:.1%})")
        w()
        w(f"  Avg Overall Score  : {stats['avg_overall']:.4f}")
        w(f"  Avg Tier 1 (Struct): {stats['avg_t1']:.1%}")
        w(f"  Avg Tier 2 (Diag)  : {stats['avg_t2']:.1%}")
        w(f"  Avg Tier 3 (Med)   : {stats['avg_t3']:.1%}")
        w(f"  Avg Tier 4 (Conf)  : {stats['avg_t4']:.1%}")
    else:
        w("  No validated samples to summarise.")
    w()

    # --------------------------------------------------------- per-class table
    w(BAR)
    w("  PER-CLASS BREAKDOWN")
    w(BAR)
    cls_stats = stats.get("per_class", {})
    if cls_stats:
        w(f"  {'Class':<12} {'N':>3}  {'Pass':>4}  {'Pass%':>5}  "
          f"{'Overall':>7}  {'T1':>5}  {'T2':>5}  {'T3':>5}  {'T4':>5}")
        w("  " + THIN[2:])
        for cls in CLASSES:
            if cls not in cls_stats:
                continue
            c        = cls_stats[cls]
            pass_pct = c["passed"] / c["n"] if c["n"] else 0.0
            w(
                f"  {cls:<12} {c['n']:>3}  {c['passed']:>4}  {pass_pct:>4.0%}   "
                f"{c['avg_overall']:>7.4f}  {c['avg_t1']:>4.0%}   "
                f"{c['avg_t2']:>4.0%}   {c['avg_t3']:>4.0%}   {c['avg_t4']:>4.0%}"
            )
    else:
        w("  No per-class data available.")
    w()

    # ---------------------------------------------------- individual results
    w(BAR)
    w("  INDIVIDUAL SAMPLE RESULTS")
    w(BAR)
    w()

    valid_results  = [(row, r) for row, r in results if r is not None]
    skipped_results = [(row, r) for row, r in results if r is None]

    for row, report in valid_results:
        fname      = row.get("image_filename", "?")
        true_cls   = row.get("true_class", "?").upper()
        pred_cls   = report.predicted_class.upper()
        conf       = report.model_confidence
        score      = report.overall_score
        status     = "PASS" if report.overall_pass else "FAIL"
        t1         = report.tier1_structure_score
        t2         = "Y" if report.tier2_diagnostic_consistent else "N"
        t3         = report.tier3_medical_plausibility_score
        t4         = "Y" if report.tier4_confidence_aligned else "N"
        diag       = report.extracted_diagnosis or "—"
        urgency    = report.extracted_urgency   or "—"
        conf_lbl   = report.extracted_confidence_text or "—"

        w(f"  [{status}] {fname}")
        w(f"         True class  : {true_cls:<12}  Predicted : {pred_cls}")
        w(f"         DL conf     : {f'{conf:.1%}':<10}  LLM conf label : {conf_lbl}")
        w(f"         Score       : {score:.4f}    T1:{t1:.0%}  T2:{t2}  T3:{t3:.0%}  T4:{t4}")
        w(f"         LLM diag    : {diag}         Urgency : {urgency}")

        errors   = [i.message for i in report.issues if i.severity == "ERROR"
                    and "Overall" not in i.tier]
        warnings = [i.message for i in report.issues if i.severity == "WARNING"]
        for msg in errors:
            w(f"         [ERROR]   {msg}")
        for msg in warnings[:2]:       # cap at 2 warnings to keep file readable
            w(f"         [WARNING] {msg}")
        if len(warnings) > 2:
            w(f"         [WARNING] ... and {len(warnings)-2} more warning(s)")
        w()

    for row, _ in skipped_results:
        fname    = row.get("image_filename", "?")
        true_cls = row.get("true_class", "?").upper()
        w(f"  [SKIP] {fname}  ({true_cls}) — llm_output was empty")
    if skipped_results:
        w()

    # ------------------------------------------------------ interpretation guide
    w(BAR)
    w("  BENCHMARK INTERPRETATION GUIDE")
    w(BAR)
    w()
    w("  Score Range  Quality       Interpretation")
    w("  " + THIN[2:])
    w("  0.90 – 1.00  Excellent     All tiers pass. Report is clinically coherent,")
    w("                             structurally complete, and medically plausible.")
    w()
    w("  0.75 – 0.89  Good          Minor gaps: possibly 1–2 missing medical terms,")
    w("                             or an urgency-level mismatch. Acceptable for")
    w("                             research / educational use.")
    w()
    w("  0.65 – 0.74  Acceptable    Passes the threshold but has structural or")
    w("                             medical issues worth investigating.")
    w()
    w("  0.50 – 0.64  Borderline    FAILS threshold. Significant tier failures.")
    w("                             LLM report needs review before use.")
    w()
    w("  0.00 – 0.49  Poor          FAILS. Fundamental diagnostic inconsistency.")
    w("                             LLM likely hallucinated class or sections.")
    w()
    w("  Diagnosing failures — check these signals first:")
    w("  ┌─────────────────────────────────────────────────────────────────────┐")
    w("  │ Tier 1 < 80%  →  missing sections or very short section content     │")
    w("  │ Tier 2 = N    →  LLM ignored or contradicted the DL prediction      │")
    w("  │ Tier 3 < 60%  →  critical medical terms absent (hallucination risk)  │")
    w("  │ Tier 4 = N    →  LLM confidence label doesn't match DL numeric score │")
    w("  └─────────────────────────────────────────────────────────────────────┘")
    w()
    w("  IMPORTANT LIMITATIONS")
    w("  ----------------------")
    w("  This validator is an AUTOMATED CONTRACT CHECKER — it verifies")
    w("  structural and lexical properties, not clinical correctness.")
    w("  A report can PASS all tiers and still contain:")
    w("    • Clinically incorrect dosages or treatment recommendations")
    w("    • Hallucinated patient details")
    w("    • Medically implausible reasoning that happens to use correct terms")
    w()
    w("  Human expert review is ALWAYS required before clinical use.")
    w()
    w(BAR)

    output_path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(csv_path: Path = INPUT_CSV) -> None:
    print("\n" + "=" * 60)
    print("  LLM OUTPUT VALIDATION BENCHMARK")
    print("=" * 60)
    print(f"  Input  : {csv_path}")
    print(f"  Output : {BENCHMARK_TXT}")
    print(f"  Results dir : {RESULTS_DIR}")
    print("=" * 60 + "\n")

    if not csv_path.exists():
        print(f"[ERROR] CSV not found: {csv_path}")
        print("  Run collect_llm_outputs.py first to generate it.")
        sys.exit(1)

    rows = _read_csv(csv_path)
    print(f"  Loaded {len(rows)} rows from CSV\n")

    results = _run_validation(rows)
    stats   = _compute_stats(results)
    _write_benchmark_txt(stats, results, BENCHMARK_TXT)

    valid_count = sum(1 for _, r in results if r is not None)
    passed      = sum(1 for _, r in results if r is not None and r.overall_pass)

    print("=" * 60)
    print(f"  Validated  : {valid_count}/{len(rows)} rows")
    if valid_count:
        print(f"  Pass rate  : {passed}/{valid_count}  ({passed/valid_count:.1%})")
    print(f"  Report     : {BENCHMARK_TXT}")
    print(f"  JSON files : {RESULTS_DIR}/")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run LLM output validation and write benchmark report."
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=INPUT_CSV,
        help="Path to llm_outputs.csv (default: llm_output_validate/llm_outputs.csv)",
    )
    args = parser.parse_args()
    main(csv_path=args.csv)
