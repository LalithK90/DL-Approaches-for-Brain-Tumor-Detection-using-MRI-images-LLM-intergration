"""
Collect LLM outputs from the Flask API.
=========================================
Sends 10 images per tumor class to the running Flask API (/predict),
captures the LLM-generated clinical report for each image, and writes
all results to  llm_output_validate/llm_outputs.csv

Columns saved:
  timestamp | model_name | true_class | predicted_class |
  confidence | image_filename | llm_output

Prerequisites
-------------
  - Flask API must be running on http://localhost:5001
    Start it with:
        cd brain_tumor_identification_api
        conda activate mri_data
        python app.py

Usage
-----
  # From project root or llm_output_validate/ directory:
  python llm_output_validate/collect_llm_outputs.py
  python llm_output_validate/collect_llm_outputs.py --model propose_balanced --images-per-class 10
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import requests

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR   = Path(__file__).resolve().parent          # llm_output_validate/
PROJECT_ROOT = SCRIPT_DIR.parent                        # brain_tumor_app/
TESTING_DIR  = PROJECT_ROOT / "xai_validation" / "testing"
OUTPUT_CSV   = SCRIPT_DIR / "llm_outputs.csv"

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CLASSES          = ["glioma", "meningioma", "notumor", "pituitary"]
IMAGES_PER_CLASS = 10
MODEL_NAME       = "propose_balanced"
API_URL          = "http://localhost:5001/predict"
REQUEST_TIMEOUT  = 600   # seconds — LLM + XAI pipeline can be slow per image

CSV_FIELDS = [
    "timestamp",
    "model_name",
    "true_class",
    "predicted_class",
    "confidence",
    "image_filename",
    "llm_output",
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _check_flask_running() -> bool:
    """Return True if Flask is responding on port 5001."""
    try:
        requests.get("http://localhost:5001", timeout=5)
        return True
    except requests.exceptions.ConnectionError:
        return False


def _pick_images(cls: str, n: int) -> list[Path]:
    """Return n randomly-sampled image paths from the testing/<cls> folder."""
    cls_dir = TESTING_DIR / cls
    if not cls_dir.is_dir():
        raise FileNotFoundError(
            f"Testing directory not found: {cls_dir}\n"
            f"Expected layout: xai_validation/testing/{cls}/*.jpg"
        )
    candidates = (
        sorted(cls_dir.glob("*.jpg"))
        + sorted(cls_dir.glob("*.jpeg"))
        + sorted(cls_dir.glob("*.png"))
    )
    if len(candidates) < n:
        raise ValueError(
            f"Only {len(candidates)} images found in {cls_dir}, need {n}."
        )
    return random.sample(candidates, n)


def _call_predict(img_path: Path, model_name: str) -> dict:
    """POST one image to the Flask /predict endpoint and return the JSON."""
    with open(img_path, "rb") as fh:
        files = {"file": (img_path.name, fh, "image/jpeg")}
        data  = {"model_name": model_name}
        resp  = requests.post(API_URL, files=files, data=data,
                              timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def _extract_confidence(response: dict) -> float:
    """
    Extract numeric top-1 confidence from the API response.

    The /predict route returns top3 as a list of (label, prob) pairs,
    serialised to JSON as [[label, prob], ...].
    """
    top3 = response.get("top3")
    if top3 and isinstance(top3, list):
        first = top3[0]
        if isinstance(first, (list, tuple)) and len(first) >= 2:
            return float(first[1])
    return 0.0


# ---------------------------------------------------------------------------
# Main collection loop
# ---------------------------------------------------------------------------

def run_collection(
    model_name: str       = MODEL_NAME,
    images_per_class: int = IMAGES_PER_CLASS,
) -> None:
    print("\n" + "=" * 60)
    print("  LLM OUTPUT COLLECTION")
    print("=" * 60)
    print(f"  Model          : {model_name}")
    print(f"  Images / class : {images_per_class}")
    print(f"  Total images   : {images_per_class * len(CLASSES)}")
    print(f"  Output CSV     : {OUTPUT_CSV}")
    print("=" * 60)

    if not _check_flask_running():
        print("\n[ERROR] Flask API is NOT running on http://localhost:5001")
        print("  Start it with:")
        print("    cd brain_tumor_identification_api")
        print("    conda activate mri_data && python app.py")
        sys.exit(1)

    print("\n  Flask API status : Running OK\n")

    # Open CSV in append mode so reruns add rows instead of overwriting
    write_header = not OUTPUT_CSV.exists()
    rows_saved   = 0
    rows_failed  = 0

    with open(OUTPUT_CSV, "a", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()

        for cls in CLASSES:
            print(f"[{cls.upper()}]")
            images = _pick_images(cls, images_per_class)

            for idx, img_path in enumerate(images, start=1):
                print(f"  {idx:02d}/{images_per_class}  {img_path.name:<30}", end="", flush=True)
                t0 = time.time()

                try:
                    response    = _call_predict(img_path, model_name)
                    elapsed     = time.time() - t0
                    predicted   = response.get("prediction", "unknown")
                    confidence  = _extract_confidence(response)
                    llm_output  = response.get("final_report", "")

                    if not llm_output:
                        print(f"  WARNING: final_report is empty  [{elapsed:.0f}s]")

                    writer.writerow({
                        "timestamp"       : datetime.now().isoformat(),
                        "model_name"      : model_name,
                        "true_class"      : cls,
                        "predicted_class" : predicted,
                        "confidence"      : round(confidence, 4),
                        "image_filename"  : img_path.name,
                        "llm_output"      : llm_output,
                    })
                    csvfile.flush()
                    rows_saved += 1
                    print(f"  -> {predicted} ({confidence:.1%})  [{elapsed:.0f}s]")

                except Exception as exc:
                    elapsed      = time.time() - t0
                    rows_failed += 1
                    print(f"  -> FAILED [{elapsed:.0f}s]  {exc}")

            print()

    print("=" * 60)
    print(f"  Saved  : {rows_saved} rows")
    print(f"  Failed : {rows_failed} rows")
    print(f"  CSV    : {OUTPUT_CSV}")
    print("=" * 60)
    print()
    if rows_saved > 0:
        print("  Next step: run the validation benchmark")
        print("    python llm_output_validate/run_llm_validation_benchmark.py")
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Collect LLM outputs from the Flask /predict API."
    )
    parser.add_argument(
        "--model",
        default=MODEL_NAME,
        help=f"DL model name passed to API (default: {MODEL_NAME})",
    )
    parser.add_argument(
        "--images-per-class",
        type=int,
        default=IMAGES_PER_CLASS,
        help=f"Number of images per class (default: {IMAGES_PER_CLASS})",
    )
    args = parser.parse_args()
    run_collection(model_name=args.model, images_per_class=args.images_per_class)
