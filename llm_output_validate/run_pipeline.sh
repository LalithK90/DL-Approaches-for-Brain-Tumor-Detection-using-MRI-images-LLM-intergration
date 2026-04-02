#!/bin/bash

set -euo pipefail

# ---------------------------------------------------------------------------
# Colours
# ---------------------------------------------------------------------------
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

# ---------------------------------------------------------------------------
# Paths  (script lives inside llm_output_validate/)
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
FLASK_DIR="$PROJECT_ROOT/brain_tumor_identification_api"
COLLECT_SCRIPT="$SCRIPT_DIR/collect_llm_outputs.py"
BENCHMARK_SCRIPT="$SCRIPT_DIR/run_llm_validation_benchmark.py"
OUTPUT_CSV="$SCRIPT_DIR/llm_outputs.csv"
BENCHMARK_TXT="$SCRIPT_DIR/llm_validation_benchmark.txt"
FLASK_LOG="$PROJECT_ROOT/logs/flask_server.log"

FLASK_PID=""
WE_STARTED_FLASK=false

# ---------------------------------------------------------------------------
# All available models (mirrors brain_tumor_identification_api/src/models/models.py)
# ---------------------------------------------------------------------------
ALL_MODELS=(
    "vgg19_balanced"
    "vgg19_imbalanced"
    "vgg16_balanced"
    "vgg16_imbalanced"
    "propose_balanced"
    "propose_imbalanced"
    "ResNet50_balanced"
    "ResNet50_imbalanced"
    "MobileVNet_balanced"
    "MobileVNet_imbalanced"
    "GoogleLeNet_balanced"
    "GoogleLeNet_imbalanced"
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
print_header() {
    echo ""
    echo -e "${BLUE}${BOLD}============================================================${NC}"
    echo -e "${BLUE}${BOLD}  $1${NC}"
    echo -e "${BLUE}${BOLD}============================================================${NC}"
}

print_step() {
    echo ""
    echo -e "${CYAN}${BOLD}▶  $1${NC}"
}

print_ok()   { echo -e "${GREEN}  ✓  $1${NC}"; }
print_warn() { echo -e "${YELLOW}  ⚠  $1${NC}"; }
print_err()  { echo -e "${RED}  ✗  $1${NC}"; }

# ---------------------------------------------------------------------------
# Cleanup on exit / Ctrl-C
# ---------------------------------------------------------------------------
cleanup() {
    echo ""
    if [ "$WE_STARTED_FLASK" = true ] && [ -n "$FLASK_PID" ]; then
        echo -e "${YELLOW}  Stopping Flask (PID $FLASK_PID)...${NC}"
        kill "$FLASK_PID" 2>/dev/null || true
        wait "$FLASK_PID" 2>/dev/null || true
        print_ok "Flask stopped"
    fi
    echo ""
}
trap cleanup EXIT SIGINT SIGTERM

# ---------------------------------------------------------------------------
# Conda activation
# ---------------------------------------------------------------------------
activate_conda() {
    print_step "Activating conda environment 'mri_data'"

    for conda_sh in \
        "$HOME/anaconda3/etc/profile.d/conda.sh" \
        "$HOME/miniconda3/etc/profile.d/conda.sh" \
        "/opt/anaconda3/etc/profile.d/conda.sh" \
        "/opt/miniconda3/etc/profile.d/conda.sh" \
        "/opt/homebrew/anaconda3/etc/profile.d/conda.sh" \
        "/opt/homebrew/miniconda3/etc/profile.d/conda.sh" \
        "/usr/local/anaconda3/etc/profile.d/conda.sh" \
        "/usr/local/miniconda3/etc/profile.d/conda.sh"
    do
        if [ -f "$conda_sh" ]; then
            # shellcheck source=/dev/null
            source "$conda_sh"
            break
        fi
    done

    if ! conda activate mri_data 2>/dev/null; then
        print_err "Could not activate conda environment 'mri_data'"
        echo -e "${YELLOW}  Please activate it manually and re-run this script.${NC}"
        exit 1
    fi
    print_ok "conda environment 'mri_data' active"
}

# ---------------------------------------------------------------------------
# Flask helpers
# ---------------------------------------------------------------------------
flask_is_running() {
    curl -s --max-time 3 http://localhost:5001 > /dev/null 2>&1
}

wait_for_flask() {
    local attempts=0
    echo -n "  Waiting for Flask"
    while ! flask_is_running; do
        echo -n "."
        sleep 2
        attempts=$((attempts + 1))
        if [ $attempts -ge 30 ]; then
            echo ""
            print_err "Flask did not start within 60 seconds."
            echo "  Check logs: $FLASK_LOG"
            exit 1
        fi
    done
    echo ""
    print_ok "Flask is ready on http://localhost:5001"
}

start_flask() {
    print_step "Starting Flask API"

    if flask_is_running; then
        print_ok "Flask is already running — skipping start"
        WE_STARTED_FLASK=false
        return
    fi

    mkdir -p "$(dirname "$FLASK_LOG")"
    pushd "$FLASK_DIR" > /dev/null
    python app.py > "$FLASK_LOG" 2>&1 &
    FLASK_PID=$!
    popd > /dev/null
    WE_STARTED_FLASK=true
    print_ok "Flask launched (PID $FLASK_PID)  log → $FLASK_LOG"
    wait_for_flask
}

# ---------------------------------------------------------------------------
# Ask: images per class
# ---------------------------------------------------------------------------
ask_images_per_class() {
    print_step "Images per class"
    echo ""
    echo -e "  How many images per class should be sent to the API?"
    echo -e "  (4 classes × N images = total API calls)"
    echo -e "  ${YELLOW}Note: each call runs XAI + LLM, so budget ~2–5 min/image.${NC}"
    echo ""
    while true; do
        read -r -p "  Enter images per class [default: 10]: " IMAGES_PER_CLASS
        IMAGES_PER_CLASS="${IMAGES_PER_CLASS:-10}"
        if [[ "$IMAGES_PER_CLASS" =~ ^[1-9][0-9]*$ ]]; then
            local total=$(( IMAGES_PER_CLASS * 4 ))
            print_ok "Using $IMAGES_PER_CLASS images per class  ($total total API calls)"
            break
        else
            print_warn "Please enter a positive integer."
        fi
    done
}

# ---------------------------------------------------------------------------
# Ask: which models to run
# ---------------------------------------------------------------------------
ask_models() {
    print_step "Model selection"
    echo ""
    echo -e "  Available models:"
    echo ""
    for i in "${!ALL_MODELS[@]}"; do
        printf "    %2d)  %s\n" "$((i+1))" "${ALL_MODELS[$i]}"
    done
    echo ""
    echo -e "  Type  ${BOLD}all${NC}  to run all 12 models."
    echo -e "  Or enter numbers separated by spaces or commas  (e.g. 1 3 5  or  1,3,5)."
    echo ""

    while true; do
        read -r -p "  Your choice: " MODEL_INPUT
        MODEL_INPUT="${MODEL_INPUT:-all}"
        MODEL_INPUT="$(echo "$MODEL_INPUT" | tr '[:upper:]' '[:lower:]')"   # lowercase (bash 3 compat)

        SELECTED_MODELS=()

        if [ "$MODEL_INPUT" = "all" ]; then
            SELECTED_MODELS=("${ALL_MODELS[@]}")
            break
        fi

        # Normalise commas → spaces, then split
        MODEL_INPUT="$(echo "$MODEL_INPUT" | tr ',' ' ')"
        valid=true
        for token in $MODEL_INPUT; do
            if [[ "$token" =~ ^[0-9]+$ ]]; then
                idx=$(( token - 1 ))
                if [ "$idx" -ge 0 ] && [ "$idx" -lt "${#ALL_MODELS[@]}" ]; then
                    SELECTED_MODELS+=("${ALL_MODELS[$idx]}")
                else
                    print_warn "Number $token is out of range (1–${#ALL_MODELS[@]})."
                    valid=false
                    break
                fi
            else
                print_warn "Unrecognised input: '$token'.  Use numbers or 'all'."
                valid=false
                break
            fi
        done

        if [ "$valid" = true ] && [ "${#SELECTED_MODELS[@]}" -gt 0 ]; then
            break
        elif [ "${#SELECTED_MODELS[@]}" -eq 0 ]; then
            print_warn "No models selected. Please try again."
        fi
    done

    echo ""
    echo -e "  ${GREEN}Selected models (${#SELECTED_MODELS[@]}):${NC}"
    for m in "${SELECTED_MODELS[@]}"; do
        echo -e "    • $m"
    done

    # Time estimate
    local total_calls=$(( IMAGES_PER_CLASS * 4 * ${#SELECTED_MODELS[@]} ))
    local est_min=$(( total_calls * 3 ))   # rough 3 min/call
    echo ""
    echo -e "  ${YELLOW}Estimated total API calls : $total_calls${NC}"
    echo -e "  ${YELLOW}Rough time estimate        : ~${est_min} min (varies by hardware/LLM)${NC}"
    echo ""
    read -r -p "  Proceed? [Y/n]: " CONFIRM
    CONFIRM="${CONFIRM:-Y}"
    if [[ ! "$CONFIRM" =~ ^[Yy]$ ]]; then
        echo "  Aborted."
        exit 0
    fi
}

# ---------------------------------------------------------------------------
# Ask: clear previous CSV?
# ---------------------------------------------------------------------------
ask_clear_csv() {
    if [ ! -f "$OUTPUT_CSV" ]; then
        return
    fi

    existing=$(( $(wc -l < "$OUTPUT_CSV") - 1 ))
    echo ""
    print_warn "Existing CSV found with $existing rows: $OUTPUT_CSV"
    echo ""
    echo -e "  Options:"
    echo -e "    1)  Append  — add new results to the existing file"
    echo -e "    2)  Clear   — delete existing file and start fresh"
    echo ""
    while true; do
        read -r -p "  Your choice [1/2, default: 1]: " CSV_CHOICE
        CSV_CHOICE="${CSV_CHOICE:-1}"
        case "$CSV_CHOICE" in
            1)
                print_ok "Appending to existing CSV"
                break
                ;;
            2)
                rm "$OUTPUT_CSV"
                print_ok "Cleared existing CSV"
                break
                ;;
            *)
                print_warn "Please enter 1 or 2."
                ;;
        esac
    done
}

# ---------------------------------------------------------------------------
# Collection loop
# ---------------------------------------------------------------------------
run_collection() {
    print_header "PHASE 1 — Collecting LLM Outputs"

    local total_models="${#SELECTED_MODELS[@]}"
    local model_idx=0

    for model in "${SELECTED_MODELS[@]}"; do
        model_idx=$(( model_idx + 1 ))
        echo ""
        echo -e "${CYAN}  ── Model ${model_idx}/${total_models}: ${BOLD}${model}${NC}"
        echo ""

        python "$COLLECT_SCRIPT" \
            --model "$model" \
            --images-per-class "$IMAGES_PER_CLASS"

        local exit_code=$?
        if [ $exit_code -ne 0 ]; then
            print_warn "collect_llm_outputs.py exited with code $exit_code for model '$model'"
            print_warn "Continuing with next model..."
        fi
    done

    echo ""
    print_ok "Collection phase complete → $OUTPUT_CSV"
}

# ---------------------------------------------------------------------------
# Validation benchmark
# ---------------------------------------------------------------------------
run_benchmark() {
    print_header "PHASE 2 — LLM Validation Benchmark"
    echo ""

    python "$BENCHMARK_SCRIPT" --csv "$OUTPUT_CSV"

    local exit_code=$?
    if [ $exit_code -ne 0 ]; then
        print_err "Benchmark script failed (exit code $exit_code)"
        exit 1
    fi

    print_ok "Benchmark complete → $BENCHMARK_TXT"
}

# ---------------------------------------------------------------------------
# Show summary
# ---------------------------------------------------------------------------
show_summary() {
    print_header "DONE"
    echo ""
    echo -e "  ${GREEN}${BOLD}Pipeline completed successfully.${NC}"
    echo ""
    echo -e "  Output files:"
    echo -e "    CSV  : $OUTPUT_CSV"
    echo -e "    TXT  : $BENCHMARK_TXT"
    echo -e "    JSON : $SCRIPT_DIR/results/"
    echo ""

    if command -v wc > /dev/null 2>&1 && [ -f "$OUTPUT_CSV" ]; then
        local rows=$(( $(wc -l < "$OUTPUT_CSV") - 1 ))
        echo -e "  Rows in CSV : $rows"
    fi
    echo ""
    echo -e "  ${CYAN}To view the benchmark report:${NC}"
    echo -e "    cat '$BENCHMARK_TXT'"
    echo -e "    open '$BENCHMARK_TXT'   # macOS — opens in TextEdit"
    echo ""
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
print_header "LLM BENCHMARK PIPELINE"
echo ""
echo -e "  Project : $PROJECT_ROOT"
echo -e "  Scripts : $SCRIPT_DIR"

activate_conda
ask_images_per_class
ask_models
ask_clear_csv
start_flask
run_collection
run_benchmark
show_summary
