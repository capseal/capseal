#!/usr/bin/env bash
#
# run_parity_stability_test.sh
#
# Runs the redundant XOR-8 experiment across multiple seeds and verifies stability.
# This script tests whether ENN training produces consistent results across different
# random seeds, which is important for ensuring the model is robust.
#
# Usage:
#   ./scripts/run_parity_stability_test.sh
#
# Requirements:
#   - Python 3 with numpy
#   - Compiled bicep_to_enn binary in ./apps/
#
# Exit codes:
#   0 - All seeds passed (>= 95% test accuracy)
#   1 - One or more seeds failed

set -euo pipefail

# ============================================================================
# Configuration
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Seeds to test
SEEDS=(42 123 999)

# Success threshold
ACCURACY_THRESHOLD=0.95

# Training parameters
N_BITS=8
EPOCHS=2000
EVAL_EVERY=10
EARLY_STOP_PATIENCE=50
GRAD_CLIP=1.0
WEIGHT_DECAY=0.001

# Output directory
OUTPUT_DIR="/tmp/parity_stability_test_$$"

# Binary and script paths
BICEP_TO_ENN="${PROJECT_ROOT}/apps/bicep_to_enn"
GEN_PARITY_SCRIPT="${PROJECT_ROOT}/scripts/gen_parity_dataset.py"

# ============================================================================
# Helper Functions
# ============================================================================

log_info() {
    echo "[INFO] $(date '+%Y-%m-%d %H:%M:%S') - $*"
}

log_error() {
    echo "[ERROR] $(date '+%Y-%m-%d %H:%M:%S') - $*" >&2
}

log_warn() {
    echo "[WARN] $(date '+%Y-%m-%d %H:%M:%S') - $*" >&2
}

cleanup() {
    if [[ -d "${OUTPUT_DIR}" ]]; then
        log_info "Cleaning up temporary files in ${OUTPUT_DIR}"
        rm -rf "${OUTPUT_DIR}"
    fi
}

check_prerequisites() {
    local missing=0

    # Check for Python
    if ! command -v python3 &> /dev/null && ! command -v python &> /dev/null; then
        log_error "Python is not installed or not in PATH"
        missing=1
    fi

    # Check for numpy
    local python_cmd
    python_cmd=$(command -v python3 || command -v python)
    if ! "${python_cmd}" -c "import numpy" 2>/dev/null; then
        log_error "NumPy is not installed. Install with: pip install numpy"
        missing=1
    fi

    # Check for dataset generation script
    if [[ ! -f "${GEN_PARITY_SCRIPT}" ]]; then
        log_error "Dataset generation script not found: ${GEN_PARITY_SCRIPT}"
        missing=1
    fi

    # Check for bicep_to_enn binary
    if [[ ! -x "${BICEP_TO_ENN}" ]]; then
        log_error "bicep_to_enn binary not found or not executable: ${BICEP_TO_ENN}"
        log_error "Build it with: cd ${PROJECT_ROOT} && make"
        missing=1
    fi

    return ${missing}
}

get_python_cmd() {
    command -v python3 || command -v python
}

# ============================================================================
# Dataset Generation
# ============================================================================

generate_dataset() {
    local seed=$1
    local output_path=$2
    local python_cmd
    python_cmd=$(get_python_cmd)

    log_info "Generating ${N_BITS}-bit parity dataset with seed ${seed}"

    if ! "${python_cmd}" "${GEN_PARITY_SCRIPT}" \
        --n_bits "${N_BITS}" \
        --mode full \
        --seed "${seed}" \
        --output "${output_path}"; then
        log_error "Failed to generate dataset for seed ${seed}"
        return 1
    fi

    log_info "Dataset saved to: ${output_path}"
    return 0
}

# ============================================================================
# Training
# ============================================================================

run_training() {
    local seed=$1
    local data_path=$2
    local ckpt_path=$3
    local telemetry_path=$4

    log_info "Training ENN with seed ${seed}"

    # Check if early stopping/checkpoint features are available
    # by checking if the binary accepts the flags without error
    local has_early_stop=false
    if "${BICEP_TO_ENN}" --help 2>&1 | grep -q "save_best_ckpt"; then
        # The flag exists in help, but the actual implementation might not be complete
        # We check the training loop behavior - for now assume it's available
        has_early_stop=true
    fi

    local train_cmd=("${BICEP_TO_ENN}" "${data_path}"
        --predict_final_only
        --grad_clip "${GRAD_CLIP}"
        --weight_decay "${WEIGHT_DECAY}"
        --epochs "${EPOCHS}"
        --eval_every "${EVAL_EVERY}"
        --telemetry "${telemetry_path}")

    # Add early stopping flags if available
    # NOTE: These flags are parsed but may not be fully implemented yet.
    # The script will still work - it just won't use early stopping.
    if [[ "${has_early_stop}" == "true" ]]; then
        # TODO: Once early stopping is fully implemented, uncomment these:
        # train_cmd+=(--early_stop_patience "${EARLY_STOP_PATIENCE}")
        # train_cmd+=(--save_best_ckpt "${ckpt_path}")
        :
    fi

    log_info "Running: ${train_cmd[*]}"

    # Capture output for parsing
    local train_output
    if ! train_output=$("${train_cmd[@]}" 2>&1); then
        log_error "Training failed for seed ${seed}"
        echo "${train_output}"
        return 1
    fi

    # Save full output for debugging
    echo "${train_output}" > "${OUTPUT_DIR}/train_output_s${seed}.log"

    log_info "Training completed for seed ${seed}"
    return 0
}

# ============================================================================
# Result Parsing
# ============================================================================

parse_results() {
    local seed=$1
    local telemetry_path=$2

    # The telemetry file is the main predictions file, but we need the curves file
    local curves_path="${telemetry_path}.curves.csv"

    if [[ ! -f "${curves_path}" ]]; then
        log_error "Curves file not found: ${curves_path}"
        echo "0 0.0"  # Return defaults
        return 1
    fi

    # Parse the curves CSV to find best test accuracy and corresponding epoch
    # Format: epoch,train_loss,train_eval_loss,train_acc,test_eval_loss,test_acc
    local best_epoch=0
    local best_test_acc=0.0

    # Use awk to find the epoch with the best test accuracy
    local result
    result=$(awk -F',' 'NR>1 {
        if ($6 > best_acc) {
            best_acc = $6;
            best_epoch = $1
        }
    }
    END {
        printf "%d %.6f", best_epoch, best_acc
    }' "${curves_path}")

    echo "${result}"
}

# ============================================================================
# Main Execution
# ============================================================================

main() {
    log_info "=== PARITY STABILITY TEST ==="
    log_info "Testing ${#SEEDS[@]} seeds: ${SEEDS[*]}"
    log_info "Success threshold: ${ACCURACY_THRESHOLD}"

    # Check prerequisites
    if ! check_prerequisites; then
        log_error "Prerequisites check failed"
        exit 1
    fi

    # Create output directory
    mkdir -p "${OUTPUT_DIR}"
    log_info "Output directory: ${OUTPUT_DIR}"

    # Arrays to store results
    declare -a result_seeds
    declare -a result_epochs
    declare -a result_accs
    declare -a result_status

    local all_passed=true
    local pass_count=0
    local total_count=${#SEEDS[@]}

    # Process each seed
    for seed in "${SEEDS[@]}"; do
        log_info "----------------------------------------"
        log_info "Processing seed ${seed}"
        log_info "----------------------------------------"

        local data_path="${OUTPUT_DIR}/parity_s${seed}.csv"
        local ckpt_path="${OUTPUT_DIR}/best_s${seed}.ckpt"
        local telemetry_path="${OUTPUT_DIR}/curves_s${seed}.csv"

        # Generate dataset
        if ! generate_dataset "${seed}" "${data_path}"; then
            result_seeds+=("${seed}")
            result_epochs+=("-")
            result_accs+=("-")
            result_status+=("GEN_FAIL")
            all_passed=false
            continue
        fi

        # Run training
        if ! run_training "${seed}" "${data_path}" "${ckpt_path}" "${telemetry_path}"; then
            result_seeds+=("${seed}")
            result_epochs+=("-")
            result_accs+=("-")
            result_status+=("TRAIN_FAIL")
            all_passed=false
            continue
        fi

        # Parse results
        local parsed
        parsed=$(parse_results "${seed}" "${telemetry_path}")
        local best_epoch best_test_acc
        best_epoch=$(echo "${parsed}" | awk '{print $1}')
        best_test_acc=$(echo "${parsed}" | awk '{print $2}')

        # Determine pass/fail status
        local status
        if (( $(echo "${best_test_acc} >= ${ACCURACY_THRESHOLD}" | bc -l) )); then
            status="PASS"
            ((pass_count++))
        else
            status="FAIL"
            all_passed=false
        fi

        result_seeds+=("${seed}")
        result_epochs+=("${best_epoch}")
        result_accs+=("${best_test_acc}")
        result_status+=("${status}")

        log_info "Seed ${seed}: best_epoch=${best_epoch}, best_test_acc=${best_test_acc}, status=${status}"
    done

    # Print summary
    echo ""
    echo "=== PARITY STABILITY TEST RESULTS ==="
    echo "Seed  | Best Epoch | Best Test Acc | Status"
    echo "------|------------|---------------|--------"

    local min_epoch=999999
    local max_epoch=0

    for i in "${!result_seeds[@]}"; do
        local acc_display
        if [[ "${result_accs[$i]}" != "-" ]]; then
            # Convert to percentage
            acc_display=$(printf "%.2f%%" "$(echo "${result_accs[$i]} * 100" | bc -l)")

            # Update epoch range for passed tests
            if [[ "${result_status[$i]}" == "PASS" ]]; then
                local ep="${result_epochs[$i]}"
                if [[ "${ep}" =~ ^[0-9]+$ ]]; then
                    if ((ep < min_epoch)); then min_epoch=$ep; fi
                    if ((ep > max_epoch)); then max_epoch=$ep; fi
                fi
            fi
        else
            acc_display="-"
        fi

        printf "%-5s | %-10s | %-13s | %s\n" \
            "${result_seeds[$i]}" \
            "${result_epochs[$i]}" \
            "${acc_display}" \
            "${result_status[$i]}"
    done

    echo ""
    echo "OVERALL: ${pass_count}/${total_count} seeds achieved >=$(echo "${ACCURACY_THRESHOLD} * 100" | bc -l | sed 's/\.00$//')% test accuracy"

    if [[ "${pass_count}" -gt 1 ]] && [[ "${min_epoch}" -ne 999999 ]]; then
        echo "Epoch range: ${min_epoch}-${max_epoch} (stable)"
    fi

    # Print warnings and suggestions for failures
    if [[ "${all_passed}" != "true" ]]; then
        echo ""
        echo "=== WARNING: SOME SEEDS FAILED ==="
        echo ""
        echo "Suggestions for improving stability:"
        echo "  1. Increase epochs (current: ${EPOCHS})"
        echo "  2. Adjust learning rate (try 5e-4 or 2e-3)"
        echo "  3. Increase weight decay (current: ${WEIGHT_DECAY}, try 0.01)"
        echo "  4. Reduce gradient clipping threshold (current: ${GRAD_CLIP}, try 0.5)"
        echo "  5. Check for data issues in failed seeds"
        echo ""
        echo "Failed seeds' logs can be found in: ${OUTPUT_DIR}/train_output_s*.log"
        echo ""

        # Don't cleanup on failure so logs can be examined
        log_warn "Keeping output directory for debugging: ${OUTPUT_DIR}"
        exit 1
    fi

    echo ""
    log_info "All tests passed!"

    # Cleanup on success (optional - comment out to keep results)
    # cleanup

    log_info "Results saved in: ${OUTPUT_DIR}"
    exit 0
}

# Run main function
main "$@"
