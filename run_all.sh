#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════
#  Q-RLSTC: Run All — Tests → Smoke Matrix → Full Experiments
# ═══════════════════════════════════════════════════════════════════
#
#  Usage:
#    ./run_all.sh                    # full pipeline (tests + smoke + experiments)
#    ./run_all.sh tests              # unit + integration tests only (~2s)
#    ./run_all.sh smoke              # tests + smoke matrix (~5 min)
#    ./run_all.sh experiments        # tests + full thesis experiments (~2-4h)
#    ./run_all.sh --fast             # tests + smoke with --amount 50 (~2 min)
#
#  Environment:
#    AMOUNT    — trajectory count (default: 500 experiments, 100 smoke)
#    EPOCHS    — training epochs (default: 3 experiments, 1 smoke)
#    SEEDS     — comma-separated seeds (default: 42,7)

set -euo pipefail

cd "$(dirname "$0")"

PYTHON=".venv/bin/python"
MODE="${1:-all}"
SMOKE_AMOUNT="${AMOUNT:-100}"
EXP_AMOUNT="${AMOUNT:-500}"
EXP_EPOCHS="${EPOCHS:-3}"
SEEDS="${SEEDS:-42,7}"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

banner() { echo -e "\n${CYAN}═══════════════════════════════════════════════════${NC}"; echo -e "${CYAN}  $1${NC}"; echo -e "${CYAN}═══════════════════════════════════════════════════${NC}\n"; }
pass()   { echo -e "${GREEN}✓ $1${NC}"; }
fail()   { echo -e "${RED}✗ $1${NC}"; exit 1; }
info()   { echo -e "${YELLOW}→ $1${NC}"; }

# ── Fast mode override ────────────────────────────────────────
if [[ "$MODE" == "--fast" ]]; then
    MODE="smoke"
    SMOKE_AMOUNT=50
fi

# ── Step 1: Tests ─────────────────────────────────────────────
run_tests() {
    banner "Step 1/3 — Unit + Integration Tests"
    $PYTHON -m pytest tests/ -v --tb=short || fail "Tests failed — aborting."
    pass "All tests passed"
}

# ── Step 2: Smoke Matrix ─────────────────────────────────────
run_smoke() {
    banner "Step 2/3 — Smoke Matrix (${SMOKE_AMOUNT} trajectories, 1 epoch)"
    $PYTHON experiments/smoke_matrix.py \
        --amount "$SMOKE_AMOUNT" \
        --seeds "$SEEDS" || fail "Smoke matrix failed — check results/smoke/smoke_matrix.json"
    pass "Smoke matrix passed — report at results/smoke/smoke_matrix.json"
}

# ── Step 3: Full Experiments ──────────────────────────────────
run_experiments() {
    banner "Step 3/3 — Full Thesis Experiments (${EXP_AMOUNT} trajectories, ${EXP_EPOCHS} epochs)"
    info "This may take 2-4 hours..."
    $PYTHON experiments/run_thesis_experiments.py \
        --amount "$EXP_AMOUNT" \
        --epochs "$EXP_EPOCHS" \
        --seeds "$SEEDS" || fail "Experiments failed"
    pass "All experiments completed — results in results/thesis/"
}

# ── Dispatch ──────────────────────────────────────────────────
case "$MODE" in
    tests)
        run_tests
        ;;
    smoke)
        run_tests
        run_smoke
        ;;
    experiments)
        run_tests
        run_experiments
        ;;
    all)
        run_tests
        run_smoke
        run_experiments
        ;;
    *)
        echo "Usage: $0 {tests|smoke|experiments|all|--fast}"
        exit 1
        ;;
esac

echo ""
pass "Done."
