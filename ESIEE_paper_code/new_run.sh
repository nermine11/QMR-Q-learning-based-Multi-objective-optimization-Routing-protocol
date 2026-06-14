#!/bin/bash
#  Run fixed & dynamic QMR, then plot the comparison.

# ---------- Experiment parameters ----------
MODE1="fixed"
MODE2="dynamic"
MIN_NODES=1
MAX_NODES=49
STEP=1
WIDTH=500
LENGTH=500
RANGE=200
PACKETS=1000
CBR_INTERVAL=200
BASE_SEED=42

# ---------- Run the first experiment ----------
echo "=========================================="
echo "Running $MODE1 ω experiment..."
python3 new_experiment.py \
    --mode "$MODE1" \
    --min-nodes "$MIN_NODES" --max-nodes "$MAX_NODES" --step "$STEP" \
    --width "$WIDTH" --length "$LENGTH" --range "$RANGE" \
    --packets "$PACKETS" --cbr-interval "$CBR_INTERVAL" \
    --seed "$BASE_SEED"
echo ""

# ---------- Run the second experiment ----------
echo "=========================================="
echo "Running $MODE2 ω experiment..."
python3 new_experiment.py \
    --mode "$MODE2" \
    --min-nodes "$MIN_NODES" --max-nodes "$MAX_NODES" --step "$STEP" \
    --width "$WIDTH" --length "$LENGTH" --range "$RANGE" \
    --packets "$PACKETS" --cbr-interval "$CBR_INTERVAL" \
    --seed "$BASE_SEED"
echo ""

# ---------- Plot the comparison ----------
echo "=========================================="
echo "Generating comparison graph..."
python3 plot_comparaison.py
echo "Done. See comparison_lifetime.png"