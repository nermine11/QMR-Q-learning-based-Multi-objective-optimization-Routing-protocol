#!/bin/bash
# run_both_and_plot.sh – Run fixed & dynamic QMR, then plot the comparison.

# ---------- Experiment parameters ----------
MODE1="fixed"
MODE2="dynamic"
MIN_NODES=5
MAX_NODES=200
STEP=5
WIDTH=500
LENGTH=500
RANGE=200
PACKETS=100
SEEDS=20

# ---------- Run the first experiment ----------
echo "=========================================="
echo "Running $MODE1 ω experiment..."
python3 experiment.py \
    --mode "$MODE1" \
    --min-nodes "$MIN_NODES" --max-nodes "$MAX_NODES" --step "$STEP" \
    --width "$WIDTH" --length "$LENGTH" --range "$RANGE" \
    --packets "$PACKETS" --seeds "$SEEDS"
echo ""

# ---------- Run the second experiment ----------
echo "=========================================="
echo "Running $MODE2 ω experiment..."
python3 experiment.py \
    --mode "$MODE2" \
    --min-nodes "$MIN_NODES" --max-nodes "$MAX_NODES" --step "$STEP" \
    --width "$WIDTH" --length "$LENGTH" --range "$RANGE" \
    --packets "$PACKETS" --seeds "$SEEDS"
echo ""

# ---------- Plot the comparison ----------
echo "=========================================="
echo "Generating comparison graph..."
python3 plot_comparaison.py
echo "Done. See comparison_lifetime.png"