#!/bin/bash

start_time=$(date +%s)

# Clear previous results
mkdir -p results
> results/comparison_res.csv

echo "=== Running multi-paper accelerator comparison ==="
python run_comparison.py --synth_csv ./params/synth/systolic_array_synth_papers_2.csv

end_time=$(date +%s)
duration=$((end_time - start_time))
echo "Simulation time: $duration seconds."

echo "=== Generating visualization ==="
python EnergyAll_comparison.py

echo "=== Done ==="
echo "Results: ./results/comparison_res.csv"
echo "Chart:   ./results/fig_comparison.pdf"
