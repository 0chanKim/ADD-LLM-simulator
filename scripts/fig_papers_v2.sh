#!/bin/bash

start_time=$(date +%s)

# Clear previous results
mkdir -p results
> results/papers_v2_res.csv

echo "=== Running multi-paper accelerator comparison (v2: add_moe_core_32_v2) ==="
python run_papers_v2.py --synth_csv ./params/synth/systolic_array_synth_papers_2.csv

end_time=$(date +%s)
duration=$((end_time - start_time))
echo "Simulation time: $duration seconds."

echo "=== Generating visualization ==="
python EnergyAll_papers_v2.py

echo "=== Done ==="
echo "Results: ./results/papers_v2_res.csv"
echo "Chart:   ./results/fig_papers_v2.pdf"
