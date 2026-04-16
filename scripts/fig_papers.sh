#!/bin/bash

start_time=$(date +%s)

# Clear previous results
mkdir -p results
> results/papers_res.csv

echo "=== Running multi-paper accelerator comparison ==="
python run_papers.py --synth_csv ./params/synth/systolic_array_synth_papers.csv

end_time=$(date +%s)
duration=$((end_time - start_time))
echo "Simulation time: $duration seconds."

echo "=== Generating visualization ==="
python EnergyAll_papers.py

echo "=== Done ==="
echo "Results: ./results/papers_res.csv"
echo "Chart:   ./results/fig_papers.pdf"
