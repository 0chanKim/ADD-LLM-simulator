#!/bin/bash

start_time=$(date +%s)

# Clear previous results
mkdir -p results
> results/addllms_res.csv

echo "=== Running ADD-LLM v1 vs v2 comparison ==="
python run_addllms.py --synth_csv ./params/synth/systolic_array_synth_addllms.csv

end_time=$(date +%s)
duration=$((end_time - start_time))
echo "Simulation time: $duration seconds."

echo "=== Generating visualization ==="
python EnergyAll_addllms.py

echo "=== Done ==="
echo "Results: ./results/addllms_res.csv"
echo "Chart:   ./results/fig_add_llms.pdf"
