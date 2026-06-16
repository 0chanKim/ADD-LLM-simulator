"""
run_phases.py --- Prefill / Decode phase token/s measurement for ADD-LLM

Measures per-phase throughput (token/s) for Llama-2 7B on ADD-LLM variants:
  - Decode phase:  seq_len=1, no attention score matmuls (standard decode)
  - Prefill phase: seq_len=512, includes QK^T and SV attention matmuls

Accelerators evaluated (ADD-LLM only):
  addllm_s_v2  (32x32, 28nm, 1 GHz)
  addllm_l_v2  (64x64, 28nm, 1 GHz)

Output: results/phases_res.csv
"""

import pandas
import os
import numpy as np
import AxCore.src.benchmarks.benchmarks as benchmarks
from AxCore.src.simulator.stats import Stats
from AxCore.src.simulator.simulator import AxCoreSimulator
from AxCore.src.sweep.sweep import check_pandas_or_run_ax

PREFILL_SEQ_LEN = 512

sim_sweep_columns = [
    'N', 'M',
    'Max Precision (bits)', 'Min Precision (bits)',
    'Network', 'Layer',
    'Cycles', 'Memory wait cycles',
    'WBUF Read', 'WBUF Write',
    'OBUF Read', 'OBUF Write',
    'IBUF Read', 'IBUF Write',
    'DRAM Read', 'DRAM Write',
    'Bandwidth (bits/cycle)',
    'WBUF Size (bits)', 'OBUF Size (bits)', 'IBUF Size (bits)',
    'Batch size',
]

results_dir = './results'
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

accel_configs = [
    ('addllm_s_v2', 'params/conf/conf_addllm_s_v2.ini', './params/synth/systolic_array_synth_addllms.csv'),
    ('addllm_l_v2', 'params/conf/conf_addllm_l_v2.ini', './params/synth/systolic_array_synth_addllms.csv'),
]

phase_benches = {
    'decode':  'llama2_7b',
    'prefill': 'llama2_7b_prefill_512',
}

def df_to_stats(df):
    stats = Stats()
    stats.total_cycles    = float(df['Cycles'].iloc[0])
    stats.mem_stall_cycles = float(df['Memory wait cycles'].iloc[0])
    stats.reads['act']  = float(df['IBUF Read'].iloc[0])
    stats.reads['out']  = float(df['OBUF Read'].iloc[0])
    stats.reads['wgt']  = float(df['WBUF Read'].iloc[0])
    stats.reads['dram'] = float(df['DRAM Read'].iloc[0])
    stats.writes['act']  = float(df['IBUF Write'].iloc[0])
    stats.writes['out']  = float(df['OBUF Write'].iloc[0])
    stats.writes['wgt']  = float(df['WBUF Write'].iloc[0])
    stats.writes['dram'] = float(df['DRAM Write'].iloc[0])
    return stats

rows = []

for accel_name, config_file, synth_csv in accel_configs:
    print(f"\n=== Simulating {accel_name} ===")
    sim = AxCoreSimulator(config_file, synth_csv=synth_csv, verbose=False)
    frequency = sim.config.getint('accelerator', 'frequency')

    for phase, bench_name in phase_benches.items():
        print(f"  Phase: {phase}  ({bench_name})")
        csv_name = f'phases_{accel_name}_{phase}.csv'
        sweep_csv = os.path.join(results_dir, csv_name)
        sweep_df = pandas.DataFrame(columns=sim_sweep_columns)

        results = check_pandas_or_run_ax(
            sim, sweep_df, sweep_csv,
            batch_size=1,
            bench_type='axcore',
            weight_stationary=True,
            list_bench=[bench_name],
        )
        results = results.groupby('Network', as_index=False).agg(np.sum)

        stats = df_to_stats(results.loc[results['Network'] == bench_name])
        total_cycles = stats.total_cycles

        if phase == 'decode':
            seq_len = 1
        else:
            seq_len = PREFILL_SEQ_LEN

        tok_per_sec = seq_len * frequency / total_cycles

        rows.append({
            'Accelerator': accel_name,
            'Phase': phase,
            'seq_len': seq_len,
            'total_cycles': int(total_cycles),
            'token_per_sec': tok_per_sec,
        })

        print(f"    total_cycles = {total_cycles:,.0f}")
        print(f"    token/s      = {tok_per_sec:.2f}")

# --- Save CSV ---
res_csv_path = os.path.join(results_dir, 'phases_res.csv')
res_df = pandas.DataFrame(rows)
res_df.to_csv(res_csv_path, index=False)
print(f"\nResults saved to {res_csv_path}")

# --- Print summary table ---
print("\n" + "=" * 70)
print("Prefill / Decode Phase Throughput — Llama-2 7B on ADD-LLM")
print("=" * 70)
print(f"{'Accelerator':<14} {'Phase':<10} {'seq_len':<10} {'Cycles':>16} {'token/s':>12}")
print("-" * 70)
for row in rows:
    print(f"{row['Accelerator']:<14} {row['Phase']:<10} {row['seq_len']:<10} "
          f"{row['total_cycles']:>16,} {row['token_per_sec']:>12.2f}")
print("=" * 70)
