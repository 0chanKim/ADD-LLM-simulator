"""
run_comparison.py --- Multi-paper accelerator energy/performance comparison

Compares 7 accelerators:
  addllm_s   (ADD-LLM-S v2, Ours, 32x32, 28nm)
  addllm_l   (ADD-LLM-L, Ours, 64x64, 28nm)
  daypq      (DayPQ, TVLSI'26, 32x16, 22nm)
  bitmod     (BitMoD, ToC'25, 32x32, 28nm)
  m2xfp      (M2XFP, ASPLOS'26, 32x32, 28nm, 500MHz)
  amove      (AMove, 32x32, 28nm, 500MHz)

Baseline: daypq
"""

import pandas
import configparser
import os
import numpy as np
import AxCore.src.benchmarks.benchmarks as benchmarks
from AxCore.src.simulator.stats import Stats
from AxCore.src.simulator.simulator import AxCoreSimulator
from AxCore.src.sweep.sweep import check_pandas_or_run_ax
from bitfusion.src.utils.utils import *
from bitfusion.src.optimizer.optimizer import optimize_for_order, get_stats_fast
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--synth_csv', type=str,
                    default='./params/synth/systolic_array_synth_papers_2.csv',
                    help='Path to multi-paper synthesis results csv')
args = parser.parse_args()

synth_filename = os.path.basename(args.synth_csv)
config_name = os.path.splitext(synth_filename)[0]

def df_to_stats(df):
    stats = Stats()
    stats.total_cycles = float(df['Cycles'].iloc[0])
    stats.mem_stall_cycles = float(df['Memory wait cycles'].iloc[0])
    stats.reads['act'] = float(df['IBUF Read'].iloc[0])
    stats.reads['out'] = float(df['OBUF Read'].iloc[0])
    stats.reads['wgt'] = float(df['WBUF Read'].iloc[0])
    stats.reads['dram'] = float(df['DRAM Read'].iloc[0])
    stats.writes['act'] = float(df['IBUF Write'].iloc[0])
    stats.writes['out'] = float(df['OBUF Write'].iloc[0])
    stats.writes['wgt'] = float(df['WBUF Write'].iloc[0])
    stats.writes['dram'] = float(df['DRAM Write'].iloc[0])
    return stats

sim_sweep_columns = ['N', 'M',
        'Max Precision (bits)', 'Min Precision (bits)',
        'Network', 'Layer',
        'Cycles', 'Memory wait cycles',
        'WBUF Read', 'WBUF Write',
        'OBUF Read', 'OBUF Write',
        'IBUF Read', 'IBUF Write',
        'DRAM Read', 'DRAM Write',
        'Bandwidth (bits/cycle)',
        'WBUF Size (bits)', 'OBUF Size (bits)', 'IBUF Size (bits)',
        'Batch size']

batch_size = 32

results_dir = './results'
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

# --- Accelerator configurations ---
accel_configs = [
    ('addllm_s', 'params/conf/conf_addllm_s.ini', 'papers_v2_addllm_s.csv'),
    ('addllm_l', 'params/conf/conf_addllm_l.ini', 'papers_v2_addllm_l.csv'),
    ('daypq',    'params/conf/conf_daypq.ini',    'papers_v2_daypq.csv'),
    ('bitmod',   'params/conf/conf_bitmod.ini',   'papers_v2_bitmod.csv'),
    ('m2xfp',    'params/conf/conf_m2xfp.ini',   'papers_v2_m2xfp.csv'),
    ('amove',    'params/conf/conf_amove.ini',    'papers_v2_amove.csv'),
    ('tender',   'params/conf/conf_tender.ini',   'papers_v2_tender.csv'),
]

# Exclude these benchmarks
skip_benchmarks = {'opt_13b', 'opt_30b'}
active_benchlist = [b for b in benchmarks.benchlist if b not in skip_benchmarks]

all_cycles = {}
all_energy = {}
all_area = {}   # core area in mm^2

for accel_name, config_file, csv_name in accel_configs:
    print(f"=== Simulating {accel_name} ===")
    sim = AxCoreSimulator(config_file, synth_csv=args.synth_csv, verbose=False)
    sweep_csv = os.path.join(results_dir, csv_name)
    sweep_df = pandas.DataFrame(columns=sim_sweep_columns)
    results = check_pandas_or_run_ax(sim, sweep_df, sweep_csv,
                                     batch_size=batch_size,
                                     bench_type='axcore',
                                     weight_stationary=True)
    results = results.groupby('Network', as_index=False).agg(np.sum)

    cycles_list = []
    energy_list = []
    for name in active_benchlist:
        stats = df_to_stats(results.loc[results['Network'] == name])
        cycles_list.append(stats.total_cycles)
        energy_list.append(stats.get_energy_breakdown(sim.get_energy_cost()))

    # DayPQ: halve latency (cycles /2, Core/Static energy /2)
    if accel_name == 'daypq':
        cycles_list = [c / 2 for c in cycles_list]
        energy_list = [[s / 2, dram, buf, core / 2]
                       for s, dram, buf, core in energy_list]

    all_cycles[accel_name] = cycles_list
    all_energy[accel_name] = energy_list
    # Read core area from synth CSV (per-PE um^2 * N*M -> mm^2)
    synth_df = pandas.read_csv(args.synth_csv)
    row = synth_df[synth_df['Module'] == accel_name]
    pe_area_um2 = float(row['Area (um^2)'].iloc[0])
    n_pe = sim.accelerator.N * sim.accelerator.M
    all_area[accel_name] = pe_area_um2 * n_pe * 1e-6  # mm^2

# --- Model name mapping ---
model_name_dict = {
    'opt_125m': 'Opt125M', 'opt_350m': 'Opt350M',
    'opt_1_3b': 'Opt1.3B', 'opt_2_7b': 'Opt2.7B',
    'opt_6_7b': 'Opt6.7B', 'opt_13b': 'Opt13B',
    'opt_30b': 'Opt30B', 'opt_66b': 'Opt66B',
    'llama2_7b': 'LLama2-7B', 'llama3_8b': 'LLama3-8B',
}

# Display order
display_order = ['daypq', 'bitmod', 'm2xfp', 'amove', 'tender', 'addllm_s', 'addllm_l']
display_labels = {
    'addllm_s': 'ADD-LLM-S',
    'addllm_l': 'ADD-LLM-L',
    'daypq':    'DayPQ',
    'bitmod':   'BitMoD',
    'm2xfp':    'M2XFP',
    'amove':    'AMove',
    'tender':   'Tender',
}

# --- Write results CSV ---
res_csv_path = os.path.join(results_dir, 'comparison_res.csv')
ff = open(res_csv_path, "w")
ff.write(f"Configuration: {config_name}\n")

# Baseline = daypq
baseline_name = 'daypq'

# --- Normalized Cycles ---
wr_model = "Time, "
wr_bench = ", "
wr_line = ", "
for i, bench_name in enumerate(active_benchlist):
    model_label = model_name_dict.get(bench_name, bench_name)
    wr_model += model_label + ", " * len(display_order)
    for accel_name in display_order:
        wr_bench += display_labels[accel_name] + ", "
        cyc_norm = all_cycles[accel_name][i] / all_cycles[baseline_name][i]
        wr_line += "%.2f, " % cyc_norm

# Geomean
wr_model += "Geomean, " * len(display_order) + "\n"
for accel_name in display_order:
    wr_bench += display_labels[accel_name] + ", "
    mean_cyc = np.mean([all_cycles[accel_name][j] / all_cycles[baseline_name][j]
                        for j in range(len(active_benchlist))])
    wr_line += "%.2f, " % mean_cyc
wr_bench += "\n"
wr_line += "\n"

ff.write(wr_model)
ff.write(wr_bench)
ff.write(wr_line)

# --- Normalized Energy Breakdown ---
ff.write("\n")
ff.write(wr_model)
ff.write(wr_bench)

component_names = ['Static', 'Dram', 'Buffer', 'Core']
for comp_idx, comp_name in enumerate(component_names):
    wr_line = comp_name + ", "
    for i, bench_name in enumerate(active_benchlist):
        baseline_total = sum(all_energy[baseline_name][i])
        for accel_name in display_order:
            val = all_energy[accel_name][i][comp_idx] / baseline_total
            wr_line += "%.4f, " % val

    # Geomean for each accelerator
    for accel_name in display_order:
        mean_val = np.mean([all_energy[accel_name][j][comp_idx] / sum(all_energy[baseline_name][j])
                           for j in range(len(active_benchlist))])
        wr_line += "%.4f, " % mean_val
    wr_line += "\n"
    ff.write(wr_line)

# --- Normalized TOPS/Area ---
ff.write("\n")
ff.write(wr_model)
ff.write(wr_bench)

wr_line = "TOPS/Area, "
baseline_area = all_area[baseline_name]
for i, bench_name in enumerate(active_benchlist):
    baseline_ac = baseline_area * all_cycles[baseline_name][i]
    for accel_name in display_order:
        accel_ac = all_area[accel_name] * all_cycles[accel_name][i]
        val = baseline_ac / accel_ac if accel_ac != 0 else 1.0
        wr_line += "%.4f, " % val

# Geomean
for accel_name in display_order:
    mean_val = np.mean([baseline_area * all_cycles[baseline_name][j] /
                        (all_area[accel_name] * all_cycles[accel_name][j])
                        for j in range(len(active_benchlist))])
    wr_line += "%.4f, " % mean_val
wr_line += "\n"
ff.write(wr_line)

ff.close()
print(f"\nResults saved to {res_csv_path}")

# --- Print summary table ---
print("\n" + "=" * 90)
print("Multi-Paper Accelerator Energy/Performance Summary (v2: add_moe_core_32_v2)")
print("=" * 90)
print(f"{'Accelerator':<14} {'Paper':<16} {'Tech':<6} {'Array':<8} {'Freq':<8} {'Area(mm2)':<10} {'Power(mW)'}")
print("-" * 90)

accel_info = {
    'addllm_s': ('Ours v2',     '28nm', '32x32', '1GHz',   0.884,  300.2),
    'addllm_l': ('Ours v2',     '28nm', '64x64', '1GHz',   3.841, 1252.3),
    'daypq':    ("TVLSI'26",    '22nm', '32x16', '1GHz',   3.89,  2462.9),
    'bitmod':   ("ToC'25",      '28nm', '32x32', '1GHz',   1.66,   629.76),
    'm2xfp':    ("ASPLOS'26",   '28nm', '32x32', '500MHz', 1.051,  204.02),
    'amove':    ('AMove',       '28nm', '32x32', '500MHz', 2.507,  570.20),
    'tender':   ("ISCA'24",     '28nm', '64x64', '1GHz',   3.98,  1600.0),
}

for accel_name in display_order:
    paper, tech, array, freq, area, power = accel_info[accel_name]
    label = display_labels[accel_name]
    print(f"{label:<14} {paper:<16} {tech:<6} {array:<8} {freq:<8} {area:<10.3f} {power:.2f}")

print(f"\nBaseline: {display_labels[baseline_name]}")
print("All configs: W4A16, Leakage=0, SRAM=128KB each")
print("addllm_s: add_moe_core_32_v2 (area=863 um^2/PE, dynamic=293117 nW/PE)")

# Print per-benchmark energy breakdown
print("\n" + "-" * 90)
print(f"Normalized Energy (relative to {display_labels[baseline_name]} total energy)")
print("-" * 90)
for i, bench_name in enumerate(active_benchlist):
    model_label = model_name_dict.get(bench_name, bench_name)
    print(f"\n  {model_label}:")
    baseline_total = sum(all_energy[baseline_name][i])
    for accel_name in display_order:
        total_norm = sum(all_energy[accel_name][i]) / baseline_total
        core_norm = all_energy[accel_name][i][3] / baseline_total
        label = display_labels[accel_name]
        accel_ac = all_area[accel_name] * all_cycles[accel_name][i]
        baseline_ac = all_area[baseline_name] * all_cycles[baseline_name][i]
        topsarea = baseline_ac / accel_ac if accel_ac != 0 else 1.0
        print(f"    {label:<14} Total={total_norm:.3f}  Core={core_norm:.3f}  TOPS/W={1/total_norm:.2f}x  TOPS/Area={topsarea:.2f}x")

print("\n" + "=" * 90)
