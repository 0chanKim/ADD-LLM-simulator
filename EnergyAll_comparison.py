"""
EnergyAll_comparison.py --- Visualization for multi-paper accelerator comparison
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Times New Roman'
plt.rcParams['mathtext.it'] = 'Times New Roman:italic'
plt.rcParams['mathtext.bf'] = 'Times New Roman:bold'

font_config = {
    'title': 28,
    'axis_label': 26,
    'legend': 22,
    'tick_label': 20,
    'bar_label': 16,
    'group_label': 22,
}

# All accelerators in CSV order (must match run_papers_v2.py display_order)
all_categories = ['DayPQ', 'BitMoD', 'M2XFP', 'AMove', 'Tender', 'ADD-LLM-S', 'ADD-LLM-L']
baseline_idx = 0  # DayPQ
ours_labels = ['ADD-LLM-S', 'ADD-LLM-L']


def load_papers_data():
    """Load energy data from comparison_res.csv"""
    with open('./results/comparison_res.csv', 'r') as f:
        lines = f.readlines()

    # Find blank-line-separated sections
    sections = []
    current_start = None
    for i, line in enumerate(lines):
        if line.strip() == '':
            if current_start is not None:
                sections.append((current_start, i))
            current_start = i + 1
        elif current_start is None:
            current_start = 0
    if current_start is not None:
        sections.append((current_start, len(lines)))

    # Section 0: Normalized Cycles (skip)
    # Section 1: Normalized Energy Breakdown (4 component lines)
    # Section 2: TOPS/Area (1 line)

    n_accel = len(all_categories)

    # --- Parse Energy Breakdown (section 1) ---
    energy_start = sections[1][0]
    data_start = energy_start + 2  # skip model name + bench name lines

    components = {}
    for comp_idx in range(4):
        line = lines[data_start + comp_idx].strip()
        parts = line.split(', ')
        comp_name = parts[0]
        values = [float(v.strip().rstrip(',')) for v in parts[1:] if v.strip().rstrip(',')]
        components[comp_name] = values

    n_benchmarks = (len(components['Static']) - n_accel) // n_accel

    # Reshape energy: [bench][accel][component]
    all_energy = []
    for bench_idx in range(n_benchmarks):
        bench_data = []
        for accel_idx in range(n_accel):
            val_idx = bench_idx * n_accel + accel_idx
            bench_data.append([
                components['Static'][val_idx],
                components['Dram'][val_idx],
                components['Buffer'][val_idx],
                components['Core'][val_idx],
            ])
        all_energy.append(bench_data)

    # Add geomean
    geomean_data = []
    geo_offset = n_benchmarks * n_accel
    for accel_idx in range(n_accel):
        val_idx = geo_offset + accel_idx
        geomean_data.append([
            components['Static'][val_idx],
            components['Dram'][val_idx],
            components['Buffer'][val_idx],
            components['Core'][val_idx],
        ])
    all_energy.append(geomean_data)

    # --- Parse TOPS/Area (section 2) ---
    topsarea_start = sections[2][0]
    topsarea_data_start = topsarea_start + 2  # skip model name + bench name lines
    topsarea_line = lines[topsarea_data_start].strip()
    topsarea_parts = topsarea_line.split(', ')
    topsarea_values = [float(v.strip().rstrip(',')) for v in topsarea_parts[1:] if v.strip().rstrip(',')]

    # Reshape TOPS/Area: [bench][accel]
    all_topsarea = []
    for bench_idx in range(n_benchmarks):
        bench_data = []
        for accel_idx in range(n_accel):
            val_idx = bench_idx * n_accel + accel_idx
            bench_data.append(topsarea_values[val_idx])
        all_topsarea.append(bench_data)

    # Geomean
    geomean_ta = []
    for accel_idx in range(n_accel):
        val_idx = n_benchmarks * n_accel + accel_idx
        geomean_ta.append(topsarea_values[val_idx])
    all_topsarea.append(geomean_ta)

    return np.array(all_energy), np.array(all_topsarea)


def compute_power_efficiency(energy_data):
    """Power Efficiency = baseline_energy / this_energy (higher=better)"""
    pe_data = []
    for bench_data in energy_data:
        baseline_total = sum(bench_data[baseline_idx])
        bench_pe = []
        for accel_data in bench_data:
            accel_total = sum(accel_data)
            ratio = baseline_total / accel_total if accel_total != 0 else 1.0
            bench_pe.append(ratio)
        pe_data.append(bench_pe)
    return np.array(pe_data)


def compute_sorted_order(data_geomean):
    """Ascending sort by Geomean, Ours always rightmost."""
    others_indices = []
    ours_indices = []
    for i, label in enumerate(all_categories):
        if label in ours_labels:
            ours_indices.append(i)
        else:
            others_indices.append(i)
    others_sorted = sorted(others_indices, key=lambda i: data_geomean[i])
    return others_sorted + ours_indices


# Load data
energy_data, topsarea_data = load_papers_data()
pe_data = compute_power_efficiency(energy_data)

# Sort each chart independently by its own Geomean
pe_sorted_order = compute_sorted_order(pe_data[-1])
ta_sorted_order = compute_sorted_order(topsarea_data[-1])

pe_categories = [all_categories[i] for i in pe_sorted_order]
ta_categories = [all_categories[i] for i in ta_sorted_order]

pe_data_sorted = pe_data[:, pe_sorted_order]
ta_data_sorted = topsarea_data[:, ta_sorted_order]

# Benchmark labels
bench_labels = ['OPT-125M', 'OPT-350M', 'OPT-1.3B', 'OPT-2.7B', 'OPT-6.7B',
                'LLama2-7B', 'LLama3-8B']
groups = bench_labels + ['Average']

# Colors and hatches
bar_colors = {
    'DayPQ':     '#78A7D8',
    'BitMoD':    '#A2C6EB',
    'M2XFP':     '#CCE5FE',
    'AMove':     '#B0C4B8',
    'Tender':    '#9B59B6',
    'ADD-LLM-S': '#E07B54',
    'ADD-LLM-L': '#C0392B',
}

hatch_map = {
    'DayPQ':     '\\\\',
    'BitMoD':    'xx',
    'M2XFP':     '//',
    'AMove':     '--',
    'Tender':    'oo',
    'ADD-LLM-S': '...',
    'ADD-LLM-L': '+++',
}

# Style
bar_width = 0.45
bar_spacing = 0.12
group_spacing = 1.4
margin = 0.4

n_cats = len(all_categories)

def calc_x_positions():
    x_positions = []
    offset = 0
    for _ in groups:
        x = np.arange(n_cats) * (bar_width + bar_spacing) + offset
        x_positions.append(x)
        offset += n_cats * (bar_width + bar_spacing) + group_spacing
    return x_positions

x_positions = calc_x_positions()
x_flat = np.concatenate(x_positions)

# --- Figure ---
fig = plt.figure(figsize=(28, 12))
gs = fig.add_gridspec(5, 1, height_ratios=[0.04, 0.04, 0.46, 0.04, 0.42])
legend_ax = fig.add_subplot(gs[0])
label_ax = fig.add_subplot(gs[1])
ax_pe = fig.add_subplot(gs[2])
space_ax = fig.add_subplot(gs[3])
ax_ta = fig.add_subplot(gs[4])

legend_ax.axis('off')
label_ax.axis('off')
space_ax.axis('off')


def plot_bars(ax, x_positions, data_sorted, categories, ylabel, show_xticks=False):
    """Plot grouped bar chart."""
    max_val = 0
    for g_idx, (x_group, bench_vals) in enumerate(zip(x_positions, data_sorted)):
        for cat_idx, (x_bar, val) in enumerate(zip(x_group, bench_vals)):
            cat_name = categories[cat_idx]
            max_val = max(max_val, val)
            ax.bar(x_bar, val, width=bar_width,
                   color=bar_colors.get(cat_name, '#78A7D8'),
                   hatch=hatch_map.get(cat_name, ''),
                   edgecolor='black', linewidth=0.8)
            ax.text(x_bar, val + 0.02, f'{val:.2f}x',
                    ha='center', va='bottom', fontweight='bold',
                    fontsize=font_config['bar_label'], rotation=90)

    ax.set_ylabel(ylabel, fontsize=font_config['axis_label'])
    ax.set_ylim(0, max_val * 1.35)
    ax.set_xlim(x_flat[0] - bar_width/2 - margin,
                x_flat[-1] + bar_width/2 + margin)
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    ax.tick_params(axis='y', labelsize=font_config['tick_label'])

    if show_xticks:
        ax.set_xticks(x_flat)
        ax.set_xticklabels([cat for _ in groups for cat in categories],
                           fontsize=font_config['tick_label'], rotation=90, ha='right')
        ax.tick_params(axis='x', length=0, pad=10)
    else:
        ax.set_xticks([])

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor('black')
        spine.set_linewidth(1.5)


# --- Top: Normalized TOPS/W ---
plot_bars(ax_pe, x_positions, pe_data_sorted, pe_categories, 'Normalized TOPS/W')

# --- Bottom: TOPS/Area ---
plot_bars(ax_ta, x_positions, ta_data_sorted, ta_categories, 'Normalized TOPS/Area', show_xticks=False)

# Group labels (below bottom chart)
for g_idx, (x_group, group_name) in enumerate(zip(x_positions, groups)):
    x_center = (x_group[0] + x_group[-1]) / 2
    y_min, y_max = ax_ta.get_ylim()
    ax_ta.text(x_center, y_min - (y_max - y_min) * 0.12, group_name,
               ha='center', va='top', fontsize=font_config['group_label'] - 4,
               fontweight='bold', fontstyle='italic')

# Group separator lines
for g_idx in range(len(groups) - 1):
    boundary = x_positions[g_idx][-1] + (bar_width + group_spacing + bar_spacing) / 2
    ax_pe.axvline(x=boundary, color='black', linewidth=1.5,
                  ymin=0, ymax=1, clip_on=False)
    ax_ta.axvline(x=boundary, color='black', linewidth=1.5,
                  ymin=0, ymax=1, clip_on=False)

# Legend (use PE chart order for legend)
legend_elements = []
for cat_name in pe_categories:
    patch = Patch(facecolor=bar_colors.get(cat_name, '#78A7D8'),
                  hatch=hatch_map.get(cat_name, ''),
                  edgecolor='black', linewidth=0.8, label=cat_name)
    legend_elements.append(patch)

legend_ax.legend(handles=legend_elements, loc='center', ncol=len(pe_categories),
                 frameon=True, fontsize=font_config['legend'],
                 labelspacing=0.5, handlelength=1.5, handletextpad=0.5,
                 edgecolor='black')

plt.tight_layout()
plt.subplots_adjust(hspace=0.08)

output_path = './results/fig_comparison.pdf'
plt.savefig(output_path, format='pdf', bbox_inches='tight', dpi=300)
print(f"Chart saved to: {output_path}")
