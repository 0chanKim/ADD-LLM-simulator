"""
papers_energy_consum.py --- Normalized Energy breakdown visualization

Reuses papers_res.csv (generated for multi-paper accelerator comparison) and
plots a stacked bar chart in the same visual style as EnergyAll.py:
  - Stacks: Static / Dram / Buffer / Core
  - Groups: 7 benchmarks + Average (geomean)
  - Accelerators per group: DayPQ / BitMoD / M2XFP / ADD-LLM-S / ADD-LLM-L
  - Normalization: each benchmark's components divided by that benchmark's
                   baseline (DayPQ) TOTAL energy, so DayPQ bar sums to 1.00.

Output: ./results/fig_papers_energy.pdf
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

# =============================================================================
# Font configuration (matches EnergyAll.py style)
# =============================================================================
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Times New Roman'
plt.rcParams['mathtext.it'] = 'Times New Roman:italic'
plt.rcParams['mathtext.bf'] = 'Times New Roman:bold'

font_config = {
    'title'      : 30,
    'axis_label' : 30,
    'legend'     : 28,
    'tick_label' : 22,
    'bar_label'  : 20,
    'group_label': 26,
}

# =============================================================================
# Fixed schema (must match run_papers.py display order)
# =============================================================================
all_categories = ['DayPQ', 'BitMoD', 'M2XFP', 'ADD-LLM-S', 'ADD-LLM-L']
baseline_idx   = 0   # DayPQ

bench_labels = ['OPT-125M', 'OPT-350M', 'OPT-1.3B', 'OPT-2.7B', 'OPT-6.7B',
                'LLama2-7B', 'LLama3-8B']
groups = bench_labels + ['Average']


# =============================================================================
# Data loader (same parsing logic as EnergyAll_papers.py)
# =============================================================================
def load_papers_data():
    """Load energy breakdown from papers_res.csv.

    Returns:
        all_energy : ndarray of shape [n_groups, n_accel, 4]
                     last group = geomean (provided by CSV)
                     components order = [Static, Dram, Buffer, Core]
    """
    with open('./results/papers_res.csv', 'r') as f:
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

    # Section 0: Normalized Cycles          (skip)
    # Section 1: Normalized Energy Breakdown (4 component lines)
    # Section 2: TOPS/Area                  (skip)
    n_accel = len(all_categories)

    energy_start = sections[1][0]
    data_start   = energy_start + 2       # skip model name + bench name lines

    components = {}
    for comp_idx in range(4):
        line  = lines[data_start + comp_idx].strip()
        parts = line.split(', ')
        comp_name = parts[0]
        values = [float(v.strip().rstrip(','))
                  for v in parts[1:] if v.strip().rstrip(',')]
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
                components['Dram'  ][val_idx],
                components['Buffer'][val_idx],
                components['Core'  ][val_idx],
            ])
        all_energy.append(bench_data)

    # Append geomean row (last n_accel entries in CSV)
    geomean_data = []
    geo_offset = n_benchmarks * n_accel
    for accel_idx in range(n_accel):
        val_idx = geo_offset + accel_idx
        geomean_data.append([
            components['Static'][val_idx],
            components['Dram'  ][val_idx],
            components['Buffer'][val_idx],
            components['Core'  ][val_idx],
        ])
    all_energy.append(geomean_data)

    return np.array(all_energy, dtype=float)


def normalize_to_baseline(energy):
    """Normalize each benchmark's components by baseline (DayPQ) total energy.

    After normalization, baseline bar total = 1.00 per benchmark.
    """
    normed = np.zeros_like(energy)
    n_groups, n_accel, n_comp = energy.shape
    for g in range(n_groups):
        baseline_total = energy[g, baseline_idx, :].sum()
        if baseline_total == 0:
            normed[g] = energy[g]
            continue
        normed[g] = energy[g] / baseline_total
    return normed


# =============================================================================
# Load + normalize
# =============================================================================
energy_raw    = load_papers_data()
energy_normed = normalize_to_baseline(energy_raw)

# =============================================================================
# Plot configuration
# =============================================================================
# Component textures (EnergyAll.py blue scheme)
sub_textures = {
    'Static' : {'color': '#CCE5FE', 'hatch': '/'},
    'Dram'   : {'color': '#A2C6EB', 'hatch': '\\\\'},
    'Buffer' : {'color': '#78A7D8', 'hatch': 'xx'},
    'Core'   : {'color': '#236AB3', 'hatch': '||'},
}
component_order = ['Static', 'Dram', 'Buffer', 'Core']

style_config = {
    'figsize'            : (30, 10),
    'bar_width'          : 0.42,
    'bar_spacing'        : 0.15,
    'group_spacing'      : 0.9,
    'margin'             : 0.3,
    'label_offset'       : 0.012,
    'boundary_linewidth' : 1.5,
}

n_cats = len(all_categories)


def calc_x_positions(bar_width, bar_spacing, group_spacing):
    x_positions = []
    offset = 0
    for _ in groups:
        x = np.arange(n_cats) * (bar_width + bar_spacing) + offset
        x_positions.append(x)
        offset += n_cats * (bar_width + bar_spacing) + group_spacing
    return x_positions


x_positions = calc_x_positions(
    style_config['bar_width'],
    style_config['bar_spacing'],
    style_config['group_spacing'],
)
x_flat = np.concatenate(x_positions)


# =============================================================================
# Figure & axes
# =============================================================================
fig = plt.figure(figsize=style_config['figsize'])
gs = fig.add_gridspec(2, 1, height_ratios=[0.08, 0.92])
legend_ax = fig.add_subplot(gs[0])
main_ax   = fig.add_subplot(gs[1])
legend_ax.axis('off')


# =============================================================================
# Stacked bars
# =============================================================================
max_total = 0.0
for g_idx, (x_group, group_vals) in enumerate(zip(x_positions, energy_normed)):
    for cat_idx, x_bar in enumerate(x_group):
        bottom = 0.0
        components = group_vals[cat_idx]   # [Static, Dram, Buffer, Core]
        for comp_idx, comp_name in enumerate(component_order):
            val = components[comp_idx]
            tex = sub_textures[comp_name]
            main_ax.bar(
                x_bar, val,
                width=style_config['bar_width'],
                bottom=bottom,
                color=tex['color'],
                hatch=tex['hatch'],
                edgecolor='black',
                linewidth=0.8,
            )
            bottom += val
        max_total = max(max_total, bottom)

        # Top label (total stacked value)
        main_ax.text(
            x_bar,
            bottom + style_config['label_offset'],
            f'{bottom:.2f}',
            ha='center', va='bottom',
            fontweight='bold',
            fontsize=font_config['bar_label'],
            rotation=90,
        )

# Y-axis
y_upper = max_total * 1.25
main_ax.set_ylim(0, y_upper)
main_ax.set_ylabel('Normalized Energy', fontsize=font_config['axis_label'])
main_ax.tick_params(axis='y', labelsize=font_config['tick_label'])
main_ax.grid(axis='y', linestyle='--', alpha=0.3)

# X-axis limits
main_ax.set_xlim(
    x_flat[0]  - style_config['bar_width'] / 2 - style_config['margin'],
    x_flat[-1] + style_config['bar_width'] / 2 + style_config['margin'],
)

# X tick labels (accelerator names under each bar)
main_ax.set_xticks(x_flat)
main_ax.set_xticklabels(
    [cat for _ in groups for cat in all_categories],
    fontsize=font_config['tick_label'],
    rotation=90,
)
main_ax.tick_params(axis='x', length=0, pad=10)

# Group labels (benchmark name) below tick labels
y_min, y_max = main_ax.get_ylim()
group_label_y = y_min - (y_max - y_min) * 0.28
for x_group, group_name in zip(x_positions, groups):
    x_center = (x_group[0] + x_group[-1]) / 2
    main_ax.text(
        x_center, group_label_y, group_name,
        ha='center', va='top',
        fontsize=font_config['group_label'],
        fontweight='bold',
        fontstyle='italic',
    )

# Group separator lines
for g_idx in range(len(groups) - 1):
    boundary = x_positions[g_idx][-1] + (
        style_config['bar_width'] + style_config['group_spacing']
        + style_config['bar_spacing']
    ) / 2
    main_ax.axvline(
        x=boundary, color='black',
        linewidth=style_config['boundary_linewidth'],
        ymin=0, ymax=1, clip_on=False,
    )

# Frame
for spine in main_ax.spines.values():
    spine.set_visible(True)
    spine.set_edgecolor('black')
    spine.set_linewidth(style_config['boundary_linewidth'])

# =============================================================================
# Legend (reverse order: Core on top to match stack visual order)
# =============================================================================
legend_elements = []
for comp_name in component_order:
    tex = sub_textures[comp_name]
    legend_elements.append(Patch(
        facecolor=tex['color'],
        hatch=tex['hatch'],
        edgecolor='black',
        linewidth=0.8,
        label=comp_name,
    ))
legend_elements.reverse()

legend_ax.legend(
    handles=legend_elements,
    loc='center',
    ncol=4,
    frameon=True,
    fontsize=font_config['legend'],
    labelspacing=0.5,
    handlelength=1.5,
    handletextpad=0.5,
    edgecolor='black',
)

# =============================================================================
# Save
# =============================================================================
plt.tight_layout()
plt.subplots_adjust(hspace=0.05)

output_path = './results/fig_papers_energy.pdf'
plt.savefig(output_path, format='pdf', bbox_inches='tight', dpi=300)
print(f"Chart saved to: {output_path}")
