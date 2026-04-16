# ADD-LLM Simulator

A cycle-level energy/performance simulator for systolic-array-based LLM accelerators.
This project is built upon the following open-source simulators:

- [BitFusion](https://github.com/hsharma35/bitfusion) (ISCA '18)
- [AxCore](https://github.com/CLab-HKUST-GZ/micro58-axcore) (MICRO'25) (based on BitFusion)
- [DNNWeaver v2](https://github.com/hsharma35/dnnweaver2)

We extend these frameworks with ADD-LLM PE architectures and multi-paper accelerator comparison capabilities.

## Project Structure

```
simulator/
├── params/
│   ├── conf/           # Accelerator configuration files (.ini)
│   └── synth/          # Synthesis results for PE arrays (.csv)
├── scripts/            # Shell scripts for running experiments
├── results/            # Simulation outputs (.csv, .pdf)
├── AxCore/             # Extended simulator (AxCoreSimulator)
├── bitfusion/          # Original BitFusion simulator
└── dnnweaver2/         # DNN graph & layer definitions
```

## Prerequisites

- Ubuntu 22.04 LTS
- Conda (>= 25.1)
- Python 3.9
- GCC 11.4

## Getting Started

```bash
# 1. Create and activate conda environment
conda create -n addllm_sim python=3.9
conda activate addllm_sim
pip install -r requirements.txt

# 2. Build CACTI (memory model)
git clone https://github.com/HewlettPackard/cacti ./bitfusion/sram/cacti/
make -C ./bitfusion/sram/cacti/
cp -r ./bitfusion/sram/cacti ./AxCore/sram/
```

## Evaluation

### Available Scripts

| Script | Description | Output |
|--------|-------------|--------|
| `scripts/fig_papers.sh` | Multi-paper accelerator comparison (v1) | `results/papers_res.csv`, `results/fig_papers.pdf` |
| `scripts/fig_papers_v2.sh` | Multi-paper accelerator comparison (v2) | `results/papers_v2_res.csv`, `results/fig_papers_v2.pdf` |
| `scripts/fig_add_llms.sh` | ADD-LLM v1 vs v2 internal comparison | `results/addllms_res.csv`, `results/fig_add_llms.pdf` |

### Running

```bash
conda activate addllm_sim

# Multi-paper comparison (v1)
bash scripts/fig_papers.sh

# Multi-paper comparison (v2, updated DC results)
bash scripts/fig_papers_v2.sh

# ADD-LLM variant comparison
bash scripts/fig_add_llms.sh
```

### Compared Accelerators

**fig_papers / fig_papers_v2** -- cross-architecture comparison:

| Accelerator | Array Size | Technology |
|-------------|-----------|------------|
| ADD-LLM-S (Ours) | 32x32 | 28nm |
| ADD-LLM-L (Ours) | 64x64 | 28nm |
| DayPQ (TVLSI'26) | 32x16 | 22nm |
| BitMoD (ToC'25) | 32x32 | 28nm |
| M2XFP (ASPLOS'26) | 32x32 | 28nm |
| AMove | 32x32 | 45nm |
| Tender (ISCA'24) | 64x64 | 28nm |

**fig_add_llms** -- internal variant comparison:

| Variant | Array Size | Description |
|---------|-----------|-------------|
| ADD-LLM-S v1 | 32x32 | Original DC results |
| ADD-LLM-L v1 | 64x64 | Original DC results |
| ADD-LLM-S v2 | 32x32 | Updated (add_moe_core_32_v2) |
| ADD-LLM-L v2 | 64x64 | Updated (add_moe_core_v2) |

### Dataflow

The simulator supports two systolic array dataflow modes, controlled by the `weight_stationary` flag:

- **Weight Stationary** (`weight_stationary=True`): Weights are held in the PE array; activations and partial sums are streamed. Currently used by all run scripts.
- **Output Stationary** (`weight_stationary=False`): Partial sums accumulate in place; weights and activations are streamed.

### Configuration

Each accelerator is defined by an `.ini` file in `params/conf/`. Key parameters:

```ini
[accelerator]
N = 32              # Array rows
M = 32              # Array columns
pmax = 8            # Max precision (bits)
pmin = 4            # Min precision (bits)
frequency = 1000    # Clock frequency (MHz)
module_name = ...   # PE module name (for synthesis lookup)
```

Synthesis power/area data is provided via CSV files in `params/synth/`.

## Acknowledgements

This simulator is based on the BitFusion, AxCore, and DNNWeaver v2 projects. We thank the original authors for making their code publicly available.
