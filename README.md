# llm-config-benchmark  
### Empirical Runtime Optimization Harness for Local LLM Inference

A parametric benchmarking system designed to determine the optimal runtime configuration for local Large Language Models.

Instead of relying on anecdotal tuning, this tool performs a controlled grid search across critical inference parameters:

- GPU Layer Offloading (`ngl`)
- CPU Thread Allocation (`threads`)
- Context Size (`ctx`)

Each configuration runs in a fully isolated subprocess while hardware telemetry is captured independently to ensure measurement integrity.

This is empirical optimization — not guesswork.

---

## Why This Exists

Optimizing local LLM inference is non-trivial.

Common tuning approaches rely on:

- Trial-and-error adjustments
- Subjective latency impressions
- Self-reported backend metrics
- Static "recommended settings"

These methods fail to capture:

- True peak VRAM usage
- Real CPU saturation
- Power draw behavior
- Stability under varying context sizes

This project provides controlled experimentation with hardware-level monitoring and reproducible CSV exports.

---

## Design Goals

- **Isolated Execution** — Each configuration runs in a clean subprocess
- **Sidecar Telemetry** — Hardware polling independent of model output
- **Automated Grid Search** — Systematic parameter sweeping
- **Deterministic Reporting** — Structured CSV export for comparison
- **Deadlock-Safe Execution** — Subprocess control with pipe protection
- **Windows CUDA Optimization** — Targeted for NVIDIA-accelerated systems

---

## System Architecture

The system is intentionally decoupled to prevent measurement contamination.

### 1. Orchestrator (`main.py`)
- Defines Smart Search parameter bounds
- Manages execution order
- Aggregates results
- Exports benchmark_results.csv
- Determines winning configuration

### 2. Subprocess Controller (`src/runner.py`)
- Launches `llama-cli.exe` safely
- Injects CLI parameters
- Prevents pipe deadlocks
- Ensures clean termination

### 3. Sidecar Monitor (`src/monitor.py`)
- Polls NVIDIA hardware via `pynvml`
- Tracks CPU/RAM via `psutil`
- Runs asynchronously
- Captures peak VRAM and average power draw

### 4. Log Parser (`src/parser.py`)
- Parses raw stdout/stderr
- Extracts tokens-per-second metrics via Regex
- Converts raw CLI output into structured data

---

## Architectural Characteristics

- Fully decoupled measurement pipeline
- Hardware polling independent of CLI reporting
- Subprocess isolation for reproducibility
- CSV-based experiment tracking
- Deterministic winner selection

---

## Tech Stack

- **Language:** Python 3.10+
- **GPU Telemetry:** pynvml
- **System Metrics:** psutil
- **Data Handling:** pandas
- **Progress Tracking:** tqdm
- **Target Backend:** llama.cpp (llama-cli.exe)

---

## Prerequisites

- **Operating System:** Windows
- **GPU:** NVIDIA CUDA architecture
- **Python:** 3.10+
- **Binary:** Compiled `llama-cli.exe`
- **Model:** `.gguf` file

---

## Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/llm-config-benchmark.git
cd llm-config-benchmark
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Dependencies include:

- pynvml
- psutil
- pandas
- tqdm

---

## Configuration

Open `main.py` and update your local paths:

```python
CLI_PATH = r"C:\path\to\llama-cli.exe"
MODEL_PATH = r"C:\path\to\model.gguf"
```

Adjust the Smart Search bounds if desired:

```python
N_GPU_LAYERS_TO_TEST = [...]
THREADS_TO_TEST = [...]
CONTEXT_SIZES_TO_TEST = [...]
```

---

## Running the Benchmark

Execute from the root directory:

```bash
python main.py
```

The system will:

1. Iterate across all parameter combinations
2. Launch isolated subprocesses
3. Monitor hardware in parallel
4. Parse performance metrics
5. Export results to `benchmark_results.csv`
6. Print the best-performing configuration

---

## Output Metrics

Each tested configuration records:

- Tokens per second
- Peak VRAM usage
- Average CPU utilization
- Peak RAM usage
- Average GPU power draw

Example CSV structure:

```
ngl,threads,ctx,tokens_per_sec,peak_vram_gb,avg_power_watts
32,8,4096,72.4,7.8,162
40,8,4096,75.9,8.4,174
...
```

The winning configuration is selected based on throughput under hardware constraints.

---

## Project Structure

```
llm-config-benchmark/
├── src/
│   ├── runner.py
│   ├── monitor.py
│   └── parser.py
├── main.py
├── requirements.txt
└── README.md
```

---

## Troubleshooting

### Process Hanging / Deadlocks

If execution stalls during initialization:

Ensure `runner.py` includes the `--no-cnv` flag to prevent interactive chat mode in newer `llama.cpp` builds.

---

### Regex Mismatch (0.0 t/s)

If tokens/sec shows as `0.0`, your backend CLI output format may have changed.

Update the Regex capture groups in `src/parser.py`.

---

## Limitations

- Windows-only design
- Single-machine benchmarking (no distributed execution)
- Dependent on CLI output format stability
- Grid search only (no Bayesian optimization or adaptive pruning)

---

## What This Project Demonstrates

- Controlled experimental system design
- Subprocess isolation patterns
- Hardware-level telemetry integration
- Parametric search automation
- Performance benchmarking methodology
- Measurement integrity engineering

---

## Project Status

- Feature complete
- Stable for Windows CUDA benchmarking
- No active feature development planned

This project focuses on empirical optimization, hardware-aware experimentation, and reproducible inference benchmarking.
