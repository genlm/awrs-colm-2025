### Scripts overview

This directory contains helper scripts to run experiments and to aggregate the resulting metrics.

- `molecular_synthesis.sh`: Runs the molecular-synthesis task.
- `pattern_matching.sh`: Runs the pattern-matching task.
- `text_to_sql.sh`: Runs the text-to-SQL task.
- `json.sh`: Runs the JSON task.
- `summarize_results.py`: Aggregates results into a CSV and generates LaTeX tables with bootstrap confidence intervals.


### Usage

Make the shell scripts executable (first time only):

```bash
chmod +x *.sh
```

Run tasks (default output dirs shown; you may pass a custom results dir as the first argument where supported):

- Molecular synthesis:

```bash
./molecular_synthesis.sh [results/molecular_synthesis]
```

- Pattern matching:

```bash
./pattern_matching.sh [results/pattern_matching]
```

- Text-to-SQL (2 replicates, chat format):

```bash
./text_to_sql.sh [results/text_to_sql]
```

- JSON (2 replicates, chat format). Note: this script requires a results dir argument:

```bash
./json.sh [results/json]
```

Each task script runs the following methods (with the shown default hyperparameters in the scripts):

- `base-lm`
- `lcd`
- `sample-rerank` (e.g., `--num-particles 10`)
- `twisted-smc` (e.g., `--num-particles 10 --ess-threshold 0.90`)
- `awrs-smc` (e.g., `--num-particles 5 --ess-threshold 0.5`)

Edit the scripts directly to change `MODEL_NAME`, `N_REPLICATES`, particle counts, and ESS thresholds.

### Summarize results

Aggregate metrics across all tasks/methods and produce a CSV and LaTeX tables:

```bash
python awrs-colm-2025/scripts/summarize_results.py \
  --results-dir results \
  --output results/aggregated_results.csv \
  --latex-dir results/latex_tables \
  --bootstrap-samples 10000
```

- Looks for files ending with `-results.json` under `--results-dir`.
- Computes mean accuracy and runtime with 95% bootstrap confidence intervals.
- Writes a single CSV and per-task LaTeX tables.
