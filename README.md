# Adaptive Weighted Rejection Sampling

[![arXiv](https://img.shields.io/badge/arXiv-2504.05410-b31b1b.svg)](https://arxiv.org/abs/2504.05410)

This repository contains content related to the paper *Fast Controlled Generation from Language Models with Adaptive Weighted Rejection Sampling* (COLM 2025).

The most performant reference implementation of the AWRS-SMC algorithm can be found in the actively maintained [genlm/genlm-control](https://github.com/genlm/genlm-control) library.

This repo contains scripts to replicate the paper experiments based on the [genlm/genlm-control](https://github.com/genlm/genlm-control) and [genlm/genlm-eval](https://github.com/genlm/genlm-eval) libraries.

## Setup

### Install project

1. Clone this repository:

    ```bash
    git clone https://github.com/genlm/awrs-colm-2025.git
    cd awrs-colm-2025
    ```

2. Create and activate a uv environment:

    ```bash
    uv venv --python=3.11
    source .venv/bin/activate
    ```

    *To install uv, run `curl -LsSf https://github.com/astral-sh/uv/releases/latest/download/uv-installer.sh`.*


3. Install the project dependencies:

    ```bash
    uv pip install -e .
    ```

### Download required datasets

#### Molecular synthesis

Download the GBD-17 dataset (`GDB17.50000000.smi.gz`) from [here](https://gdb.unibe.ch/downloads/) and unzip it in the `data/molecular_synthesis` directory:

```bash
mkdir -p data/molecular_synthesis
cd data/molecular_synthesis
# Download GDB17.50000000.smi.gz to this directory, then:
gunzip GDB17.50000000.smi.gz
cd ../..
```

#### Text to SQL

Download and unzip the Spider dataset in the `data/spider` directory with the following commands:
```bash

cd data/spider
gdown 'https://drive.google.com/u/0/uc?id=1403EGqzIDoHMdQF4c9Bkyl7dZLZ5Wt6J&export=download'
unzip spider_data.zip
cd ../..
```

## Running experiments

See the `scripts` directory for scripts which run the experiments from the paper.

## Repository structure

```
awrs-colm-2025/
├── experiments/              # Core experimental code
│   ├── __main__.py           # Main CLI entry point for running experiments
│   ├── tasks.py              # Task definitions and registry for different domains
│   ├── methods.py            # Implementation of sampling methods (AWRS SMC, baselines)
│   └── sampler.py            # Implementation of the core AWRS algorithm for constrained-generation
├── scripts/                  # Experiment execution scripts
│   ├── text_to_sql.sh        # Script to run Text-to-SQL experiments
│   └── pattern_matching.sh   # Script to run pattern matching experiments
├── data/                     # Dataset storage
│   ├── spider/               # Spider Text-to-SQL dataset (needs to be downloaded)
│   ├── molecular_synthesis/  # Location of GDB-17 molecular synthesis dataset (needs to be downloaded)
│   └── pattern_matching/     # Pattern matching dataset
└── tests/                    # Test suite
```

## Citation

```bibtex
@inproceedings{awrs2025colm,
    title = {Fast Controlled Generation from Language Models with Adaptive Weighted Rejection Sampling},
    author = {Lipkin, Benjamin and LeBrun, Benjamin and Vigly, Jacob Hoover and Loula, Jo{\~a}o and MacIver, David R and Du, Li and Eisner, Jason and Cotterell, Ryan and Mansinghka, Vikash and O'Donnell, Timothy J and others},
    booktitle = {Second Conference on Language Modeling},
    year = {2025}
}
```
