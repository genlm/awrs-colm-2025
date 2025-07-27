# Adaptive Weighted Rejection Sampling

[![arXiv](https://img.shields.io/badge/arXiv-2504.05410-b31b1b.svg)](https://arxiv.org/abs/2504.05410)

This repository contains content related to the paper *Fast Controlled Generation from Language Models with Adaptive Weighted Rejection Sampling* (COLM 2025).

The most performant reference implementation of the AWRS-SMC algorithm can be found in the actively maintained [genlm/genlm-control](https://github.com/genlm/genlm-control) library.

Tidied scripts to replicate the paper experiments based on the [genlm/genlm-control](https://github.com/genlm/genlm-control) and [genlm/genlm-eval](https://github.com/genlm/genlm-eval) libraries will be available shortly.

## Setup


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

## Citation

```bibtex
@inproceedings{awrs2025colm,
    title = {Fast Controlled Generation from Language Models with Adaptive Weighted Rejection Sampling},
    author = {Lipkin, Benjamin and LeBrun, Benjamin and Vigly, Jacob Hoover and Loula, Jo{\~a}o and MacIver, David R and Du, Li and Eisner, Jason and Cotterell, Ryan and Mansinghka, Vikash and O'Donnell, Timothy J and others},
    booktitle = {Second Conference on Language Modeling},
    year = {2025}
}
```
