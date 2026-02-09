# Bayesian Difference-in-Difference

A [Bayesian difference-in-differences](https://arxiv.org/pdf/2508.02970v1) framework for sensitivity analysis under violations of the parallel trends assumption, with applications to policy evaluation.


## Installation

Create a new environment and install the required packages:

```
conda create -n bdid python=3.10 -y
conda activate bdid
pip install -r requirements.txt
```

## Source Code Overview

The core implementation is organized under `src/`:

- **`model.py`** contains the Bayesian difference-in-differences models, including variants that allow for violations of parallel trends via AR(1) deviation processes and empirical Bayes calibration.

- **`plot.py`** provides functions for visualizing counterfactual outcomes and posterior sensitivity analyses based on fitted model outputs.

- **`tools.py`** defines lightweight data abstractions and reusable plotting components shared across analyses.

- **`utils.py`** includes data preprocessing routines and auxiliary helper functions used by the modeling code.


## Repository Structure

```txt
Bayesian-Difference-in-Difference/
├─ README.md
├─ requirements.txt
└─ src/
   ├─ model.py
   ├─ plot.py
   ├─ tools.py
   └─ utils.py
