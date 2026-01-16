# Notebooks — CFD / Thermo-hydraulics examples

This folder contains three small, runnable Jupyter notebooks that generate synthetic CFD/thermo-hydraulics data, train simple models, and visualize results:

- `regression.ipynb` — synthetic pipe-flow dataset; linear regression to predict a simplified pressure drop; predicted vs true scatter plot.
- `classification.ipynb` — Reynolds-number based synthetic cases; logistic regression to classify laminar vs turbulent flow; confusion matrix and distribution plots.
- `clustering.ipynb` — synthesize vortex patches, extract features (mean speed, kinetic energy, vorticity), cluster with KMeans and show representative patches.

Quick prerequisites

1. Create and activate a virtual environment (recommended):

```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install the project example dependencies and a few extras used by notebooks:

```bash
pip install -r examples/requirements.txt
pip install matplotlib jupyter
```

(If you prefer to use the project's existing venv, substitute its Python executable. Example: `/home/jamal/PROJECTS/ml-basics-fluid-mech/.venv/bin/python -m pip install -r examples/requirements.txt`)

Run the notebooks interactively

- Start Jupyter Notebook (or JupyterLab) from the project root and open the files under `examples/notebooks`:

```bash
jupyter notebook
# or
jupyter lab
```

- In VS Code you can also open the `.ipynb` files and use the "Run All" buttons; ensure the active interpreter is the project's venv.

Execute notebooks from the command line (batch / CI)

- You can execute and save outputs using `nbconvert` (requires `jupyter`):

```bash
jupyter nbconvert --to notebook --execute examples/notebooks/regression.ipynb --output executed_regression.ipynb
```

Notes and tips

- The notebooks use small, synthetic datasets intended for learning and demonstration. Replace the synthetic generators with your own simulation outputs (CSV, HDF5, or native arrays) when experimenting with real CFD/thermo datasets.
- For plotting and larger visualizations consider adding `seaborn` and/or `plotly` to the environment:

```bash
pip install seaborn plotly
```

- If you plan to run notebooks non-interactively in CI, prefer smaller sample sizes or add a `--ExecutePreprocessor.timeout=60` flag to `nbconvert` to avoid long-running cells.

Contact / next steps

If you want, I can:

- execute the notebooks here and save the executed versions with outputs,
- add a notebook that reads a sample CFD dataset (CSV/HDF5) and demonstrates preprocessing and visualization,
- or convert one notebook into a standalone Python script that produces figures for documentation.

Tell me which you prefer and I'll implement it.