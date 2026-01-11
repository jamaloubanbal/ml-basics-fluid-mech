Examples — ML basics (lightweight)

This folder contains small, runnable examples illustrating core ML topics, with CFD / thermo-hydraulics-themed synthetic problems.

Prerequisites

Install the minimal dependencies (preferably in a venv):

```bash
pip install -r examples/requirements.txt
```


Scripts (CFD / thermo-hydraulics themed)

- `supervised_regression.py` — synthetic pipe-flow pressure-drop regression (LinearRegression).
- `supervised_classification.py` — flow regime classification (laminar vs turbulent) from Reynolds-derived features.
- `unsupervised_clustering.py` — cluster synthetic flow-field patches (vortex strengths) with KMeans.
- `rl_q_learning.py` — tiny Q-learning on a 4x4 grid world (kept as a control example for RL).
- `generative_char_rnn.py` — tiny char-level RNN (kept as a simple generative example).

Run an example:

```bash
python examples/supervised_regression.py
```

Run tests:

```bash
pytest -q
```
