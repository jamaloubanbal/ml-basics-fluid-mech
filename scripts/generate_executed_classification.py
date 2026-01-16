import io
import base64
import nbformat
from nbformat.v4 import new_notebook, new_code_cell, new_markdown_cell, output_from_msg, new_output
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score


def compute_reynolds(flow_rate, density, viscosity, diameter):
    area = np.pi * (diameter / 2) ** 2
    velocity = flow_rate / area
    Re = density * velocity * diameter / viscosity
    return Re

rng = np.random.RandomState(2)
n = 400
flow_rate = rng.uniform(1e-4, 0.6, size=n)
density = rng.uniform(900.0, 1100.0, size=n)
viscosity = rng.uniform(1e-5, 1e-2, size=n)
diameter = rng.uniform(0.005, 0.2, size=n)
Re = np.array([compute_reynolds(q, rho, mu, D) for q, rho, mu, D in zip(flow_rate, density, viscosity, diameter)])
# labels by median to guarantee two classes
y = (Re >= np.median(Re)).astype(int)
X = np.vstack([flow_rate, density, viscosity, diameter, Re]).T

# median-Re baseline
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
thresh = np.median(X[:, 4])
preds = (X_test[:, 4] >= thresh).astype(int)
acc = accuracy_score(y_test, preds)

# confusion matrix plot
cm = confusion_matrix(y_test, preds, labels=[0,1])
fig1, ax1 = plt.subplots(figsize=(4,3))
ax1.imshow(cm, interpolation='nearest', cmap='Blues')
ax1.set_title('Confusion matrix')
ax1.set_xlabel('Predicted')
ax1.set_ylabel('True')
for (i, j), v in np.ndenumerate(cm):
    ax1.text(j, i, str(v), ha='center', va='center')
buf1 = io.BytesIO()
fig1.tight_layout()
fig1.savefig(buf1, format='png')
buf1.seek(0)
img1_b64 = base64.b64encode(buf1.read()).decode('ascii')
plt.close(fig1)

# histogram plot
fig2, ax2 = plt.subplots(figsize=(6,4))
ax2.hist([Re[y==0], Re[y==1]], bins=40, stacked=True, label=['low Re','high Re'])
ax2.set_xlabel('Reynolds number')
ax2.set_ylabel('count')
ax2.legend()
ax2.set_title('Distribution of Re (synthetic)')
buf2 = io.BytesIO()
fig2.tight_layout()
fig2.savefig(buf2, format='png')
buf2.seek(0)
img2_b64 = base64.b64encode(buf2.read()).decode('ascii')
plt.close(fig2)

# Build executed notebook
nb = new_notebook()
nb['cells'] = []
nb['cells'].append(new_markdown_cell('# Classification: Flow regime (median-Re baseline)'))
nb['cells'].append(new_code_cell('''# Data generation and median-Re baseline\n# (Executed outputs included below)'''))

# Code cell with outputs: print acc and show two images
code_cell = new_code_cell("""# Executed results\nprint(f'Median-Re baseline accuracy: {acc:.3f}')""")
# add stdout stream
code_cell['outputs'] = []
code_cell['outputs'].append(new_output(output_type='stream', name='stdout', text=f"Median-Re baseline accuracy: {acc:.3f}\n"))
code_cell['outputs'].append(new_output(output_type='display_data', data={'image/png': img1_b64}, metadata={}))
code_cell['outputs'].append(new_output(output_type='display_data', data={'image/png': img2_b64}, metadata={}))
nb['cells'].append(code_cell)

# write executed notebook
outpath = 'examples/notebooks/executed_classification.ipynb'
nbformat.write(nb, outpath)
print('Wrote', outpath)
