# Machine Learning & AI Notes

> Converted from `notes.txt` (source file). Generated: 2026-01-11

## Table of contents

- [Types of ML Systems](#types-of-ml-systems)
  - [Supervised learning](#supervised-learning)
  - [Unsupervised learning](#unsupervised-learning)
  - [Reinforcement learning](#reinforcement-learning)
  - [Generative AI](#generative-ai)


## Types of ML Systems

### Supervised learning

- Supervised learning models learn to make predictions from labeled data (examples paired with the correct answers).
- A human provides training data that includes the desired output, and the model discovers relationships that map inputs to those outputs.
- Common supervised tasks:
  - Regression — predict a numeric value (e.g., house price).
  - Binary classification — predict one of two classes (e.g., spam vs. not spam).
  - Multiclass classification — predict one of more than two classes (e.g., digit recognition).

### Unsupervised learning

- Unsupervised learning models find patterns in data without labeled outcomes.
- A common technique is clustering, where the algorithm groups similar data points together.
- Key difference from classification: cluster categories are discovered by the algorithm rather than pre-defined.

### Reinforcement learning

- Reinforcement learning (RL) learns by interacting with an environment and receiving rewards or penalties.
- The agent learns a policy that maps states to actions to maximize cumulative reward.

### Generative AI

- Generative AI refers to models that create new content from input prompts.
- Inputs and outputs can vary (text → text, text → image, image → text, audio → audio, etc.).
- Often described as "input-type → output-type" (for instance, text-to-image).


---

## Suggested next edits

Below are concrete, small examples and references you can paste into an editor to try the ideas quickly.

### Examples & simple code snippets

- Supervised regression (scikit-learn): predict a scalar target (e.g., simplified pressure drop)

```python
from sklearn.linear_model import LinearRegression
import numpy as np

# X: rows of samples, columns of features; y: target vector
X = np.random.rand(100, 3)
y = X @ np.array([2.0, -1.0, 0.5]) + 0.1 * np.random.randn(100)
model = LinearRegression().fit(X, y)
print('mse=', ((model.predict(X) - y) ** 2).mean())
```

- Classification (scikit-learn): tiny binary classifier (flow regime by Re)

```python
from sklearn.linear_model import LogisticRegression
import numpy as np

Re = np.random.uniform(100, 1e5, size=200)
y = (Re > 2300).astype(int)  # laminar vs turbulent
X = np.vstack([Re]).T
clf = LogisticRegression(max_iter=200).fit(X, y)
print('acc=', clf.score(X, y))
```

- Unsupervised clustering (KMeans): group flow patches / features

```python
from sklearn.cluster import KMeans
import numpy as np

X = np.random.randn(200, 3)  # replace with [mean_speed, vorticity, ke]
labels = KMeans(n_clusters=3, random_state=0).fit_predict(X)
print('counts=', np.bincount(labels))
```

- Reinforcement learning (very small Q-learning loop outline):

```python
import numpy as np
# Q[s,a] table, basic epsilon-greedy updates
Q = np.zeros((16, 4))
for episode in range(100):
  s = 0
  done = False
  while not done:
    a = np.argmax(Q[s]) if np.random.rand() > 0.1 else np.random.randint(4)
    s2, r, done = step(s, a)  # implement step for your env
    Q[s, a] += 0.1 * (r + 0.99 * Q[s2].max() - Q[s, a])
    s = s2
```

These tiny snippets are meant to be minimal starting points — replace the synthetic data with CFD/thermo-hydraulics features (Reynolds number, mean speed, temperature difference, non-dimensional parameters) when you integrate with your domain datasets.

### Generative AI — expanded overview

Generative AI refers to models that produce new data (text, images, audio, fields) conditioned on some input. Common model families:

- Autoregressive transformers (e.g., GPT-family): predict the next token conditioned on previous tokens. Great for text, code, and sequence modeling. Use cases: report generation from simulation logs, code assistants, conditional sequence prediction.
- Diffusion models / score-based models: iterative denoising processes used for high-quality image and signal generation (also adapted to continuous field generation). Use cases: generating realistic synthetic images, data augmentation, super-resolution of flow fields.
- GANs (Generative Adversarial Networks): generator + discriminator trained adversarially. Use cases: data augmentation, generating plausible flow-field snapshots, style transfer between simulation grids.
- VAEs (Variational Autoencoders): probabilistic encoder/decoder that learn a latent space; useful for compression, interpolation, and uncertainty-aware generation.
- Physics-Informed Generative Models / Conditional models: combine physical constraints (PDE residuals, invariants) with learning to produce physically consistent samples.

When applying generative models to CFD / thermo-hydraulics, consider:
- Conditioning on physical parameters (Re, geometry descriptors, boundary conditions).
- Enforcing conservation laws (mass/energy) either by architecture (conservative layers) or loss penalties.
- Using learned surrogates for fast parameter sweeps and uncertainty quantification.

Example (pseudo-code, text generation with transformers):

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained('gpt2')
model = AutoModelForCausalLM.from_pretrained('gpt2')
prompt = 'Given Re=5000 and inlet velocity=2.0 m/s, summary:'
input_ids = tok(prompt, return_tensors='pt').input_ids
out = model.generate(input_ids, max_length=150)
print(tok.decode(out[0]))
```

Note: for field generation (grids/volumes), adapt model inputs/outputs (patch-based, convolutional or Fourier features), or use diffusion/GAN variants built for continuous data.

### Recommended readings & course references

- Andrew Ng — "Machine Learning" (Coursera) — excellent introduction to core supervised/unsupervised methods: https://www.coursera.org/learn/machine-learning
- "Pattern Recognition and Machine Learning" — Christopher M. Bishop (textbook): https://www.microsoft.com/en-us/research/people/cmbishop/
- "Deep Learning" — Ian Goodfellow, Yoshua Bengio, Aaron Courville (book): https://www.deeplearningbook.org/
- Sutton & Barto — "Reinforcement Learning: An Introduction" (RL fundamentals): http://incompleteideas.net/book/the-book-2nd.html
- Goodfellow et al. — GANs paper: https://arxiv.org/abs/1406.2661
- Kingma & Welling — VAE: https://arxiv.org/abs/1312.6114
- Vaswani et al. — "Attention Is All You Need" (transformers): https://arxiv.org/abs/1706.03762
- Ho et al. — Diffusion models: https://arxiv.org/abs/2006.11239
- PINNs and physics-informed models review: Raissi et al., "Physics-informed neural networks": https://doi.org/10.1016/j.jcp.2019.109.369

---



## Tags

`supervised`, `unsupervised`, `reinforcement-learning`, `generative-ai`, `ml-basics`
