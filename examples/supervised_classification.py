"""CFD-themed classification: predict flow regime (laminar vs turbulent) using Reynolds number features.

We create synthetic pipe flow cases and classify whether flow is laminar (Re < 2300) or turbulent.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def compute_reynolds(flow_rate, density, viscosity, diameter):
    area = np.pi * (diameter / 2) ** 2
    velocity = flow_rate / area
    Re = density * velocity * diameter / viscosity
    return Re


def run_classification(random_state: int = 0, n_samples: int = 300):
    rng = np.random.RandomState(random_state)
    flow_rate = rng.uniform(1e-3, 0.5, size=n_samples)
    density = rng.uniform(900.0, 1100.0, size=n_samples)
    viscosity = rng.uniform(1e-5, 1e-2, size=n_samples)
    diameter = rng.uniform(0.005, 0.2, size=n_samples)

    Re = np.array([compute_reynolds(q, rho, mu, D) for q, rho, mu, D in zip(flow_rate, density, viscosity, diameter)])
    # label: 0=laminar (Re<2300), 1=turbulent
    y = (Re >= 2300).astype(int)
    X = np.vstack([flow_rate, density, viscosity, diameter, Re]).T

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)
    clf = LogisticRegression(max_iter=500)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    return {"accuracy": float(acc)}


if __name__ == "__main__":
    out = run_classification()
    print("Flow regime classification — accuracy:", out["accuracy"])    
