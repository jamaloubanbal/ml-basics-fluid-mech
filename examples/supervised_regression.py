"""CFD-themed regression: predict pressure drop in a pipe from simple synthetic features.

This example generates synthetic samples of pipe flow and computes a target
pressure-drop using a simplified Darcy–Weisbach inspired formula with noise.
It then fits a LinearRegression model and reports MSE.
"""
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


def compute_pressure_drop(flow_rate, density, viscosity, length, diameter):
    """Simple proxy for pressure drop. Not physically exact; meant for examples.

    Uses a quadratic dependence on flow rate and inverse dependence on diameter.
    """
    # basic scaling: Δp ∝ rho * (Q/A)^2 * (L/D)
    area = np.pi * (diameter / 2) ** 2
    velocity = flow_rate / area
    dp = 0.5 * density * velocity ** 2 * (length / diameter)
    # add a small viscous correction
    dp += viscosity * flow_rate / (diameter ** 2)
    return dp


def run_regression(random_state: int = 0, n_samples: int = 300):
    rng = np.random.RandomState(random_state)
    # sample reasonable ranges for small-scale piping
    flow_rate = rng.uniform(0.01, 0.5, size=n_samples)  # m^3/s
    density = rng.uniform(950.0, 1050.0, size=n_samples)  # kg/m^3
    viscosity = rng.uniform(1e-4, 5e-3, size=n_samples)  # Pa.s
    length = rng.uniform(0.5, 10.0, size=n_samples)  # m
    diameter = rng.uniform(0.01, 0.2, size=n_samples)  # m

    X = np.vstack([flow_rate, density, viscosity, length, diameter]).T
    y = np.array([compute_pressure_drop(q, rho, mu, L, D)
                  for q, rho, mu, L, D in X])
    # add noise
    y += rng.normal(scale=0.05 * np.mean(y), size=y.shape)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    return {
        "mse": float(mse),
        "coef": model.coef_.tolist(),
        "intercept": float(model.intercept_),
    }


if __name__ == "__main__":
    out = run_regression()
    print("Pipe flow regression — MSE:", out["mse"])    
    print("Coefficients:", out["coef"])    
