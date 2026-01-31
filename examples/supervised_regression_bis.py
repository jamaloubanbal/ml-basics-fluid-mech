"""Improved regression example: predict a synthetic target from scaled features.

This example uses sklearn's make_regression to generate synthetic data with a known
linear relationship, ensuring good performance with LinearRegression.
"""
import numpy as np
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def run_regression(random_state: int = 0, n_samples: int = 300, n_features: int = 5):
    # Generate synthetic regression data with linear relationship
    X, y = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        noise=0.1,  # small noise for good fit
        random_state=random_state
    )

    # Scale features to improve conditioning
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=random_state
    )

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
    print("Improved regression — MSE:", out["mse"])
    print("Coefficients:", out["coef"])