"""CFD-themed clustering: cluster synthetic flow-field features.

We synthesize small 2D velocity patches with different vortex strengths and
cluster them based on summary features (mean speed, vorticity, kinetic energy).
"""
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def make_vortex_patch(strength, size=16):
    # create a simple rotational velocity field around patch center
    x = np.linspace(-1, 1, size)
    y = np.linspace(-1, 1, size)
    X, Y = np.meshgrid(x, y)
    # tangential velocity ~ strength * r
    u = -strength * Y
    v = strength * X
    return u, v


def extract_features(u, v):
    speed = np.sqrt(u ** 2 + v ** 2)
    mean_speed = np.mean(speed)
    ke = 0.5 * np.mean(speed ** 2)
    # approximate z-vorticity: dv/dx - du/dy with central differences
    dvdx = np.gradient(v, axis=1)
    dudy = np.gradient(u, axis=0)
    vorticity = np.mean(dvdx - dudy)
    return [mean_speed, ke, vorticity]


def run_clustering(random_state: int = 0, n_samples: int = 300):
    rng = np.random.RandomState(random_state)
    strengths = rng.choice([0.5, 1.5, 3.0, 5.0], size=n_samples)
    features = []
    for s in strengths:
        u, v = make_vortex_patch(s)
        features.append(extract_features(u, v))
    X = np.array(features)
    kmeans = KMeans(n_clusters=4, random_state=random_state)
    labels = kmeans.fit_predict(X)
    score = silhouette_score(X, labels)
    return {"silhouette": float(score), "centers": kmeans.cluster_centers_.tolist()}


if __name__ == "__main__":
    out = run_clustering()
    print("Flow patch clustering — silhouette score:", out["silhouette"])    
