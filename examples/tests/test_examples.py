"""Basic tests that import and run the example functions quickly."""
from examples.supervised_regression import run_regression
from examples.supervised_classification import run_classification
from examples.unsupervised_clustering import run_clustering
from examples.rl_q_learning import q_learning, policy_to_arrows
from examples.generative_char_rnn import train_and_generate


def test_regression_runs():
    out = run_regression(random_state=1)
    assert "mse" in out and out["mse"] >= 0


def test_classification_runs():
    out = run_classification(random_state=1)
    assert "accuracy" in out and 0.0 <= out["accuracy"] <= 1.0


def test_clustering_runs():
    out = run_clustering(random_state=1)
    assert "silhouette" in out


def test_rl_runs():
    Q, policy, env = q_learning(num_episodes=50)
    arrows = policy_to_arrows(policy, env)
    assert len(arrows) == env.size * env.size


def test_rnn_runs():
    # very short training for CI speed — use a thermo-hydraulics themed tiny corpus
    sample = train_and_generate(corpus="heat flow\n", epochs=6, gen_len=30)
    assert isinstance(sample, str)
