"""Simple Q-Learning on a small gridworld (no external dependencies)."""
import random
import numpy as np
from typing import Tuple, List


class GridWorld:
    def __init__(self, size: int = 4):
        self.size = size
        self.start = (0, 0)
        self.goal = (size - 1, size - 1)

    def state_to_idx(self, s: Tuple[int, int]) -> int:
        return s[0] * self.size + s[1]

    def idx_to_state(self, idx: int) -> Tuple[int, int]:
        return (idx // self.size, idx % self.size)

    def step(self, state: Tuple[int, int], action: int) -> Tuple[Tuple[int, int], float, bool]:
        # actions: 0=up,1=right,2=down,3=left
        i, j = state
        if action == 0:
            i = max(0, i - 1)
        elif action == 1:
            j = min(self.size - 1, j + 1)
        elif action == 2:
            i = min(self.size - 1, i + 1)
        elif action == 3:
            j = max(0, j - 1)
        new_state = (i, j)
        if new_state == self.goal:
            return new_state, 10.0, True
        else:
            return new_state, -1.0, False


def q_learning(num_episodes: int = 500, alpha: float = 0.5, gamma: float = 0.95, epsilon: float = 0.1, size: int = 4):
    env = GridWorld(size=size)
    n_states = size * size
    n_actions = 4
    Q = np.zeros((n_states, n_actions))

    for ep in range(num_episodes):
        state = env.start
        done = False
        while not done:
            s_idx = env.state_to_idx(state)
            if random.random() < epsilon:
                a = random.randrange(n_actions)
            else:
                a = int(np.argmax(Q[s_idx]))
            new_state, reward, done = env.step(state, a)
            ns_idx = env.state_to_idx(new_state)
            Q[s_idx, a] += alpha * (reward + gamma * np.max(Q[ns_idx]) - Q[s_idx, a])
            state = new_state

    # derive policy
    policy = np.argmax(Q, axis=1)  # action per state
    return Q, policy, env


def policy_to_arrows(policy: List[int], env: GridWorld) -> List[str]:
    mapping = {0: "↑", 1: "→", 2: "↓", 3: "←"}
    arrows = []
    for idx, a in enumerate(policy):
        s = env.idx_to_state(idx)
        if s == env.goal:
            arrows.append("G")
        else:
            arrows.append(mapping[int(a)])
    return arrows


if __name__ == "__main__":
    Q, policy, env = q_learning()
    arrows = policy_to_arrows(policy, env)
    # print grid
    for i in range(env.size):
        row = arrows[i * env.size : (i + 1) * env.size]
        print(" ".join(row))
