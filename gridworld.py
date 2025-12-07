import numpy as np
import random
import matplotlib.pyplot as plt

# -------- GridWorld Environment --------
class GridWorldEnv:
    # Simple 5x5 grid with a start, goal, and a couple of traps
    def __init__(self, size=5):
        self.size = size
        self.n_states = size * size
        self.n_actions = 4

        self.start = (0, 0)
        self.goal = (size - 1, size - 1)

        # Traps give negative reward
        self.traps = [(1, 3), (3, 1)]

        self.state = self.start

    def reset(self):
        self.state = self.start
        return self._to_index(self.state)

    def _to_index(self, state):
        r, c = state
        return r * self.size + c

    def _to_state(self, index):
        return (index // self.size, index % self.size)

    def step(self, action):
        r, c = self.state

        # Move agent
        if action == 0:   r = max(r - 1, 0)                # up
        elif action == 1: c = min(c + 1, self.size - 1)    # right
        elif action == 2: r = min(r + 1, self.size - 1)    # down
        elif action == 3: c = max(c - 1, 0)                # left

        self.state = (r, c)
        reward = -1
        done = False

        # Trap and goal checks
        if self.state in self.traps:
            reward = -10
            done = True

        if self.state == self.goal:
            reward = 10
            done = True

        return self._to_index(self.state), reward, done, {}

    def render(self, policy=None):
        arrows = {0: "↑", 1: "→", 2: "↓", 3: "←"}

        for r in range(self.size):
            row = []
            for c in range(self.size):
                cell = (r, c)
                if cell == self.start: row.append("S")
                elif cell == self.goal: row.append("G")
                elif cell in self.traps: row.append("X")
                else:
                    if policy is None:
                        row.append(".")
                    else:
                        idx = self._to_index(cell)
                        row.append(arrows[policy[idx]])
            print(" ".join(row))
        print()
        

# -------- Q-Learning Training --------
def q_learning_train(env, episodes=1000, alpha=0.1, gamma=0.99,
                     eps_start=1.0, eps_end=0.1, eps_decay=0.995):

    Q = np.zeros((env.n_states, env.n_actions))
    rewards = []
    eps = eps_start

    for ep in range(episodes):
        state = env.reset()
        total = 0
        done = False

        while not done:
            # Epsilon-greedy choice
            if random.random() < eps:
                action = random.randint(0, env.n_actions - 1)
            else:
                action = np.argmax(Q[state])

            next_state, reward, done, _ = env.step(action)

            # Q-learning update
            best_next = np.argmax(Q[next_state])
            Q[state, action] += alpha * (reward + gamma * Q[next_state, best_next] - Q[state, action])

            state = next_state
            total += reward

        rewards.append(total)
        eps = max(eps_end, eps * eps_decay)

        if (ep + 1) % 100 == 0:
            print(f"Episode {ep+1}/{episodes} | Reward: {total} | Eps: {eps:.3f}")

    return Q, rewards


# -------- Helpers --------
def plot_rewards(rewards):
    plt.plot(rewards)
    plt.title("Training Rewards")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.grid(True)
    plt.show()

def extract_policy(Q):
    return np.argmax(Q, axis=1)

def test_agent(env, Q):
    state = env.reset()
    path = []
    total = 0
    done = False

    while not done:
        path.append(env._to_state(state))
        action = np.argmax(Q[state])
        state, reward, done, _ = env.step(action)
        total += reward

    path.append(env._to_state(state))
    print("Test path:", path)
    print("Test total reward:", total)


# -------- Main --------
if __name__ == "__main__":
    env = GridWorldEnv(size=5)

    Q, rewards = q_learning_train(env)
    plot_rewards(rewards)

    policy = extract_policy(Q)
    print("Learned Policy:")
    env.render(policy)

    print("Testing agent...")
    test_agent(env, Q)
