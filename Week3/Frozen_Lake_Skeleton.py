import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

 # Environment
env = gym.make(
    "FrozenLake-v1",
    map_name="4x4",
    is_slippery=True
)

n_states = env.observation_space.n
n_actions = env.action_space.n

# -----------------------------
# Q-learning parameters
# -----------------------------

"""alpha =  0.1          # learning rate
gamma =  0.99         # discount factor

epsilon =   1       # initial exploration rate
epsilon_min =  0.01    # minimum exploration rate
epsilon_decay = 0.995   # exploration decay factor

num_episodes =   10000  
max_steps =    100   
while i was using the graph was a horizontal line due to less episode and a slow decay""" 
alpha =  0.1          # learning rate
gamma =  0.99         # discount factor

epsilon =   1       # initial exploration rate
epsilon_min =  0.01    # minimum exploration rate
epsilon_decay = 0.9995   # exploration decay factor

num_episodes =   20000  
max_steps =    200    

# -----------------------------
# Q-table
# -----------------------------
Q = np.zeros((n_states, n_actions))

# -----------------------------
# Tracking performance
# -----------------------------
success_window = deque(maxlen=100)
success_rates = []

# -----------------------------
# Training loop
# -----------------------------
for episode in range(num_episodes):
    state, _ = env.reset()
    done = False
    success = 0

    for _ in range(max_steps):
        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q[state])

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        Q[state, action] += alpha * (
            reward + gamma * np.max(Q[next_state]) - Q[state, action]
        )

        state = next_state

        if done:
            success = reward
            break

    epsilon = max(epsilon_min, epsilon * epsilon_decay)

    success_window.append(success)
    success_rates.append(np.mean(success_window))

# -----------------------------
# Plot learning curve
# -----------------------------
plt.figure()
plt.plot(success_rates)
plt.xlabel("Episode")
plt.ylabel("Success Rate (last 100 episodes)")
plt.title("FrozenLake Q-Learning Performance")
plt.show()