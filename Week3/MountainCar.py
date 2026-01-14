import gymnasium as gym
import math
import random
import matplotlib.pyplot as plt
from collections import deque, namedtuple
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
#parameters
BATCH_SIZE = 64
GAMMA = 0.99
EPS_START = 1.0
EPS_END = 0.01
EPS_DECAY = 0.995   # Slow decay is crucial for Mountain Car 
TAU = 0.005         # Soft update rate for target network
LR = 1e-3           # Learning rate

# device setup (uses GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#neural network
class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        # A deeper network helps capture the complex curve needed to solve this
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)


env = gym.make("MountainCar-v0")

# Structure to store experiences
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# Get number of actions and observations
n_actions = env.action_space.n
state, _ = env.reset()
n_observations = len(state)

#networks and optimizers
policy_net = DQN(n_observations, n_actions).to(device)
target_net = DQN(n_observations, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict()) # Synchronize initially

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(10000)

steps_done = 0

#helper functions
def select_action(state):
    global steps_done
    sample = random.random()
    
    # Epsilon decay formula
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / 10000) # Decay over ~10000 steps logic
    
    steps_done += 1
    
    # Epsilon-Greedy Logic
    if sample > eps_threshold:
        with torch.no_grad():
            # "Exploit": Pick the action with the largest expected reward
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        # "Explore": Pick a random action
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return

    # 1. Sample a batch from memory
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    # Mask for non-final states (states that didn't end the game)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])

    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # 2. Compute Q(s, a) - the model's current prediction
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # 3. Compute V(s_{t+1}) for all next states.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]
    
    # Compute the expected Q values (Bellman equation)
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # 4. Compute Loss (Huber loss is more robust to outliers)
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # 5. Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping (prevents exploding gradients)
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()

#training loop
num_episodes = 1000
episode_rewards = []

print("Starting training (this may take a few minutes)...")

for i_episode in range(num_episodes):
    state, _ = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    total_reward = 0
    
    for t in range(200): # Limit steps per episode to speed up training
        # Select and perform an action
        action = select_action(state)
        observation, reward, terminated, truncated, _ = env.step(action.item())
        
        # Reward shaping (Optional but helpful):
        # Give a small bonus if the car moves higher up the hill
        # position = observation[0]
        # if position > -0.5: reward += (0.5 + position)

        reward = torch.tensor([reward], device=device)
        done = terminated or truncated

        if terminated:
            next_state = None
        else:
            next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

        # Store the transition in memory
        memory.push(state, action, next_state, reward)

        # Move to the next state
        state = next_state
        total_reward += reward.item()

        # Perform one step of the optimization (on the policy network)
        optimize_model()

        # Soft update of the target network's weights
        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
        target_net.load_state_dict(target_net_state_dict)

        if done:
            break
            
    episode_rewards.append(total_reward)
    
    # Print progress every 20 episodes
    if i_episode % 40 == 0:
        print(f"Episode {i_episode}: Total Reward {total_reward:.2f}")

print("Training Complete")
env.close()

# Plot results
plt.figure(figsize=(10,5))
plt.plot(episode_rewards)
plt.title("DQN Performance on MountainCar")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.show()