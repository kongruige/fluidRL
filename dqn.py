import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt
from fluid_env import FluidEnv # Import our custom environment

# --- Set device ---
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using CUDA (NVIDIA GPU)")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS (Apple GPU)")
else:
    device = torch.device("cpu")
    print("Using CPU")

# --- 1. Define the Q-Network (Lec 7) ---
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.layer1 = nn.Linear(state_size, 64)
        self.layer2 = nn.Linear(64, 64)
        self.layer3 = nn.Linear(64, action_size)
        
    def forward(self, state):
        x = self.layer1(state)
        x = F.relu(x)
        x = self.layer2(x)
        x = F.relu(x)
        return self.layer3(x)
        

# --- 2. Define the Replay Buffer (Lec 8) ---
class ReplayBuffer:
    def __init__(self, buffer_size, batch_size):
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size

    def push(self, state, action, reward, next_state, done):
        """Save an experience tuple."""
        experience = (state, action, reward, next_state, done)
        self.memory.append(experience)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, self.batch_size)

        states = torch.from_numpy(np.vstack([e[0] for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e[1] for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e[2] for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e[3] for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e[4] for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
        
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

# --- 3. Define the DQN Agent ---

class DQNAgent:
    def __init__(self, state_size, action_size):
        # --- Hyperparameters ---
        self.state_size = state_size
        self.action_size = action_size
        
        self.buffer_size = 100000
        self.batch_size = 64
        self.gamma = 0.99
        self.lr = 1e-3
        self.update_every = 4
        self.tau = 1e-3
        
        # --- Q-Network (Lec 7) ---
        # [# <-- FIX 1] Move both networks to the correct device
        self.q_network = QNetwork(state_size, action_size).to(device)
        self.target_network = QNetwork(state_size, action_size).to(device)
        
        # [# <-- FIX 2] Copy the weights from q_network to target_network
        # This ensures they start identical.
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.lr)

        # --- Replay Memory (Lec 8) ---
        self.memory = ReplayBuffer(self.buffer_size, self.batch_size)
        
        self.t_step = 0

    def select_action(self, state, epsilon):
        """Selects an action using an epsilon-greedy policy."""
        rand_num = np.random.rand()
        
        if rand_num > epsilon:
            # --- Exploitation ---
            state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(device)
            
            # [# <-- FIX 3] Set network to evaluation mode (good practice)
            self.q_network.eval() 
            with torch.no_grad():
                action_values = self.q_network(state_tensor)
            # [# <-- FIX 4] Set network back to training mode
            self.q_network.train() 

            action = np.argmax(action_values.cpu().data.numpy())
            return action
        else:
            # --- Exploration ---
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences):
        """
        Update Q-network weights using a batch of experiences.
        """
        states, actions, rewards, next_states, dones = experiences

        # 1. Get Q-values for NEXT states from the TARGET network
        Q_targets_next = self.target_network(next_states).detach().max(1)[0].unsqueeze(1)
        
        # 2. Calculate the 'Q_targets' (the TD Target y_i)
        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))
        
        # 3. Get the 'Q_expected' from the MAIN network
        Q_expected = self.q_network(states).gather(1, actions)

        # 4. Calculate the loss
        loss = F.mse_loss(Q_expected, Q_targets)
        
        # 5. Perform gradient descent
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # --- Soft update target network ---
        self.soft_update(self.q_network, self.target_network)

    def soft_update(self, local_model, target_model):
        """
        Softly copies weights from the local_model to the target_model.
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau*local_param.data + (1.0-self.tau)*target_param.data)

    def step(self, state, action, reward, next_state, done):
        """Save experience in replay memory, and learn if it's time."""
        self.memory.push(state, action, reward, next_state, done)
        
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0:
            if len(self.memory) > self.batch_size:
                experiences = self.memory.sample()
                self.learn(experiences)

# --- 4. Main Training Loop (Provided) ---

def train_dqn():
    env = FluidEnv()
    agent = DQNAgent(state_size=2, action_size=5)

    # Training Hyperparameters
    num_episodes = 2000
    max_t = env.max_steps
    start_epsilon = 1.0
    end_epsilon = 0.01
    epsilon_decay = 0.995
    epsilon = start_epsilon
    
    scores = []
    scores_window = deque(maxlen=100)

    for i_episode in range(1, num_episodes + 1):
        state = env.reset()
        score = 0
        for t in range(max_t):
            action = agent.select_action(state, epsilon)
            next_state, reward, done = env.step(action)
            
            agent.step(state, action, reward, next_state, done)
            
            state = next_state
            score += reward
            if done:
                break
        
        scores_window.append(score)
        scores.append(score)
        
        epsilon = max(end_epsilon, epsilon_decay * epsilon)
        
        if i_episode % 100 == 0:
            print(f'\rEpisode {i_episode}\tAverage Score: {np.mean(scores_window):.2f}')
            
    print('Training Complete!')
    
    # Plotting Results
    plt.figure(figsize=(10, 5))
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Total Reward')
    plt.xlabel('Episode #')
    plt.title('DQN Training: Total Reward per Episode')
    plt.savefig('dqn_rewards.png')
    plt.show()

if __name__ == "__main__":
    train_dqn()