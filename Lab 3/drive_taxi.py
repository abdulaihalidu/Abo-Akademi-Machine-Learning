import gymnasium as gym
import numpy as np
import random
from IPython.display import clear_output
import time

# Training
# Initialize the environment
env = gym.make('Taxi-v3', render_mode="human")

# Initialize the Q-table
q_table = np.zeros([env.observation_space.n, env.action_space.n])

# Hyperparameters
alpha = 0.4
gamma = 0.7
epsilon = 0.1

# For tracking metrics
all_epochs = []
all_penalties = []
all_losses = []  # Track "loss" for each episode

num_episodes = 1001

# Training the agent
for i in range(1, num_episodes):  
    state = env.reset()[0]
    epochs, penalties, reward, = 0, 0, 0
    terminated, truncated = False, False
    total_loss = 0  # Track total "loss" for the episode
    
    while not terminated and not truncated:
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()  # Explore action space
        else:
            action = np.argmax(q_table[state])  # Exploit learned values
        next_state, reward, terminated, truncated, info = env.step(action)
        if (terminated or truncated): 
            print(terminated, truncated)
        
        old_value = q_table[state, action]
        next_max = np.max(q_table[next_state])
        
        # Update the Q-value
        new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
        q_table[state, action] = new_value

        # Update total "loss"
        total_loss += np.abs(new_value - old_value)

        if reward == -10:
            penalties += 1

        state = next_state
        epochs += 1
        
    all_losses.append(total_loss)  # Append total loss for this episode
    print(f"Iteration: {i}")
    if i % 100 == 0:
        clear_output(wait=True)
        print(f"Episode: {i}")

print("Training finished.\n")
print("Saving learned Q-table to disk ...") 
np.save('q_table.npy', q_table)

# Evaluate agent's performance after Q-learning
total_epochs, total_penalties = 0, 0
episodes = 100

for _ in range(episodes):
    state = env.reset()[0]
    epochs, penalties, reward = 0, 0, 0
    terminated, truncated = False, False
    clear_output(wait=True)  # Clear the output to make the visualization clean
    env.render()  # Render the initial state
    time.sleep(1)  # Pause for a short time to visualize the state
    
    while ((not terminated) and (not truncated)):
        action = np.argmax(q_table[state])
        state, reward, terminated, truncated, info = env.step(action)
        clear_output(wait=True)
        env.render()
        time.sleep(1)  # Adjust sleep time as needed for visualization speed

        if reward == -10:
            penalties += 1

        epochs += 1

    total_epochs += epochs
    total_penalties += penalties

    time.sleep(1)  # Pause at the end of each episode

print(f"Results after {episodes} episodes:")
print(f"Average timesteps per episode: {total_epochs / episodes}")
print(f"Average penalties per episode: {total_penalties / episodes}")  

# Testing 
num_episodes = 100
num_plays = 10

for _ in range(num_episodes):
    state = env.reset()[0]
    epochs, penalties, reward = 0, 0, 0
    terminated, truncated = False, False
    clear_output(wait=True)  # Clear the output to make the visualization clean
    env.render()  # Render the initial state
    time.sleep(1)  # Pause for a short time to visualize the state
    
    while (not terminated and not truncated):
        action = np.argmax(q_table[state])
        state, reward, terminated, truncated, info = env.step(action)
        clear_output(wait=True)
        env.render()
        time.sleep(0.1)  # Adjust sleep time as needed for visualization speed

        if reward == -10:
            penalties += 1

        epochs += 1

    # total_epochs += epochs
    # total_penalties += penalties

    time.sleep(1.5)  # Pause at the end of each episode

