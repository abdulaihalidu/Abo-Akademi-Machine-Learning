import gymnasium as gym
import numpy as np
import random
from IPython.display import clear_output
import time

class TaxiAgent:
    def __init__(self, env_name='Taxi-v3', q_table_path=None):
        self.env = gym.make(env_name, render_mode="human")
        if q_table_path is not None:
            self.q_table = self.load_q_table(q_table_path)
        else:
            self.q_table = np.zeros([self.env.observation_space.n, self.env.action_space.n])
        
        # Hyperparameters
        self.alpha = 0.4
        self.gamma = 0.7
        self.epsilon = 0.1
        
        # Metrics
        self.all_epochs = []
        self.all_penalties = []
        self.all_losses = []  # Track "loss" for each episode

    def load_q_table(self, path): # To avoid retraining all the time, I have saved the trained q-table
        try:
            q_table = np.load(path)
            print("Loaded q-table successfully!")
            return q_table
        except FileNotFoundError:
            print("File not found. Initializing a new Q-table.")
            return np.zeros([self.env.observation_space.n, self.env.action_space.n])

    def train(self, num_episodes=1001):
        for i in range(1, num_episodes):
            state = self.env.reset()[0]
            epochs, penalties, reward = 0, 0, 0
            terminated, truncated = False, False
            total_loss = 0  # Track total "loss" for the episode
            
            while not terminated and not truncated:
                if random.uniform(0, 1) < self.epsilon:
                    action = self.env.action_space.sample()  # Explore action space
                else:
                    action = np.argmax(self.q_table[state])  # Exploit learned values
                next_state, reward, terminated, truncated, info = self.env.step(action)
                
                old_value = self.q_table[state, action]
                next_max = np.max(self.q_table[next_state])
                
                # Update the Q-value
                new_value = (1 - self.alpha) * old_value + self.alpha * (reward + self.gamma * next_max)
                self.q_table[state, action] = new_value

                # Update total "loss"
                total_loss += np.abs(new_value - old_value)

                if reward == -10:
                    penalties += 1

                state = next_state
                epochs += 1
            
            self.all_losses.append(total_loss)  # Append total loss for this episode
    
            if i % 100 == 0:
                clear_output(wait=True)
                print(f"Episode: {i}")

        print("Training finished.\n")
        print("Saving learned Q-table to disk ...")
        np.save('q_table.npy', self.q_table)

    def drive(self, episodes=5, sleep_time=0.1):
        total_epochs, total_penalties = 0, 0

        for _ in range(episodes):
            state = self.env.reset()[0]
            epochs, penalties, reward = 0, 0, 0
            terminated, truncated = False, False
            clear_output(wait=True)
            self.env.render()
            time.sleep(sleep_time)
            
            while not terminated and not truncated:
                action = np.argmax(self.q_table[state])
                state, reward, terminated, truncated, info = self.env.step(action)
                clear_output(wait=True)
                self.env.render()
                time.sleep(sleep_time)  # Adjust sleep time as needed for visualization speed

                if reward == -10:
                    penalties += 1

                epochs += 1

            total_epochs += epochs
            total_penalties += penalties

            time.sleep(1.5) # Sleep momentarily after each episode.

        print(f"Results after {episodes} episodes:")
        print(f"Average timesteps per episode: {total_epochs / episodes}")
        print(f"Average penalties per episode: {total_penalties / episodes}")

if __name__ == "__main__":
    # Let's use the weights of the pretrained q_table
    q_table_path = './q_table.npy'
    # Initialize a taxi agent
    agent = TaxiAgent(q_table_path=q_table_path)
    # Train the taxi
    #agent.train(num_episodes=1001)
    # Initiate self-drive
    agent.drive(20, sleep_time=0.1)
