import gymnasium as gym
import numpy as np
import pandas as pd
import random
from IPython.display import clear_output
import argparse
import time


class TaxiAgent:
    def __init__(self, env_name='Taxi-v3', q_table_path=None, logging=False, render=False):
        self.render = render
        self.log = logging
        self.env = gym.make(env_name, render_mode="human" if render else None)
        self.q_table = self.load_q_table(q_table_path) if q_table_path else np.zeros(
            [self.env.observation_space.n, self.env.action_space.n])

        # Hyperparameters
        self.alpha = 0.6
        self.gamma = 0.8
        self.epsilon = 0.1

        # Metrics
        self.all_losses = []  # Tracks "loss" for each episode

    def load_q_table(self, path):
        try:
            q_table = np.load(path)
            if self.log:
                print("Loaded q-table successfully!")
            return q_table
        except FileNotFoundError:
            if self.log:
                print("File not found. Initializing a new Q-table.")
            return np.zeros([self.env.observation_space.n, self.env.action_space.n])

    def train(self, num_episodes=1001, weights_file_name=None):
        training_error = []
        for episode in range(1, num_episodes):
            state = self.env.reset()[0]
            total_loss, penalties = 0, 0

            terminated = False
            while not terminated:
                action = random.choice(range(self.env.action_space.n)) if random.uniform(
                    0, 1) < self.epsilon else np.argmax(self.q_table[state])
                next_state, reward, terminated, _, _ = self.env.step(action)

                old_value = self.q_table[state, action]
                next_max = np.max(self.q_table[next_state])
                new_value = (1 - self.alpha) * old_value + \
                    self.alpha * (reward + self.gamma * next_max)

                self.q_table[state, action] = new_value
                total_loss += np.abs(new_value - old_value)

                penalties += reward == -10
                state = next_state
            self.epsilon = (self.epsilon - 0.001) if (self.epsilon >
                                                      0.001) else (self.epsilon)

            self.all_losses.append(total_loss)
            training_error.append({"Episode": episode, "Loss": total_loss})

            if self.log and episode % 100 == 0:
                clear_output(wait=True)
                print(f"Episode: {episode}")
            if self.log and terminated:
                print("Passenger Dropped off at Destination.")

        if weights_file_name:
            np.save(f'{weights_file_name}.npy', self.q_table)
        print("Training finished.\nSaving learned Q-table to disk...")
        # Save training loss to file
        training_error_df = pd.DataFrame(training_error)
        training_error_df.to_csv("training_errors.csv", index=False)

    def drive(self, episodes=5, sleep_time=1):
        total_epochs, total_penalties = 0, 0

        for _ in range(episodes):
            state = self.env.reset()[0]
            epochs, penalties = 0, 0
            terminated = False
            # Sleep momentatariy after setting up a new environment
            time.sleep(0.5)

            while not terminated:
                action = np.argmax(self.q_table[state])
                state, reward, terminated, _, _ = self.env.step(action)
                time.sleep(sleep_time)

                penalties += reward == -10
                epochs += 1

            total_epochs += epochs
            total_penalties += penalties
            # Sleep momentarily after deliverying the passenger.
            time.sleep(0.9)

        print(f"Results after {episodes} episodes:\nAverage timesteps per episode: {
              total_epochs / episodes}\nAverage penalties per episode: {total_penalties / episodes}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train and drive a Taxi Agent.")
    parser.add_argument('-t', '--train', action='store_true',
                        help='Train the taxi')
    parser.add_argument('-d', '--drive', action='store_true',
                        help='Drive the taxi')
    parser.add_argument('-w', '--weights', type=str,
                        help='Pre-trained weights file name')
    parser.add_argument('-l', '--log', action='store_true',
                        help='Enable messages logging')
    parser.add_argument('-e', '--drive_episodes', type=int,
                        default=5, help='Number of episodes to drive taxi')
    parser.add_argument('-s', '--save_weights', type=str,
                        help='Name of file to save weights to')
    parser.add_argument('-r', '--render', action='store_true',
                        help='Enable rendering')
    parser.add_argument('-st', '--sleep_time', type=float,
                        default=0.1, help='Sleep time between each move')

    args = parser.parse_args()

    agent = TaxiAgent(q_table_path=args.weights,
                      logging=args.log, render=args.render)

    if args.train:
        if not args.save_weights:
            print("Please provide file name to save weights to.")
        else:
            agent.train(num_episodes=1001, weights_file_name=args.save_weights)
    if args.drive:
        agent.drive(episodes=args.drive_episodes, sleep_time=args.sleep_time)
