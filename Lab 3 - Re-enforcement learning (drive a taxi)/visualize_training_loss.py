import pandas as pd
import matplotlib.pyplot as plt


# Load data
data_path = './training_errors.csv'

data = pd.read_csv(data_path)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(data['Episode'], data['Loss'], label='Loss per Episode')
plt.xlabel('Episode')
plt.ylabel('Loss')
plt.title('Loss vs. Episode')
plt.legend()
plt.grid(True)
plt.show()
