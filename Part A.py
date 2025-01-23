# Asaf Kessler 316539196

"""
Data Description-
Columns:
trial_number: Identifies each trial.
reward: Indicates whether a reward was given (1) or not (0).
side: Indicates which side the mouse interacted with (0 or 1).
reward_prob: The probability of reward (20 or 80), which switches in blocks.
green: A vector representing photometry signals (6001 data points per trial).
"""

# Import required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind

# Load the dataset
data = pd.read_pickle('Python_data.pkl')

# Separate data into reward and omission trials
reward_trials = data[data['reward'] == 1]
omission_trials = data[data['reward'] == 0]

''' Question 1. Responses to rewards and omissions '''
# Question 1: Dopamine Analysis - Responses to rewards and omissions

# Extract photometry signals
reward_signals = np.stack(reward_trials['green'].to_numpy())
omission_signals = np.stack(omission_trials['green'].to_numpy())

# Step 1: Heatmaps for reward and omission trials
plt.figure(figsize=(12, 6))

# Heatmap for reward trials
plt.subplot(1, 2, 1)
sns.heatmap(reward_signals, cmap='viridis', cbar_kws={'label': 'Signal Intensity'})
plt.title('Reward Trials')
plt.xlabel('Time (ms)')
plt.ylabel('Trial Number')

# Heatmap for omission trials
plt.subplot(1, 2, 2)
sns.heatmap(omission_signals, cmap='viridis', cbar_kws={'label': 'Signal Intensity'})
plt.title('Omission Trials')
plt.xlabel('Time (ms)')
plt.ylabel('Trial Number')

plt.tight_layout()
plt.savefig('heatmaps.png')
plt.show()

# Step 2: Average dopamine response traces
# Calculate the average dopamine response traces
time = np.arange(reward_signals.shape[1])  # Time in milliseconds
reward_mean = np.mean(reward_signals, axis=0)
omission_mean = np.mean(omission_signals, axis=0)
reward_sem = np.std(reward_signals, axis=0) / np.sqrt(reward_signals.shape[0])
omission_sem = np.std(omission_signals, axis=0) / np.sqrt(omission_signals.shape[0])

# Plot the average dopamine response traces
plt.figure(figsize=(10, 6))
plt.plot(time, reward_mean, label='Reward Trials', color='blue')
plt.fill_between(time, reward_mean - reward_sem, reward_mean + reward_sem, alpha=0.3, color='blue')
plt.plot(time, omission_mean, label='Omission Trials', color='orange')
plt.fill_between(time, omission_mean - omission_sem, omission_mean + omission_sem, alpha=0.3, color='orange')
plt.title('Average Dopamine Response Traces')
plt.xlabel('Time (ms)')
plt.ylabel('Signal Intensity')
plt.legend()
plt.grid(True)
plt.savefig('average_traces.png')
plt.show()

# Step 3: Bar graph and statistical test
# Calculate the average signal in the first 1-second window (1000 ms)
window_size = 1000  # First 1000 ms
reward_avg_response = np.mean(reward_signals[:, :window_size], axis=1)
omission_avg_response = np.mean(omission_signals[:, :window_size], axis=1)

# Perform a t-test to compare the two groups
t_stat, p_value = ttest_ind(reward_avg_response, omission_avg_response)

# Plot a bar graph showing the average responses
plt.figure(figsize=(8, 6))
plt.bar(['Reward', 'Omission'], [np.mean(reward_avg_response), np.mean(omission_avg_response)],
        yerr=[np.std(reward_avg_response) / np.sqrt(len(reward_avg_response)),
              np.std(omission_avg_response) / np.sqrt(len(omission_avg_response))],
        capsize=5, color=['blue', 'orange'])
plt.title('Average Dopamine Responses (1-second Window)')
plt.ylabel('Signal Intensity')
plt.grid(axis='y')

# Display the p-value on the plot
plt.text(0.5, max(np.mean(reward_avg_response), np.mean(omission_avg_response)),
         f'p = {p_value:.3e}', ha='center', va='bottom', fontsize=12)

plt.savefig('bar_graph.png')
plt.show()

# Statistical test result
print(f"T-statistic: {t_stat}, P-value: {p_value}")

''' Question 2. Responses to reward prediction errors '''

# Part a.1 - Reward

# Step 1: Separate data by reward probability
prob_80_trials = data[data['reward_prob'] == 80]
prob_20_trials = data[data['reward_prob'] == 20]

# Step 2: Extract trials where a reward was given
prob_80_reward = prob_80_trials[prob_80_trials['reward'] == 1]
prob_20_reward = prob_20_trials[prob_20_trials['reward'] == 1]

# Step 3: Extract photometry signals
prob_80_reward_signals = np.stack(prob_80_reward['green'].to_numpy())
prob_20_reward_signals = np.stack(prob_20_reward['green'].to_numpy())

# Calculate means
time = np.arange(prob_80_reward_signals.shape[1])  # Time in milliseconds
prob_80_reward_mean = np.mean(prob_80_reward_signals, axis=0)
prob_20_reward_mean = np.mean(prob_20_reward_signals, axis=0)

# Step 4: Plot the average reward response traces
plt.figure(figsize=(10, 6))

# Plot reward traces
plt.plot(time, prob_80_reward_mean, label='Reward (p=0.8)', color='blue')
plt.plot(time, prob_20_reward_mean, label='Reward (p=0.2)', color='green')

# Add titles and labels
plt.title('Average Dopamine Response Traces for Reward Probabilities (0.2 and 0.8)')
plt.xlabel('Time (ms)')
plt.ylabel('Signal Intensity')
plt.legend()
plt.grid(True)

# Save the plot as an image file
plot_path = 'reward_probabilities_plot.png'
plt.savefig(plot_path)

# Display the plot
plt.show()

print(f"Plot saved as {plot_path}")

# Part a.2 - Omission

# Step 1: Separate data by reward probability
prob_80_trials = data[data['reward_prob'] == 80]
prob_20_trials = data[data['reward_prob'] == 20]

# Step 2: Extract trials where a reward was omitted
prob_80_omission = prob_80_trials[prob_80_trials['reward'] == 0]
prob_20_omission = prob_20_trials[prob_20_trials['reward'] == 0]

# Step 3: Extract photometry signals
prob_80_omission_signals = np.stack(prob_80_omission['green'].to_numpy())
prob_20_omission_signals = np.stack(prob_20_omission['green'].to_numpy())

# Calculate means
time = np.arange(prob_80_omission_signals.shape[1])  # Time in milliseconds
prob_80_omission_mean = np.mean(prob_80_omission_signals, axis=0)
prob_20_omission_mean = np.mean(prob_20_omission_signals, axis=0)

# Step 4: Plot the average omission response traces
plt.figure(figsize=(10, 6))

# Plot omission traces
plt.plot(time, prob_80_omission_mean, label='Omission (p=0.8)', color='orange')
plt.plot(time, prob_20_omission_mean, label='Omission (p=0.2)', color='red')

# Add titles and labels
plt.title('Average Dopamine Response Traces for Omission Probabilities (0.2 and 0.8)')
plt.xlabel('Time (ms)')
plt.ylabel('Signal Intensity')
plt.legend()
plt.grid(True)

# Save the plot as an image file
plot_path = 'omission_probabilities_plot.png'
plt.savefig(plot_path)

# Display the plot
plt.show()

print(f"Plot saved as {plot_path}")

# Part b -  average dopamine response traces based on trial history.

'''   (Option One)   '''

# Add a column to indicate the reward outcome of the previous trial
data['prev_reward'] = data['reward'].shift(1)

# Drop the first trial as it has no history
data_with_history = data.dropna(subset=['prev_reward'])

# Separate data by current and previous trial outcomes
reward_prev_reward = data_with_history[(data_with_history['reward'] == 1) & (data_with_history['prev_reward'] == 1)]
reward_prev_omission = data_with_history[(data_with_history['reward'] == 1) & (data_with_history['prev_reward'] == 0)]
omission_prev_reward = data_with_history[(data_with_history['reward'] == 0) & (data_with_history['prev_reward'] == 1)]
omission_prev_omission = data_with_history[(data_with_history['reward'] == 0) & (data_with_history['prev_reward'] == 0)]

# Extract photometry signals and calculate average traces
# Reward after reward
reward_prev_reward_signals = np.stack(reward_prev_reward['green'].to_numpy())
reward_prev_reward_mean = np.mean(reward_prev_reward_signals, axis=0)

# Reward after omission
reward_prev_omission_signals = np.stack(reward_prev_omission['green'].to_numpy())
reward_prev_omission_mean = np.mean(reward_prev_omission_signals, axis=0)

# Omission after reward
omission_prev_reward_signals = np.stack(omission_prev_reward['green'].to_numpy())
omission_prev_reward_mean = np.mean(omission_prev_reward_signals, axis=0)

# Omission after omission
omission_prev_omission_signals = np.stack(omission_prev_omission['green'].to_numpy())
omission_prev_omission_mean = np.mean(omission_prev_omission_signals, axis=0)

# Time axis
time = np.arange(reward_prev_reward_signals.shape[1])

# Plot the average dopamine response traces for trial history
plt.figure(figsize=(12, 6))

# Plot for reward trials
plt.plot(time, reward_prev_reward_mean, label='Reward | Prev Reward', color='blue')
plt.plot(time, reward_prev_omission_mean, label='Reward | Prev Omission', color='green')

# Plot for omission trials
plt.plot(time, omission_prev_reward_mean, label='Omission | Prev Reward', color='orange', linestyle='--')
plt.plot(time, omission_prev_omission_mean, label='Omission | Prev Omission', color='red', linestyle='--')

# Add titles and labels
plt.title('Average Dopamine Response Traces Based on Trial History')
plt.xlabel('Time (ms)')
plt.ylabel('Signal Intensity')
plt.legend()
plt.grid(True)

# Save the plot as an image file
plot_path = 'Average_Dopamine_Response_Trial_History.png'
plt.savefig(plot_path)

# Display the plot
plt.show()

# Part b -  average dopamine response traces based on trial history.

'''   (Option Two)   '''

# Reload the dataset with the newly uploaded file
data = pd.read_pickle('Python_data.pkl')

# Re-define function for mean activity calculation over trials
def bar_graph_mean_activity(group_data, normalize=False):
    onset = 1000
    mean_neurons_activity = np.mean(group_data['green'], axis=0)  # Calculate the mean activity vector (mean of all trials)
    mean_activity_vec = mean_neurons_activity[onset:onset+1001]  # Extract a second of activity after onset

    mean_activity_array = np.array(group_data['green'].tolist())
    mean_activity_in_time = np.mean(mean_activity_array[:, onset:onset+1001], axis=1) # mean of each trial in time
    mean_single_val = np.mean(mean_activity_vec)
    return mean_single_val, mean_activity_in_time


# Separate data into right and left side based on reward and omission
r_data_omission = data[(data["side"] == 1) & (data["reward"] == 0)]
l_data_omission = data[(data["side"] == 0) & (data["reward"] == 0)]
r_data_reward = data[(data["side"] == 1) & (data["reward"] == 1)]
l_data_reward = data[(data["side"] == 0) & (data["reward"] == 1)]

# Define block transitions (v_lines) based on the provided changes
v_lines = [41, 82, 125, 168, 210, 247, 285, 327]  # Example block changes

# Compute mean activities for each group
_, mean_r_omission = bar_graph_mean_activity(r_data_omission, False)
_, mean_l_omission = bar_graph_mean_activity(l_data_omission, False)
_, mean_r_reward = bar_graph_mean_activity(r_data_reward, False)
_, mean_l_reward = bar_graph_mean_activity(l_data_reward, False)

# Define the window size for the rolling average
window = 6

# Smooth the reward trials using a rolling mean
s1re = pd.Series(mean_r_reward).rolling(window=window, min_periods=1).mean()  # Right Side (Reward)
s0re = pd.Series(mean_l_reward).rolling(window=window, min_periods=1).mean()  # Left Side (Reward)

# Smooth the omission trials using a rolling mean
s1om = pd.Series(mean_r_omission).rolling(window=window, min_periods=1).mean()  # Right Side (Omission)
s0om = pd.Series(mean_l_omission).rolling(window=window, min_periods=1).mean()  # Left Side (Omission)

# Plot results
plt.figure(figsize=(19, 6))

# Plot for reward trials separated by side
plt.subplot(1, 2, 1)
plt.plot(r_data_reward['trial_number'].unique(), s1re, label="Right Side (Reward)", color="blue")
plt.plot(l_data_reward['trial_number'].unique(), s0re, label="Left Side (Reward)", color="green")
plt.title("Smoothed (by 6) Mean Activity for Reward Trials by Side")
plt.xlabel("Trial Number")
plt.ylabel("Mean Photometry Activity")
for line in v_lines:
    plt.axvline(x=line, color="gray", linestyle="--", alpha=0.5)
plt.legend()
plt.grid(True)

# Plot for omission trials separated by side
plt.subplot(1, 2, 2)
plt.plot(r_data_omission['trial_number'].unique(), s1om, label="Right Side (Omission)", color="orange")
plt.plot(l_data_omission['trial_number'].unique(), s0om, label="Left Side (Omission)", color="red")
plt.title("Smoothed (by 6) Mean Activity for Omission Trials by Side - ")
plt.xlabel("Trial Number")
plt.ylabel("Mean Photometry Activity")
for line in v_lines:
    plt.axvline(x=line, color="gray", linestyle="--", alpha=0.5)
plt.legend()
plt.grid(True)

# Display the smoothed plots
plt.tight_layout()

# Save the plot as an image file
plot_path = 'last.png'
plt.savefig(plot_path)

plt.show()
