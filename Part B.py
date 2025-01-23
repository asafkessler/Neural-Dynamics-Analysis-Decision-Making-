# Asaf Kessler 316539196

# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import uniform_filter1d
from sklearn.linear_model import LinearRegression
from scipy.stats import linregress

# Load the data
file_path = 'Python_C5843.pkl'
data = pd.read_pickle(file_path)

''' Question Number 1 -  Behavior Analysis'''

# Calculate the number of trials where the monkey selected the high-probability target
high_prob_selections = data[(data['choice'] == 1) & (data['R_prob'] > data['U_prob'])].shape[0] + \
                       data[(data['choice'] == 0) & (data['U_prob'] > data['R_prob'])].shape[0]

total_trials = data.shape[0]

# Calculate the proportion of trials where the monkey selected the high-probability target
optimal_behavior_percentage = (high_prob_selections / total_trials) * 100

# Optimal behavior maximizes reward by always choosing the higher probability target
optimal_behavior_description = "Optimal behavior is to always select the target with the higher reward probability."

# Results
print("Optimal Behavior Description:", optimal_behavior_description)
print("Number of Trials with High-Probability Target Selection:", high_prob_selections)
print("Total Number of Trials:", total_trials)
print("Proportion of High-Probability Selections:", f"{optimal_behavior_percentage:.2f}%")

''' Question Number 2 - Raster Plots '''

# Reload the data from the file
file_path = 'Python_C5843.pkl'
data = pd.read_pickle(file_path)

# Add 'high_prob_target' column to indicate which target had the higher probability
data['high_prob_target'] = (data['R_prob'] > data['U_prob']).astype(int)

# Filter trials where high probability was RIGHT or UP
right_high_trials = data[data['high_prob_target'] == 1]
up_high_trials = data[data['high_prob_target'] == 0]


# Function to create a raster plot
def plot_raster(spike_data, title):
    plt.figure(figsize=(12, 6))
    for i, spikes in enumerate(spike_data):
        spike_times = np.where(spikes)[0]
        plt.vlines(spike_times, i + 0.5, i + 1.5)
    plt.title(title)
    plt.xlabel('Time (ms)')
    plt.ylabel('Trial')
    plt.savefig(title + '.png')
    plt.show()


# Create raster plots for RIGHT and UP high probability trials
plot_raster(right_high_trials['spikes'], "Raster Plot: High Probability = RIGHT")
plot_raster(up_high_trials['spikes'], "Raster Plot: High Probability = UP")

''' Question Number 3 - PSTH Plots '''

# Reload the data from the file
file_path = 'Python_C5843.pkl'
data = pd.read_pickle(file_path)

# Add 'high_prob_target' column to indicate which target had the higher probability
data['high_prob_target'] = (data['R_prob'] > data['U_prob']).astype(int)

# Filter trials where high probability was RIGHT or UP
right_high_trials = data[data['high_prob_target'] == 1]
up_high_trials = data[data['high_prob_target'] == 0]


# Function to calculate and plot PSTH
def calculate_and_plot_psth(trials, title):
    # Convert list of spike arrays to a 2D array for easier averaging
    spike_matrix = np.array(trials['spikes'].tolist())

    # Calculate the PSTH by averaging across trials and scaling to spikes/second
    psth = spike_matrix.mean(axis=0) * 1000  # Convert to spikes per second

    # Smooth the PSTH using a moving average (100 ms window)
    smoothed_psth = uniform_filter1d(psth, size=100)

    # Plot the raw and smoothed PSTH
    plt.figure(figsize=(12, 6))
    plt.plot(psth, label='Raw PSTH', alpha=0.7)
    plt.plot(smoothed_psth, label='Smoothed PSTH (100 ms)', linewidth=2)
    plt.title(f'PSTH for {title}')
    plt.xlabel('Time (ms)')
    plt.ylabel('Firing Rate (spikes/s)')
    plt.legend()
    plt.savefig(title + '.png')
    plt.show()


# Calculate and plot the PSTH for RIGHT high-probability trials
calculate_and_plot_psth(right_high_trials, "High Probability = RIGHT")

''' Question Number 4 - PSTH Plots '''

# Function to calculate and return smoothed PSTH
def calculate_smoothed_psth(trials):
    spike_matrix = np.array(trials['spikes'].tolist())
    psth = spike_matrix.mean(axis=0) * 1000  # Convert to spikes per second
    smoothed_psth = uniform_filter1d(psth, size=100)
    return smoothed_psth


''' Part 1 '''

# Different colors for each condition !

# Reload the data from the file
file_path = 'Python_C5843.pkl'
data = pd.read_pickle(file_path)

# Group the data by the unique task conditions
grouped_data = data.groupby(['R_prob', 'U_prob'])

# Initialize a dictionary to store smoothed PSTHs for each condition
psth_conditions = {}

# Calculate the smoothed PSTH for each condition
for (r_prob, u_prob), trials in grouped_data:
    psth_conditions[(r_prob, u_prob)] = calculate_smoothed_psth(trials)

# Plot all smoothed PSTHs in a single plot
plt.figure(figsize=(15, 8))
colors = ['blue', 'green', 'red', 'orange', 'purple', 'cyan', 'magenta', 'brown', 'gray', 'olive']
up_conditions = [key for key in psth_conditions if key[0] < key[1]]  # UP high-probability conditions
right_conditions = [key for key in psth_conditions if key[0] > key[1]]  # RIGHT high-probability conditions

# Plot for UP high-probability conditions
for i, condition in enumerate(up_conditions):
    plt.plot(psth_conditions[condition], label=f'UP High: R={condition[0]}%, U={condition[1]}%',
             color=colors[i % len(colors)], linestyle='dashed')

# Plot for RIGHT high-probability conditions
for i, condition in enumerate(right_conditions):
    plt.plot(psth_conditions[condition], label=f'RIGHT High: R={condition[0]}%, U={condition[1]}%',
             color=colors[i % len(colors)])

plt.title('Smoothed PSTHs for All Task Conditions')
plt.xlabel('Time (ms)')
plt.ylabel('Firing Rate (spikes/s)')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# Different colors for times when R > U and times when U > R

# Plot all smoothed PSTHs in a single plot with two colors
plt.figure(figsize=(15, 8))

# Define two colors: one for UP > RIGHT and one for RIGHT > UP
color_up = 'green'
color_right = 'blue'

# Plot for UP high-probability conditions
for condition in up_conditions:
    plt.plot(psth_conditions[condition], label=f'UP High: R={condition[0]}%, U={condition[1]}%', color=color_up,
             linestyle='dashed')

# Plot for RIGHT high-probability conditions
for condition in right_conditions:
    plt.plot(psth_conditions[condition], label=f'RIGHT High: R={condition[0]}%, U={condition[1]}%', color=color_right,
             linestyle='solid')

plt.title('Smoothed PSTHs for All Task Conditions')
plt.xlabel('Time (ms)')
plt.ylabel('Firing Rate (spikes/s)')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

''' Part 2 '''

# Calculate smoothed PSTHs for RIGHT and UP conditions
smoothed_psth_right = calculate_smoothed_psth(right_high_trials)
smoothed_psth_up = calculate_smoothed_psth(up_high_trials)

# Plot the smoothed PSTHs for all conditions in a single plot
plt.figure(figsize=(12, 6))
plt.plot(smoothed_psth_right, label='High Probability = RIGHT', color='blue')
plt.plot(smoothed_psth_up, label='High Probability = UP', color='green')
plt.title('Smoothed PSTHs for All Task Conditions')
plt.xlabel('Time (ms)')
plt.ylabel('Firing Rate (spikes/s)')
plt.legend()
plt.show()

''' Question Number 5 - Quantitative analysis of the neuronal response'''

# Reload the data from the file
file_path = 'Python_C5843.pkl'
data = pd.read_pickle(file_path)

# Add 'high_prob_target' column to indicate which target had the higher probability
data['high_prob_target'] = (data['R_prob'] > data['U_prob']).astype(int)

# Group the data by the unique task conditions
grouped_data = data.groupby(['R_prob', 'U_prob'])

# Step 1: Calculate the average number of spikes per condition
average_spikes = {}
for condition, trials in grouped_data:
    spike_matrix = np.array(trials['spikes'].tolist())
    total_spikes = spike_matrix.sum(axis=1)  # Sum spikes for each trial
    average_spikes[condition] = total_spikes.mean()  # Average across trials

# Separate conditions into UP and RIGHT
up_conditions = [(r, u) for (r, u) in average_spikes if r < u]
right_conditions = [(r, u) for (r, u) in average_spikes if r > u]

# Extract data for plotting
up_x = [u for _, u in up_conditions]  # High probability values for UP
up_y = [average_spikes[(r, u)] for (r, u) in up_conditions]  # Average spikes for UP

right_x = [r for r, _ in right_conditions]  # High probability values for RIGHT
right_y = [average_spikes[(r, u)] for (r, u) in right_conditions]  # Average spikes for RIGHT

# Step 2: Plot the average spikes as a function of the high probability
plt.figure(figsize=(12, 6))
plt.scatter(up_x, up_y, color='green', label='High Probability = UP')
plt.scatter(right_x, right_y, color='blue', label='High Probability = RIGHT')
plt.xlabel('High Probability (%)')
plt.ylabel('Average Spikes per Trial')
plt.title('Average Spikes as a Function of High Probability')
plt.legend()
plt.show()

# Step 3: Perform linear regression and plot regression lines
# UP conditions
up_x_array = np.array(up_x).reshape(-1, 1)
up_y_array = np.array(up_y)
reg_up = LinearRegression().fit(up_x_array, up_y_array)
up_y_pred = reg_up.predict(up_x_array)

# RIGHT conditions
right_x_array = np.array(right_x).reshape(-1, 1)
right_y_array = np.array(right_y)
reg_right = LinearRegression().fit(right_x_array, right_y_array)
right_y_pred = reg_right.predict(right_x_array)

# Plot with regression lines
plt.figure(figsize=(12, 6))
plt.scatter(up_x, up_y, color='green', label='High Probability = UP')
plt.scatter(right_x, right_y, color='blue', label='High Probability = RIGHT')
plt.plot(up_x, up_y_pred, color='darkgreen', linestyle='dashed', label='Regression: UP')
plt.plot(right_x, right_y_pred, color='darkblue', linestyle='dashed', label='Regression: RIGHT')
plt.xlabel('High Probability (%)')
plt.ylabel('Average Spikes per Trial')
plt.title('Average Spikes and Regression Lines')
plt.legend()
plt.show()

# Step 4: Output regression coefficients
reg_up_coef = reg_up.coef_[0], reg_up.intercept_
reg_right_coef = reg_right.coef_[0], reg_right.intercept_

# Print the results in a formatted manner
print(f"Regression Coefficients:")
print(f"High Probability = UP: Slope = {reg_up_coef[0]:.3f}, Intercept = {reg_up_coef[1]:.2f}")
print(f"High Probability = RIGHT: Slope = {reg_right_coef[0]:.3f}, Intercept = {reg_right_coef[1]:.2f}")

''' Question Number 6 - Comparative analysis of the responses'''

# Load the data from the file Python_C4886.pkl for neuron 4886
file_path = 'Python_C4886.pkl'
data = pd.read_pickle(file_path)

# Inspect the first few rows of the dataset
data.head()

# Define conditions based on R_prob and U_prob
data['Condition'] = list(zip(data['R_prob'], data['U_prob']))

# Group by condition and calculate the average number of spikes per condition
condition_avg_spikes = data.groupby('Condition')['spikes'].apply(
    lambda x: np.mean([np.sum(trial) for trial in x])
)

# Extract averages and probabilities for plotting
conditions = condition_avg_spikes.index
avg_spikes = condition_avg_spikes.values
R_probs = [cond[0] for cond in conditions]
U_probs = [cond[1] for cond in conditions]

# Separate conditions where high probability is RIGHT or UP
high_right_conditions = [cond for cond in conditions if cond[0] > cond[1]]
high_up_conditions = [cond for cond in conditions if cond[1] > cond[0]]

# Extract averages for high probability conditions
right_avg_spikes = [condition_avg_spikes[cond] for cond in high_right_conditions]
up_avg_spikes = [condition_avg_spikes[cond] for cond in high_up_conditions]

right_probs = [cond[0] for cond in high_right_conditions]
up_probs = [cond[1] for cond in high_up_conditions]

# Perform linear regression for RIGHT and UP conditions separately
right_regression = linregress(right_probs, right_avg_spikes)
up_regression = linregress(up_probs, up_avg_spikes)

# Plot the data with regression lines
plt.figure(figsize=(10, 6))
plt.scatter(right_probs, right_avg_spikes, label='High Probability RIGHT', color='blue')
plt.scatter(up_probs, up_avg_spikes, label='High Probability UP', color='orange')

# Add regression lines
plt.plot(
    right_probs,
    right_regression.intercept + right_regression.slope * np.array(right_probs),
    color='blue', linestyle='--', label='Regression (RIGHT)'
)
plt.plot(
    up_probs,
    up_regression.intercept + up_regression.slope * np.array(up_probs),
    color='orange', linestyle='--', label='Regression (UP)'
)

# Labels and title
plt.xlabel('High Probability (%)')
plt.ylabel('Average Spikes (spikes/trial)')
plt.title('Average Spikes vs. High Probability with Regression Lines (Neuron 4886)')
plt.legend()
plt.grid(True)
plt.show()

# Load the data from the file Python_C5838.pkl for neuron 5838
file_path = 'Python_C5838.pkl'
data = pd.read_pickle(file_path)

# Inspect the first few rows of the dataset
data.head()

# Define conditions based on R_prob and U_prob
data['Condition'] = list(zip(data['R_prob'], data['U_prob']))

# Group by condition and calculate the average number of spikes per condition
condition_avg_spikes = data.groupby('Condition')['spikes'].apply(
    lambda x: np.mean([np.sum(trial) for trial in x])
)

# Extract averages and probabilities for plotting
conditions = condition_avg_spikes.index
avg_spikes = condition_avg_spikes.values
R_probs = [cond[0] for cond in conditions]
U_probs = [cond[1] for cond in conditions]

# Separate conditions where high probability is RIGHT or UP
high_right_conditions = [cond for cond in conditions if cond[0] > cond[1]]
high_up_conditions = [cond for cond in conditions if cond[1] > cond[0]]

# Extract averages for high probability conditions
right_avg_spikes = [condition_avg_spikes[cond] for cond in high_right_conditions]
up_avg_spikes = [condition_avg_spikes[cond] for cond in high_up_conditions]

right_probs = [cond[0] for cond in high_right_conditions]
up_probs = [cond[1] for cond in high_up_conditions]

# Perform linear regression for RIGHT and UP conditions separately
right_regression = linregress(right_probs, right_avg_spikes)
up_regression = linregress(up_probs, up_avg_spikes)

# Plot the data with regression lines
plt.figure(figsize=(10, 6))
plt.scatter(right_probs, right_avg_spikes, label='High Probability RIGHT', color='blue')
plt.scatter(up_probs, up_avg_spikes, label='High Probability UP', color='orange')

# Add regression lines
plt.plot(
    right_probs,
    right_regression.intercept + right_regression.slope * np.array(right_probs),
    color='blue', linestyle='--', label='Regression (RIGHT)'
)
plt.plot(
    up_probs,
    up_regression.intercept + up_regression.slope * np.array(up_probs),
    color='orange', linestyle='--', label='Regression (UP)'
)

# Labels and title
plt.xlabel('High Probability (%)')
plt.ylabel('Average Spikes (spikes/trial)')
plt.title('Average Spikes vs. High Probability with Regression Lines (Neuron 5838)')
plt.legend()
plt.grid(True)
plt.show()


''' Bonus '''

# Load the newly provided data file for Neuron 5843
file_path_5843 = '/mnt/data/Python_C5843.pkl'
data_5843 = pd.read_pickle(file_path_5843)

# Add the probability of the non-selected target for Neuron 5843
data_5843['NonSelectedProb'] = data_5843.apply(
    lambda row: row['U_prob'] if row['choice'] == 1 else row['R_prob'], axis=1
)

# Group trials by the non-selected target's probability and calculate average spikes
non_selected_avg_spikes_5843 = data_5843.groupby('NonSelectedProb')['spikes'].apply(
    lambda x: np.mean([np.sum(trial) for trial in x])
)

# Perform regression analysis
non_selected_probs_5843 = non_selected_avg_spikes_5843.index
avg_spikes_non_selected_5843 = non_selected_avg_spikes_5843.values
non_selected_regression_5843 = linregress(non_selected_probs_5843, avg_spikes_non_selected_5843)

# Plot the results for Neuron 5843
plt.figure(figsize=(10, 6))
plt.scatter(non_selected_probs_5843, avg_spikes_non_selected_5843, color='red', label='Non-Selected Target')
plt.plot(
    non_selected_probs_5843,
    non_selected_regression_5843.intercept + non_selected_regression_5843.slope * np.array(non_selected_probs_5843),
    color='red', linestyle='--', label='Regression Line'
)
plt.xlabel('Non-Selected Target Probability (%)')
plt.ylabel('Average Spikes (spikes/trial)')
plt.title('Neuron 5843: Modulation by Non-Selected Target Probability')
plt.legend()
plt.grid(True)
plt.show()

# Display regression results for Neuron 5843
print (non_selected_regression_5843)
