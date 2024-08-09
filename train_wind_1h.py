def warn(*args, **kwargs):
    pass

import warnings
warnings.warn = warn

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from scipy.stats import gaussian_kde
import numpy as np
import datetime as dt
import os
from tqdm import tqdm

# Define the directory containing the weather data
weather_data_dir = './data/regions/'

# Initialize an empty list to hold DataFrames
weather_dfs = []

# Loop through all files in the directory
for file_name in os.listdir(weather_data_dir):
    if file_name.endswith('.csv'):
        file_path = os.path.join(weather_data_dir, file_name)
        df_weather = pd.read_csv(file_path)
        weather_dfs.append(df_weather)

# Concatenate all weather DataFrames
df_weather = pd.concat(weather_dfs, ignore_index=True)

# Load weather data for 2022
df_weather_2022 = pd.read_csv('./data/Weather_Data_Germany_2022.csv')

# Load realized power output data
df_power = pd.read_csv('./data/Realised_Supply_Germany_Hourly.csv')

# Calculate wind speeds
df_weather['wind_speed_10m'] = np.sqrt(df_weather['u10']**2 + df_weather['v10']**2)
df_weather['wind_speed_100m'] = np.sqrt(df_weather['u100']**2 + df_weather['v100']**2)

df_weather = df_weather[['time', 'msl', 'u10', 'v10', 'u100', 'v100', 'wind_speed_10m', 'wind_speed_100m']].dropna()

df_weather_2022['wind_speed_10m'] = np.sqrt(df_weather_2022['u10']**2 + df_weather_2022['v10']**2)
df_weather_2022['wind_speed_100m'] = np.sqrt(df_weather_2022['u100']**2 + df_weather_2022['v100']**2)

df_weather_2022 = df_weather_2022[['time', 'msl', 'u10', 'v10', 'u100', 'v100', 'wind_speed_10m', 'wind_speed_100m']].dropna()

# Preprocess power data
df_power = df_power[['time', 'Photovoltaic [MW]']].dropna()
df_power.rename(columns={'Photovoltaic [MW]': 'power_output'}, inplace=True)

# Merge datasets on time
df = pd.merge(df_weather, df_power, on='time')
df_weather_2022 = pd.merge(df_weather_2022, df_power, on='time')

# Convert time to datetime
df['time'] = pd.to_datetime(df['time'])
df['date'] = df['time'].dt.date

df_weather_2022['time'] = pd.to_datetime(df_weather_2022['time'])
df_weather_2022['date'] = df_weather_2022['time'].dt.date

# Define features and target
X = df[['msl', 'u10', 'v10', 'u100', 'v100', 'wind_speed_10m', 'wind_speed_100m']]
y = df['power_output']

# Split the data into train and test sets
X_train, X_test, y_train, y_test = X, df_weather_2022[['msl', 'u10', 'v10', 'u100', 'v100', 'wind_speed_10m', 'wind_speed_100m']], y,df_weather_2022['power_output']

# Train the k-NN model
knn = KNeighborsRegressor(n_neighbors=100)
knn.fit(X_train, y_train)

# Function to calculate the KDE
def kde(x, data, bandwidth, weights):
    kernel = np.exp(-0.5 * ((x - data[:, None]) / bandwidth) ** 2) / np.sqrt(2 * np.pi)
    weighted_kernel = kernel * weights[:, None]
    return np.sum(weighted_kernel, axis=0) / (len(data) * bandwidth)

# Function to calculate CRPS
def calculate_crps(observed, data, bandwidth, weights):
    sorted_data = np.sort(data)
    cdf_values = np.cumsum(sorted_data) / np.sum(sorted_data)
    
    obs_cdf = np.interp(observed, sorted_data, cdf_values)
    return np.sum((cdf_values - (sorted_data >= observed)) ** 2) / len(sorted_data)

# Function to predict power output using KDE with weighted neighbors
def predict_with_kde(model, X_single, original_df):
    X_copy = X_single.values.reshape(1, -1)  # Convert to NumPy array and reshape
    distances, indices = model.kneighbors(X_copy)
    power_outputs = y_train.iloc[indices[0]]
    non_zero_indices = [idx for idx, power in zip(indices[0], power_outputs) if power != 0]
    non_zero_power_outputs = y_train.iloc[non_zero_indices]
    non_zero_distances = [distances[0][i] for i, idx in enumerate(indices[0]) if y_train.iloc[idx] != 0]
    
    if len(non_zero_power_outputs) == 0:
        return 0, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # If all nearest neighbors have zero power output, return 0
    
    # Calculate weights based on distances (inverse of distances)
    weights = np.array([1 / d if d != 0 else 0 for d in non_zero_distances])
    
    # Filter out NaN or infinite values
    valid_mask = np.isfinite(weights)
    non_zero_power_outputs = non_zero_power_outputs[valid_mask]
    weights = weights[valid_mask]
    
    if len(non_zero_power_outputs) == 0:
        return 0, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # If all filtered neighbors have zero power output, return 0
    
    kde = gaussian_kde(non_zero_power_outputs, weights=weights)
    x_vals = np.linspace(min(non_zero_power_outputs), max(non_zero_power_outputs), 100)
    kde_vals = kde(x_vals)
    most_likely_value = x_vals[np.argmax(kde_vals)]
    
    return most_likely_value, kde_vals

# Disable scientific notation for printing
np.set_printoptions(suppress=True)


crps_knn_scores = []
crps_kde_scores = []
bandwidth = 1.0

test_index = 0

for i in tqdm(range(test_index, len(X_test))):
    single_test_point = X_test.iloc[test_index]
    power_actual = y_test.iloc[test_index]
    predicted_power, kde_vals = predict_with_kde(knn, single_test_point, df)
    
    # Calculate CRPS KNN
    non_zero_neighbors = y_train.iloc[knn.kneighbors(single_test_point.values.reshape(1, -1))[1][0]]
    crps_score = calculate_crps(power_actual, non_zero_neighbors, bandwidth, np.ones(len(non_zero_neighbors)) / len(non_zero_neighbors))
    crps_knn_scores.append(crps_score)
    
    # Calculate CRPS KDE
    kde_vals = np.array(kde_vals)
    crps_score = calculate_crps(power_actual, kde_vals, bandwidth, np.ones(len(kde_vals)) / len(kde_vals))
    crps_kde_scores.append(crps_score)
    
    # Print the results with the original timestamp
    # print(f"Timestamp: {df.loc[X_test.index[test_index], 'time']}")
    # print(f"Predicted Power: {predicted_power}")
    # print(f"Actual Power: {power_actual}")
    # print(f"CRPS: {crps_score}")

    test_index += 1

# Calculate average CRPS based on knn
average_crps_knn = np.nanmean(crps_knn_scores)

# Calculate average CRPS based on kde
average_crps_kde = np.nanmean(crps_kde_scores)
print(f"Average CRPS based on knn: {average_crps_knn}")
print(f"Average CRPS based on kde: {average_crps_kde}")
