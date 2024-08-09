import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from scipy.stats import gaussian_kde
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
import os

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

# Load realized power output data
df_power = pd.read_csv('./data/Realised_Supply_Germany_Hourly.csv')

# Load weather data for 2022
df_weather_2022 = pd.read_csv('./data/Weather_Data_Germany_2022.csv')

# Preprocess weather data
df_weather = df_weather[['time', 'ssr', 'tcc', 'sund', 'tsr', 'cdir']].dropna()

# Preprocess power data
df_power = df_power[['time', 'Photovoltaic [MW]']].dropna()
df_power.rename(columns={'Photovoltaic [MW]': 'power_output'}, inplace=True)

# Merge datasets on time
df = pd.merge(df_weather, df_power, on='time')
df_weather_2022 = pd.merge(df_weather, df_power, on='time')

# Convert time to datetime
df['time'] = pd.to_datetime(df['time'])
df['date'] = df['time'].dt.date
df['hour'] = df['time'].dt.hour + df['time'].dt.minute / 60.0

df_weather_2022['time'] = pd.to_datetime(df_weather_2022['time'])
df_weather_2022['date'] = df_weather_2022['time'].dt.date
df_weather_2022['hour'] = df_weather_2022['time'].dt.hour + df_weather_2022['time'].dt.minute / 60.0

# Calculate sunrise and sunset times (dummy values, may not lead to best results, but better than without)
df['sunrise'] = df['date'].apply(lambda d: dt.datetime.combine(d, dt.time(6, 0)))  # 6 AM
df['sunset'] = df['date'].apply(lambda d: dt.datetime.combine(d, dt.time(20, 0)))  # 8 PM

df_weather_2022['sunrise'] = df_weather_2022['date'].apply(lambda d: dt.datetime.combine(d, dt.time(6, 0)))  # 6 AM
df_weather_2022['sunset'] = df_weather_2022['date'].apply(lambda d: dt.datetime.combine(d, dt.time(20, 0)))  # 8 PM

# Add daytime flag
df['is_daytime'] = df['time'].between(df['sunrise'], df['sunset'])
df_weather_2022['is_daytime'] = df_weather_2022['time'].between(df_weather_2022['sunrise'], df_weather_2022['sunset'])

# Aggregate data to daily values
df = df.groupby('date').agg({
    'tcc': 'mean',
    'sund': 'mean',
    'ssr': 'mean',
    'tsr': 'mean',
    'cdir': 'mean',
    'power_output': 'sum'
}).reset_index()

df = df.groupby('date').agg({
    'tcc': 'mean',
    'sund': 'mean',
    'ssr': 'mean',
    'tsr': 'mean',
    'cdir': 'mean',
    'power_output': 'sum'
}).reset_index()

# Define features and target
X = df[['tcc', 'ssr', 'sund', 'tsr', 'cdir']]

y = df['power_output']


# Split the data into train and test sets based on 2022
X_train, X_test, y_train, y_test = X, df_weather_2022[['tcc', 'ssr', 'sund', 'tsr', 'cdir']], y, df_weather_2022['power_output']

# Split the data into train and test sets randomly
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

X_train = X_train
y_train = y_train

# Train the k-NN model with the custom distance metric
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
    # Ensure X_single retains DataFrame structure
    X_copy = X_single.values.reshape(1, -1)  # Convert to NumPy array and reshape
    index = X_single.name  # Get the index of the single sample
    
    distances, indices = model.kneighbors(X_copy)
    distances = distances.flatten()  # Flatten to 1D array
    indices = indices.flatten()  # Flatten to 1D array

    power_outputs = y_train.iloc[indices].values  # Get corresponding power outputs
    
    if len(power_outputs) == 0:
        return 0, np.zeros(100)  # If all nearest neighbors have zero power output, return 0
    
    # Calculate weights based on distances (inverse of distances)
    weights = np.array([1 / d if d != 0 else 0 for d in distances])
    
    # Normalize weights to sum to 1
    weights /= weights.sum()
    
    if len(power_outputs) == 0:
        return 0, np.zeros(100)  # If all filtered neighbors have zero power output, return 0
    
    kde = gaussian_kde(power_outputs, weights=weights)
    x_vals = np.linspace(min(power_outputs), max(power_outputs), 100)
    kde_vals = kde(x_vals)
    most_likely_value = x_vals[np.argmax(kde_vals)]
    
    return most_likely_value, kde_vals

# Disable scientific notation for printing
np.set_printoptions(suppress=True)

test_index = 0

crps_knn_scores = []
crps_kde_scores = []
bandwidth = 1.0

for i in range(test_index, len(X_test)):
    single_test_point = X_test.iloc[test_index]
    power_actual = y_test.iloc[test_index]
    predicted_power, kde_vals = predict_with_kde(knn, single_test_point, df)
    
    test = []
    distances, indices = knn.kneighbors(single_test_point.values.reshape(1, -1))
    non_zero_neighbors = y_train.iloc[indices.flatten()].values
    
    # Debugging information
    print(f"Actual: {power_actual}, Predicted: {predicted_power}")

    # Calculate CRPS KNN
    if len(non_zero_neighbors) > 0:
        crps_score = calculate_crps(power_actual, non_zero_neighbors, bandwidth, np.ones(len(non_zero_neighbors)) / len(non_zero_neighbors))
    else:
        crps_score = np.nan
    crps_knn_scores.append(crps_score)
    
    # Calculate CRPS KDE
    kde_vals = np.array(kde_vals)
    if len(kde_vals) > 0:
        crps_score = calculate_crps(power_actual, kde_vals, bandwidth, np.ones(len(kde_vals)) / len(kde_vals))
    else:
        crps_score = np.nan
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