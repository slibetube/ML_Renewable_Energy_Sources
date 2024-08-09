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


# Define features and target
X = df[['tcc', 'ssr', 'hour', 'sund', 'tsr', 'cdir']]
y = df['power_output']


# Split the data into train and test sets based on 2022
X_train, X_test, y_train, y_test = X, df_weather_2022[['tcc', 'ssr', 'hour', 'sund', 'tsr', 'cdir']], y, df_weather_2022['power_output']

# Split the data into train and test sets randomly
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)


# Train the k-NN model using only daytime training data
daytime_train_indices = df.loc[X_train.index, 'is_daytime']
X_train_daytime = X_train[daytime_train_indices]
y_train_daytime = y_train[daytime_train_indices]

# Train the k-NN model with the custom distance metric
knn = KNeighborsRegressor(n_neighbors=100)
knn.fit(X_train_daytime, y_train_daytime)

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
    index = X_single.name  # Get the index of the single sample
    time = original_df.loc[index, 'time']
    date = time.date()
    sunrise = dt.datetime.combine(date, dt.time(6, 0))
    sunset = dt.datetime.combine(date, dt.time(20, 0))
    is_daytime = sunrise <= time <= sunset
    
    if not is_daytime:
        return 0, [0,0,0,0,0,0,0,0,0,0]
    
    distances, indices = model.kneighbors(X_copy)
    daytime_indices = [idx for idx in indices[0] if original_df.loc[X_train.index[idx], 'is_daytime']]
    power_outputs = y_train.iloc[daytime_indices]
    non_zero_indices = [idx for idx, power in zip(daytime_indices, power_outputs) if power != 0]
    non_zero_power_outputs = y_train.iloc[non_zero_indices]
    non_zero_distances = [distances[0][i] for i, idx in enumerate(daytime_indices) if y_train.iloc[idx] != 0]
    
    if len(non_zero_power_outputs) == 0:
        return 0, [0,0,0,0,0,0,0,0,0,0]  # If all nearest neighbors have zero power output, return 0
    
    # Calculate weights based on distances (inverse of distances)
    weights = np.array([1 / d if d != 0 else 0 for d in non_zero_distances])
    
    # Filter out NaN or infinite values
    valid_mask = np.isfinite(weights)
    non_zero_power_outputs = non_zero_power_outputs[valid_mask]
    weights = weights[valid_mask]
    
    if len(non_zero_power_outputs) == 0:
        return 0, [0,0,0,0,0,0,0,0,0,0]  # If all filtered neighbors have zero power output, return 0
    
    kde = gaussian_kde(non_zero_power_outputs, weights=weights)
    x_vals = np.linspace(min(non_zero_power_outputs), max(non_zero_power_outputs), 100)
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
    daytime_indices = [idx for idx in indices[0] if df.loc[X_train.index[idx], 'is_daytime']]
    #non_zero_neighbors = [y_train.iloc[idx] for idx in daytime_indices if y_train.iloc[idx] != 0]
    non_zero_neighbors = y_train.iloc[daytime_indices]
    # Debugging information
    print(f"Actual: {power_actual}, Predicted: {predicted_power}")

    # Calculate CRPS KNN
    non_zero_neighbors = np.array(non_zero_neighbors)
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
    # #print(f"Test point: {X_test.iloc[test_index]}, Test Index: {test_index}")
    # print(f"Predicted Power: {predicted_power}")
    # print(f"Actual Power: {power_actual}")
    # print(f"CRPS: {crps_score}")

    # # Plot the histogram with KDE
    # plt.figure(figsize=(10, 6))
    # sns.histplot(non_zero_neighbors, bins=50, kde=True, color='blue', edgecolor='black', alpha=0.7)
    # sns.kdeplot(non_zero_neighbors, color='green', linewidth=2)
    # plt.axvline(x=predicted_power, color='orange', linestyle='dashed', linewidth=2, label='Prediction')
    # plt.axvline(x=power_actual, color='purple', linestyle='dashed', linewidth=2, label='Actual Power')
    # plt.xlabel('Power Output')
    # plt.ylabel('Frequency')
    # plt.title('Histogram of Power Outputs of k-Nearest Neighbors with KDE visualization')
    # plt.legend()
    # plt.show()

    test_index += 1

# Calculate average CRPS based on knn
average_crps_knn = np.nanmean(crps_knn_scores)

# Calculate average CRPS based on kde
average_crps_kde = np.nanmean(crps_kde_scores)
print(f"Average CRPS based on knn: {average_crps_knn}")
print(f"Average CRPS based on kde: {average_crps_kde}")
