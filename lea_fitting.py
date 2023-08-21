import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error

# Step 2: Define the time-normalized function
def h_norm(t, a, p, s):
    return a * np.exp(-(t / p) ** s)

# Step 1: Read the data
df = pd.read_csv('ht_data/ht_by_mobility_type.csv')

# List of mobility type columns
mobility_columns = ['parks_percent_change_from_baseline', 'residential_percent_change_from_baseline', 
                    'grocery_and_pharmacy_percent_change_from_baseline', 'retail_and_recreation_percent_change_from_baseline', 
                    'transit_stations_percent_change_from_baseline', 'workplaces_percent_change_from_baseline']

# Remove rows with NaN or infinite values in h_original
df = df.interpolate(method='linear')
    
# Loop through each mobility column
for mobility_column in mobility_columns:
    # Use your guidance to get an initial estimate for p
    p_init = df['Unnamed: 0'].iloc[df[mobility_column].idxmax()]

    # Calibrate the parameters
    bounds = ([0, 0,0], [np.inf, np.inf, np.inf])
    
    # Remove the first row where 'Unnamed: 0' value is 0
    df_filtered = df[df['Unnamed: 0'] != 0]
    # Filter out rows where mobility column contains NaN or infinite values
    df_filtered = df_filtered[np.isfinite(df_filtered[mobility_column])]

    params, _ = curve_fit(h_norm, df_filtered['Unnamed: 0'].values, df_filtered[mobility_column].values / df_filtered['Unnamed: 0'].values, 
                          p0=[1, p_init, 30],maxfev=2000)
    
    # Extract parameters
    a_est, p_est, s_est = params

    # Compute h_est(t)
    df_filtered['h_est'] = h_norm(df_filtered['Unnamed: 0'], a_est, p_est, s_est) * df_filtered['Unnamed: 0']

    # Compute R^2
    residuals = df_filtered[mobility_column] - df_filtered['h_est']
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((df_filtered[mobility_column] - np.mean(df_filtered[mobility_column]))**2)
    r2 = 1 - (ss_res / ss_tot)

    # Compute RMSE
    rmse = np.sqrt(mean_squared_error(df_filtered[mobility_column], df_filtered['h_est']))

    print(f"Mobility Column: {mobility_column}")
    print(f"R^2: {r2}")
    print(f"RMSE: {rmse:.4f}")

    shifted_t = df_filtered['Unnamed: 0'] - df_filtered['Unnamed: 0'].min()

    # Plot the data
    plt.scatter(shifted_t, df_filtered[mobility_column], label=f"{mobility_column} percent change", color='blue')
    plt.plot(shifted_t, df_filtered['h_est'], label='Estimated data', color='red')
    plt.xlabel('Time')
    plt.ylabel('h(t)')
    plt.title(f'Mobility Type: {mobility_column}\nRMSE: {rmse:.4f}')
    plt.legend()
    plt.show()
