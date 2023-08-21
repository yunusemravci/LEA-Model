import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kstest
from lmfit import Model
from scipy.optimize import minimize
import pandas as pd

# Define the Weibull hazard function
def weibull_hazard(t, A, B):
    t = np.maximum(t, np.float128(1e-10))  # Avoid zero or negative values
    return (B / A) * (t / A) ** (B - 1)

# Smoothing function using a simple moving average
def smooth_data(data, window_size=5):
    window = np.ones(window_size) / window_size
    smoothed_data = np.convolve(data, window, mode='same')
    return smoothed_data

# Define the objective function
def objective(params, t, y_observed):
    A, B = params
    y_estimated = weibull_hazard(t, A, B)
    return np.sum((y_observed - y_estimated) ** 2)

df = pd.read_csv('Hazard_Model_Data_1.csv')

for (state, mobility_type), group in df.groupby(['state', 'mobility_type']):
    t = np.array(np.arange(len(group['t'])))
    y_observed = group['h(t)'].values
    
    np.random.seed(42)  # for reproducibility
    
    best_A, best_B = None, None
    best_obj = np.inf
    num_trials = 100

    # Apply smoothing to the observed data
    smoothed_y_observed = smooth_data(y_observed)
    
    # Define the lmfit Weibull model
    weibull_model = Model(weibull_hazard)
    params = weibull_model.make_params(A=1, B=1)

    for _ in range(num_trials):
        init_params = np.random.uniform(0, 40, size=2)  # Random values between 0 and 1 for A and B
        # Define the bounds and constraints for optimization
        bounds = [(1e-10, None), (1e-10, None)]  # A can be positive, B must be positive
        constraints = ({'type': 'ineq', 'fun': lambda x: x[1] - 1e-10})  # B >= 1e-10
    
        res = minimize(objective, init_params, args=(t, y_observed), bounds=bounds, constraints=constraints)
        
        if res.fun < best_obj:
            best_obj = res.fun
            best_A, best_B = res.x

    y_predicted = weibull_hazard(t, best_A, best_B)
    
    # Calculate R^2
    SSR = np.sum((y_observed - y_predicted) ** 2)
    SST = np.sum((y_observed - np.mean(y_observed)) ** 2)
    R2 = 1 - (SSR / SST)
    
    # Calculate the Kolmogorov-Smirnov test statistic and p-value
    ks_statistic, p_value = kstest(y_observed, 'weibull_min', args=(best_A, 0, best_B))
    
    # Plot the observed data and the fitted Weibull hazard function
    
    plt.figure(figsize=(10, 6))
    plt.title(f'Observed vs. Fitted Hazard Model for {state}, {mobility_type}')
    plt.plot(t, y_observed, 'o-', label='Observed')
    plt.plot(t, y_predicted, 'x-', label='Fitted Weibull Hazard')
    plt.xlabel('t')
    plt.ylabel('h(t)')
    #plt.title('Observed vs. Fitted Weibull Hazard Function')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    print(f"Estimated A (scale parameter): {best_A}")
    print(f"Estimated B (shape parameter): {best_B}")
    print(f"R^2 value: {R2}")
    print(f"Kolmogorov-Smirnov test statistic: {ks_statistic}")
    print(f"P-value: {p_value}")
