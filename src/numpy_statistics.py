import numpy as np


# -------------------------------
# Rolling Mean (Vectorized)
# -------------------------------
def rolling_mean(data, window):
    kernel = np.ones(window) / window
    return np.convolve(data, kernel, mode='valid')


# -------------------------------
# Rolling Standard Deviation
# -------------------------------
def rolling_std(data, window):
    means = rolling_mean(data, window)
    squared = rolling_mean(data**2, window)
    variance = squared - means**2
    return np.sqrt(variance)


# -------------------------------
# Z-Score Calculation
# -------------------------------
def z_score(data):
    mean = np.mean(data)
    std = np.std(data)
    return (data - mean) / std


# -------------------------------
# Moving Window Smoothing
# -------------------------------
def moving_smoothing(data, window=5):
    kernel = np.ones(window) / window
    return np.convolve(data, kernel, mode='same')


# -------------------------------
# Example Test Block
# -------------------------------
if __name__ == "__main__":

    np.random.seed(42)
    sensor_data = np.random.normal(50, 10, 100)

    print("Rolling Mean:", rolling_mean(sensor_data, 5))
    print("Rolling Std:", rolling_std(sensor_data, 5))
    print("Z Score:", z_score(sensor_data))
    print("Smoothed Data:", moving_smoothing(sensor_data, 5))
