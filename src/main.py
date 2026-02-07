import numpy as np

from numpy_statistics import moving_smoothing
from anomaly_engine import detect_zscore_anomalies
from visualization import plot_raw_vs_smoothed, plot_anomalies


# ------------------------------
# Generate Sample Sensor Data
# ------------------------------
np.random.seed(42)
sensor_data = np.random.normal(50, 10, 200)

# Inject fake anomalies
sensor_data[40] = 120
sensor_data[150] = 5


# ------------------------------
# Processing
# ------------------------------
smoothed = moving_smoothing(sensor_data, 5)
anomalies = detect_zscore_anomalies(sensor_data)


# ------------------------------
# Visualization
# ------------------------------
plot_raw_vs_smoothed(sensor_data, smoothed)
plot_anomalies(sensor_data, anomalies)