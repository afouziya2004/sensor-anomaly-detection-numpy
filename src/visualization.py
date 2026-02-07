import numpy as np
import matplotlib.pyplot as plt


# -----------------------------------
# Plot Raw vs Smoothed Signal
# -----------------------------------
def plot_raw_vs_smoothed(raw, smoothed):
    plt.figure(figsize=(12, 5))
    plt.plot(raw, label="Raw Signal")
    plt.plot(smoothed, label="Smoothed Signal")
    plt.title("Raw vs Smoothed Sensor Signal")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.show()


# -----------------------------------
# Plot Anomalies Highlighted
# -----------------------------------
def plot_anomalies(data, anomalies):
    plt.figure(figsize=(12, 5))
    plt.plot(data, label="Sensor Data")

    anomaly_idx = np.where(anomalies == 1)[0]
    plt.scatter(anomaly_idx, data[anomaly_idx],
                label="Anomalies", marker='o')

    plt.title("Anomaly Detection Visualization")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.show()


# -----------------------------------
# Rolling Threshold Visualization
# -----------------------------------
def plot_dynamic_threshold(data, rolling_mean, upper, lower):
    plt.figure(figsize=(12, 5))
    plt.plot(data, label="Sensor Data")
    plt.plot(rolling_mean, label="Rolling Mean")

    x_axis = np.arange(len(upper))
    plt.plot(x_axis, upper, linestyle='--', label="Upper Threshold")
    plt.plot(x_axis, lower, linestyle='--', label="Lower Threshold")

    plt.title("Dynamic Threshold Monitoring")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.show()


# -----------------------------------
# Anomaly Timeline Plot
# -----------------------------------
def plot_anomaly_timeline(anomalies):
    plt.figure(figsize=(12, 2))
    plt.plot(anomalies)
    plt.title("Anomaly Timeline")
    plt.xlabel("Time")
    plt.ylabel("Anomaly Flag")
    plt.grid(True)
    plt.show()


# -----------------------------------
# Distribution Comparison
# -----------------------------------
def plot_distribution(data, smoothed):
    plt.figure(figsize=(10, 5))
    plt.hist(data, bins=30, alpha=0.5, label="Raw")
    plt.hist(smoothed, bins=30, alpha=0.5, label="Smoothed")

    plt.title("Distribution Comparison")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.legend()
    plt.show()
