import numpy as np
from numpy_statistics import rolling_mean, rolling_std, z_score


# -----------------------------------
# Z-Score Based Anomaly Detection
# -----------------------------------
def detect_zscore_anomalies(data, threshold=3.0):
    z = z_score(data)
    anomalies = np.abs(z) > threshold
    return anomalies.astype(int)


# -----------------------------------
# Dynamic Threshold Detection
# -----------------------------------
def detect_dynamic_anomalies(data, window=10, factor=2.5):
    r_mean = rolling_mean(data, window)
    r_std = rolling_std(data, window)

    # Align sizes
    trimmed_data = data[window-1:]

    upper = r_mean + factor * r_std
    lower = r_mean - factor * r_std

    anomalies = (trimmed_data > upper) | (trimmed_data < lower)
    return anomalies.astype(int)


# -----------------------------------
# Spike Detection Using Gradient
# -----------------------------------
def detect_spikes(data, gradient_threshold=20):
    gradients = np.abs(np.diff(data))
    spikes = gradients > gradient_threshold

    # pad to match original length
    spikes = np.insert(spikes, 0, 0)

    return spikes.astype(int)


# -----------------------------------
# Combine Multiple Detection Methods
# -----------------------------------
def combine_anomaly_signals(*signals):
    stacked = np.vstack(signals)
    combined = np.any(stacked, axis=0)
    return combined.astype(int)


# -----------------------------------
# Evaluation Metrics
# -----------------------------------
def evaluate_detection(predicted, actual):
    tp = np.sum((predicted == 1) & (actual == 1))
    fp = np.sum((predicted == 1) & (actual == 0))
    fn = np.sum((predicted == 0) & (actual == 1))

    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)

    return {
        "precision": precision,
        "recall": recall
    }


# -----------------------------------
# Example Test Run
# -----------------------------------
if __name__ == "__main__":

    np.random.seed(42)
    data = np.random.normal(50, 10, 200)

    # inject fake anomalies
    data[50] = 120
    data[120] = 5

    z_anom = detect_zscore_anomalies(data)
    dyn_anom = detect_dynamic_anomalies(data)
    spikes = detect_spikes(data)

    combined = combine_anomaly_signals(
        z_anom[len(z_anom)-len(spikes):],
        spikes
    )

    print("Combined anomalies:", combined)
