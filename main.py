import numpy as np
from src.numpy_statistics import moving_smoothing
from src.anomaly_engine import detect_zscore_anomalies
from src.visualization import plot_anomalies


def run_pipeline():

    np.random.seed(42)
    data = np.random.normal(50, 10, 200)

    data[80] = 120
    data[150] = 5

    smoothed = moving_smoothing(data, 5)
    anomalies = detect_zscore_anomalies(smoothed)

    plot_anomalies(smoothed, anomalies)


if __name__ == "__main__":
    run_pipeline()
