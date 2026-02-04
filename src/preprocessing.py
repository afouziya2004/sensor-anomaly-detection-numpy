import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def load_data(path):
    df = pd.read_csv(path)
    print("Data Loaded Successfully")
    return df


def inspect_data(df):
    print("\nFirst 5 rows:\n", df.head())
    print("\nData Info:")
    print(df.info())
    print("\nMissing Values:\n", df.isnull().sum())


def preprocess(df):
    # Convert timestamp
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Drop missing values
    df = df.dropna()

    return df


def convert_to_numpy(df):
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    data_array = df[numeric_cols].to_numpy()
    print("\nConverted to NumPy array")
    return data_array


def visualize_signal(df):
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    plt.figure(figsize=(10,5))
    plt.plot(df[numeric_cols[0]])
    plt.title("Raw Sensor Signal")
    plt.xlabel("Index")
    plt.ylabel("Value")
    plt.grid(True)

    plt.savefig("visuals/raw_signal.png")
    plt.show()


def main():
    df = load_data("data/sensor_timeseries.csv")

    inspect_data(df)

    df = preprocess(df)

    data_array = convert_to_numpy(df)

    visualize_signal(df)

    print("\nShape of NumPy Data:", data_array.shape)


if __name__ == "__main__":
    main()
