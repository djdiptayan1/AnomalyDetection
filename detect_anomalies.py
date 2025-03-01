import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

print("GPUs Available:", tf.config.list_physical_devices("GPU"))
BATCH_SIZE = 10000  # Process 10,000 rows at a time to avoid memory issues

# Load the faulty dataset
print("🔄 Loading scaler...")
scaler = np.load("models/scaler.npy", allow_pickle=True).item()

print("📂 Loading trained model...")
model = keras.models.load_model(
    "models/autoencoder.h5", custom_objects={"mse": keras.losses.MeanSquaredError()}
)

# Read CSV in chunks to handle large files
print("🔄 Loading data in chunks...")
chunk_iterator = pd.read_csv("data/faulty_testing.csv", chunksize=BATCH_SIZE)

anomaly_results = []

for i, chunk in enumerate(chunk_iterator):
    print(f"🔍 Processing chunk {i+1}...")

    chunk = chunk.iloc[:, 1:]  # Drop index column if present
    scaled_chunk = scaler.transform(chunk)

    # Predict reconstruction error
    reconstructed = model.predict(scaled_chunk, verbose=0)
    mse = np.mean(np.power(scaled_chunk - reconstructed, 2), axis=1)

    # Set threshold dynamically (mean + 3*std of first batch)
    if i == 0:
        threshold = np.mean(mse) + 3 * np.std(mse)

    anomalies = mse > threshold
    chunk["Anomaly"] = anomalies
    anomaly_results.append(chunk)


# ✅ Save results
print("📂 Saving detected anomalies...")
final_df = pd.concat(anomaly_results, ignore_index=True)
final_df.to_csv("data/TEP_Faulty_Training_Anomalies_dip.csv", index=False)


# ✅ Plot anomaly distribution
plt.figure(figsize=(10, 5))
plt.hist(mse, bins=50, alpha=0.75, color="blue", label="Reconstruction Error")
plt.axvline(threshold, color="red", linestyle="dashed", linewidth=2, label="Threshold")
plt.xlabel("MSE (Reconstruction Error)")
plt.ylabel("Frequency")
plt.title("Anomaly Detection using Autoencoder")
plt.legend()
plt.show()

print("✅ Anomaly detection complete! Results saved.")
