import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
import os

# Load datasets
print("🔄 Loading data...")
train_data = pd.read_csv("data/TEP_FaultFree_Training.csv")
test_data = pd.read_csv("data/TEP_FaultFree_Testing.csv")

# Drop the first column if it's an index
train_data = train_data.iloc[:, 1:]
test_data = test_data.iloc[:, 1:]

# Normalize data
scaler = MinMaxScaler()
train_scaled = scaler.fit_transform(train_data)
test_scaled = scaler.transform(test_data)

# Build Autoencoder
input_dim = train_scaled.shape[1]

model = keras.Sequential([
    keras.layers.Input(shape=(input_dim,)),
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dense(32, activation="relu"),
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dense(input_dim, activation="sigmoid")  # Output layer same shape as input
])

model.compile(optimizer="adam", loss="mse")
model.summary()

# Train model
print("🚀 Training autoencoder...")
history = model.fit(
    train_scaled, train_scaled,
    epochs=50,
    batch_size=32,
    validation_data=(test_scaled, test_scaled),
    verbose=1
)

# Save model and scaler
os.makedirs("models", exist_ok=True)
model.save("models/autoencoder.h5")
np.save("models/scaler.npy", scaler)

print("✅ Model trained and saved!")
