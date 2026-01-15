import pandas as pd
import numpy as np
import os

print("--- STARTING TRAINING ---")

# 1. Load Data
data_path = '/data/housing.csv'
print(f"Reading data from {data_path}...")

if not os.path.exists(data_path):
    print("ERROR: Data file not found!")
    exit(1)

df = pd.read_csv(data_path)
print(f"Loaded {len(df)} rows.")

# 2. Simulate Training (Math)
print("Training model...")
mean_price = np.mean(df['price'])

# 3. Output
print(f"Training complete. Average House Price: {mean_price}")
print("--- SUCCESS ---")
