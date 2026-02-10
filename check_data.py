import pandas as pd
import numpy as np

# Load the data
data = pd.read_csv('data/MicrosoftStock.csv')

print("=" * 60)
print("DATASET ANALYSIS")
print("=" * 60)

# Basic info
print("\n1. BASIC INFO:")
print(f"   Total rows: {len(data)}")
print(f"   Columns: {list(data.columns)}")
print(f"   Date range: {data['date'].min()} to {data['date'].max()}")

# Check close prices
print("\n2. CLOSE PRICE STATISTICS:")
print(data['close'].describe())
print(f"\n   Min price: ${data['close'].min():.2f}")
print(f"   Max price: ${data['close'].max():.2f}")
print(f"   Mean price: ${data['close'].mean():.2f}")
print(f"   Price range: ${data['close'].max() - data['close'].min():.2f}")

# First and last few rows
print("\n3. FIRST 5 ROWS:")
print(data.head())

print("\n4. LAST 5 ROWS:")
print(data.tail())

# Check for missing values
print("\n5. MISSING VALUES:")
print(data.isnull().sum())

# Data types
print("\n6. DATA TYPES:")
print(data.dtypes)