import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tensorflow as tf

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from tensorflow.keras import regularizers

# =======================
# 1. Data Loading and Preprocessing
# =======================

# Load the dataset
dataset = pd.read_csv('heart.csv')
print("Initial dataset head:")
print(dataset.head())

# Define categorical features to encode
categorical_features = ['Sex', 'ChestPainType', 'FastingBS', 'RestingECG', 'ExerciseAngina', 'ST_Slope']

# Create a ColumnTransformer to apply OneHotEncoder to the categorical columns while leaving others unchanged
transformer = ColumnTransformer(
    transformers=[('cat_encoder', OneHotEncoder(), categorical_features)],
    remainder='passthrough'
)

# Fit and transform the dataset
transformed_data = transformer.fit_transform(dataset)

# Get the new feature names
new_feature_names = transformer.get_feature_names_out()

# Create a DataFrame from the transformed data
data = pd.DataFrame(transformed_data, columns=new_feature_names)
print("\nTransformed data head:")
print(data.head())

# =======================
# 2. Feature and Target Separation
# =======================

# Based on your columns, the target column is 'remainder__HeartDisease'
target_col = 'remainder__HeartDisease'

# Optional: Check available columns if needed
print("\nColumns in dataset:", data.columns.tolist())

if target_col not in data.columns:
    raise KeyError(f"Column '{target_col}' not found. Available columns: {data.columns.tolist()}")

# Separate features and target
X_data = data.drop(target_col, axis=1)
y_data = data[target_col]

# Convert to numpy arrays
X = X_data.values
y = y_data.values.reshape(-1, 1)

# Determine the number of features
n = X.shape[1]

# =======================
# 3. Splitting the Data
# =======================

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
y_train = y_train.ravel()  # Flatten the target for training

# =======================
# 4. Neural Network Models
# =======================

# ----- Model 1: Basic Neural Network -----
model_1 = tf.keras.Sequential([
    tf.keras.layers.Dense(1, activation='sigmoid', input_shape=(n,))
])
model_1.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history1 = model_1.fit(X_train, y_train, epochs=200, batch_size=10, verbose=0)
test_loss1, test_accuracy1 = model_1.evaluate(X_test, y_test, verbose=0)
print("\nModel 1 - Basic Neural Network")
print("Test Loss:", test_loss1)
print("Test Accuracy:", test_accuracy1)

# ----- Model 2: Neural Network with L2 Regularization -----
model_2 = tf.keras.Sequential([
    tf.keras.Input(shape=(n,)),
    tf.keras.layers.Dense(5, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    tf.keras.layers.Dense(1, activation='sigmoid', kernel_regularizer=regularizers.l2(0.01))
])
model_2.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history2 = model_2.fit(X_train, y_train, epochs=100, batch_size=10, verbose=0)
test_loss2, test_accuracy2 = model_2.evaluate(X_test, y_test, verbose=0)
print("\nModel 2 - Neural Network with L2 Regularization")
print("Test Loss:", test_loss2)
print("Test Accuracy:", test_accuracy2)

# ----- Model 3: Larger Neural Network with L2 Regularization -----
large_model = tf.keras.Sequential([
    tf.keras.Input(shape=(n,)),
    tf.keras.layers.Dense(15, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    tf.keras.layers.Dense(5, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    tf.keras.layers.Dense(1, activation='sigmoid', kernel_regularizer=regularizers.l2(0.01))
])
large_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history3 = large_model.fit(X_train, y_train, epochs=100, batch_size=10, verbose=0)
test_loss3, test_accuracy3 = large_model.evaluate(X_test, y_test, verbose=0)
print("\nModel 3 - Larger Neural Network with L2 Regularization")
print("Test Loss:", test_loss3)
print("Test Accuracy:", test_accuracy3)
