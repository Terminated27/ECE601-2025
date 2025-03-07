# output of file: 
'''
achin@pop-os:~/Documents/ECE601-2025/HW#4 Due Date March 06/HW_4 4$ /usr/bin/python3 "/home/achin/Documents/ECE601-2025/HW#4 Due Date March 06/HW_4 4/question2.py"
2025-03-06 21:20:44.540894: I external/local_xla/xla/tsl/cuda/cudart_stub.cc:32] Could not find cuda drivers on your machine, GPU will not be used.
2025-03-06 21:20:44.546806: I external/local_xla/xla/tsl/cuda/cudart_stub.cc:32] Could not find cuda drivers on your machine, GPU will not be used.
2025-03-06 21:20:44.563804: E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:477] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
E0000 00:00:1741314044.592197   70578 cuda_dnn.cc:8310] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
E0000 00:00:1741314044.600920   70578 cuda_blas.cc:1418] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
2025-03-06 21:20:44.629647: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: AVX2 FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
Initial dataset head:
   Age Sex ChestPainType  RestingBP  Cholesterol  FastingBS RestingECG  MaxHR ExerciseAngina  Oldpeak ST_Slope  HeartDisease
0   40   M           ATA        140          289          0     Normal    172              N      0.0       Up             0
1   49   F           NAP        160          180          0     Normal    156              N      1.0     Flat             1
2   37   M           ATA        130          283          0         ST     98              N      0.0       Up             0
3   48   F           ASY        138          214          0     Normal    108              Y      1.5     Flat             1
4   54   M           NAP        150          195          0     Normal    122              N      0.0       Up             0

Transformed data head:
   cat_encoder__Sex_F  cat_encoder__Sex_M  cat_encoder__ChestPainType_ASY  ...  remainder__MaxHR  remainder__Oldpeak  remainder__HeartDisease
0                 0.0                 1.0                             0.0  ...             172.0                 0.0                      0.0
1                 1.0                 0.0                             0.0  ...             156.0                 1.0                      1.0
2                 0.0                 1.0                             0.0  ...              98.0                 0.0                      0.0
3                 1.0                 0.0                             1.0  ...             108.0                 1.5                      1.0
4                 0.0                 1.0                             0.0  ...             122.0                 0.0                      0.0

[5 rows x 22 columns]

Columns in dataset: ['cat_encoder__Sex_F', 'cat_encoder__Sex_M', 'cat_encoder__ChestPainType_ASY', 'cat_encoder__ChestPainType_ATA', 'cat_encoder__ChestPainType_NAP', 'cat_encoder__ChestPainType_TA', 'cat_encoder__FastingBS_0', 'cat_encoder__FastingBS_1', 'cat_encoder__RestingECG_LVH', 'cat_encoder__RestingECG_Normal', 'cat_encoder__RestingECG_ST', 'cat_encoder__ExerciseAngina_N', 'cat_encoder__ExerciseAngina_Y', 'cat_encoder__ST_Slope_Down', 'cat_encoder__ST_Slope_Flat', 'cat_encoder__ST_Slope_Up', 'remainder__Age', 'remainder__RestingBP', 'remainder__Cholesterol', 'remainder__MaxHR', 'remainder__Oldpeak', 'remainder__HeartDisease']
/home/achin/.local/lib/python3.10/site-packages/keras/src/layers/core/dense.py:87: UserWarning: Do not pass an `input_shape`/`input_dim` argument to a layer. When using Sequential models, prefer using an `Input(shape)` object as the first layer in the model instead.
  super().__init__(activity_regularizer=activity_regularizer, **kwargs)
2025-03-06 21:20:47.250990: E external/local_xla/xla/stream_executor/cuda/cuda_driver.cc:152] failed call to cuInit: INTERNAL: CUDA error: Failed call to cuInit: UNKNOWN ERROR (303)

Model 1 - Basic Neural Network
Test Loss: 0.30997395515441895
Test Accuracy: 0.8801742792129517

Model 2 - Neural Network with L2 Regularization
Test Loss: 0.43498915433883667
Test Accuracy: 0.8714597225189209

Model 3 - Larger Neural Network with L2 Regularization
Test Loss: 0.6921901702880859
Test Accuracy: 0.601307213306427
'''




import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
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
