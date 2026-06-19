import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from preprocessing.data_cleaning import preprocess_data
from preprocessing.feature_selection import correlation_feature_selection
from preprocessing.pca_reduction import (
    apply_pca,
    explained_variance_report
)

from models.aedn_model import build_aedn

from evaluation.confusion_matrix import plot_cm
from evaluation.efficiency import (
    measure_training_time,
    measure_inference_time,
    efficiency_report
)

# ----------------------------------
# Load Dataset
# ----------------------------------

DATASET_PATH = "datasets/pcos/pcos.csv"

TARGET_COLUMN = "PCOS"

df = pd.read_csv(DATASET_PATH)

print("Dataset Shape:", df.shape)

# ----------------------------------
# Feature Selection
# ----------------------------------

selected_features = correlation_feature_selection(
    df,
    TARGET_COLUMN,
    threshold=0.05
)

df = df[selected_features + [TARGET_COLUMN]]

# ----------------------------------
# Preprocessing
# ----------------------------------

X, y = preprocess_data(
    df,
    TARGET_COLUMN
)

# ----------------------------------
# Label Encoding
# ----------------------------------

encoder = LabelEncoder()

y = encoder.fit_transform(y)

# ----------------------------------
# PCA
# ----------------------------------

X, pca = apply_pca(
    X,
    n_components=10
)

explained_variance_report(
    pca
)

# ----------------------------------
# Split
# ----------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=42
)

# ----------------------------------
# AEDN
# ----------------------------------

num_classes = len(np.unique(y))

model = build_aedn(
    input_dim=X_train.shape[1],
    num_classes=num_classes
)

# ----------------------------------
# Training
# ----------------------------------

history, train_time = measure_training_time(
    model,
    X_train,
    y_train,
    epochs=100,
    batch_size=32
)

# ----------------------------------
# Prediction
# ----------------------------------

y_prob = model.predict(X_test)

y_pred = np.argmax(
    y_prob,
    axis=1
)

# ----------------------------------
# Metrics
# ----------------------------------

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred,
    average='weighted'
)

recall = recall_score(
    y_test,
    y_pred,
    average='weighted'
)

f1 = f1_score(
    y_test,
    y_pred,
    average='weighted'
)

print("\nPCOS RESULTS")
print("Accuracy :", accuracy)
print("Precision:", precision)
print("Recall   :", recall)
print("F1 Score :", f1)

# ----------------------------------
# Efficiency
# ----------------------------------

inference_time = measure_inference_time(
    model,
    X_test
)

efficiency_report(
    "AEDN-PCOS",
    train_time,
    inference_time
)

# ----------------------------------
# Confusion Matrix
# ----------------------------------

plot_cm(
    y_test,
    y_pred,
    "figures/pcos_confusion_matrix.png"
)

print(
    "\nConfusion matrix saved."
)
