import pandas as pd

from sklearn.model_selection import train_test_split

from preprocessing.data_cleaning import *
from preprocessing.feature_selection import *
from preprocessing.pca_reduction import *

from models.aedn_model import *

df = pd.read_csv(
    "datasets/thyroid/thyroid.csv"
)

target = "Class"

selected = correlation_feature_selection(
    df,
    target
)

df = df[selected+[target]]

X,y = preprocess_data(
    df,
    target
)

X,_ = apply_pca(
    X,
    n_components=10
)

X_train,X_test,y_train,y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=42
)

model = build_aedn(
    X_train.shape[1],
    len(set(y))
)

history = model.fit(
    X_train,
    y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.2
)
