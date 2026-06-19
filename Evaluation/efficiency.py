"""
Computational Efficiency Evaluation
-----------------------------------
Measures:
1. Training Time (seconds)
2. Inference Time (milliseconds/sample)
"""

import time
import numpy as np


def measure_training_time(model,
                          X_train,
                          y_train,
                          epochs=100,
                          batch_size=32,
                          validation_split=0.2):
    """
    Measure model training time.

    Returns
    -------
    history
    training_time
    """

    start_time = time.time()

    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        verbose=1
    )

    end_time = time.time()

    training_time = end_time - start_time

    return history, training_time


def measure_inference_time(model,
                           X_test):
    """
    Measure average inference time per sample.

    Returns
    -------
    inference_time_ms
    """

    start_time = time.time()

    predictions = model.predict(
        X_test,
        verbose=0
    )

    end_time = time.time()

    total_time = end_time - start_time

    inference_time_ms = (
        total_time / len(X_test)
    ) * 1000

    return inference_time_ms


def efficiency_report(model_name,
                      training_time,
                      inference_time):
    """
    Display efficiency results.
    """

    print("\n" + "=" * 50)
    print("COMPUTATIONAL EFFICIENCY REPORT")
    print("=" * 50)

    print(f"Model: {model_name}")

    print(
        f"Training Time: "
        f"{training_time:.2f} seconds"
    )

    print(
        f"Inference Time: "
        f"{inference_time:.4f} ms/sample"
    )

    print("=" * 50)
