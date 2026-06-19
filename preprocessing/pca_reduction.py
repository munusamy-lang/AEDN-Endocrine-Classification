"""
PCA-based Dimensionality Reduction
----------------------------------
This module performs Principal Component Analysis (PCA)
to reduce feature dimensionality while preserving the
maximum amount of variance in the data.
"""

from sklearn.decomposition import PCA
import pandas as pd
import numpy as np


def apply_pca(X,
              n_components=10):
    """
    Apply PCA to the feature matrix.

    Parameters
    ----------
    X : array-like
        Input feature matrix

    n_components : int
        Number of principal components

    Returns
    -------
    X_pca : ndarray
        PCA transformed features

    pca : PCA object
        Trained PCA model
    """

    pca = PCA(
        n_components=n_components,
        random_state=42
    )

    X_pca = pca.fit_transform(X)

    return X_pca, pca


def explained_variance_report(pca):
    """
    Display explained variance information.
    """

    print("\nExplained Variance Ratio:")
    print(pca.explained_variance_ratio_)

    print("\nTotal Variance Preserved:")
    print(
        round(
            np.sum(
                pca.explained_variance_ratio_
            ) * 100,
            2
        ),
        "%"
    )


def save_pca_features(X_pca,
                      output_path):
    """
    Save PCA-transformed features to CSV.
    """

    columns = [
        f"PC{i+1}"
        for i in range(X_pca.shape[1])
    ]

    df = pd.DataFrame(
        X_pca,
        columns=columns
    )

    df.to_csv(
        output_path,
        index=False
    )

    print(
        f"PCA features saved to {output_path}"
    )
