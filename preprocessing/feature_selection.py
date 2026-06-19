import pandas as pd
import numpy as np

def correlation_feature_selection(df,
                                  target_column,
                                  threshold=0.05):

    corr = df.corr(numeric_only=True)

    target_corr = abs(corr[target_column])

    selected_features = target_corr[
        target_corr > threshold
    ].index.tolist()

    selected_features.remove(target_column)

    return selected_features
