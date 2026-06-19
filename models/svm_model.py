from sklearn.svm import SVC

def build_svm():

    model = SVC(
        kernel='rbf',
        C=1.0,
        probability=True,
        random_state=42
    )

    return model
