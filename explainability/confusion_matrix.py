import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix

def plot_cm(
        y_true,
        y_pred,
        save_path):

    cm = confusion_matrix(
        y_true,
        y_pred
    )

    plt.figure(figsize=(7,6))

    sns.heatmap(
        cm,
        annot=True,
        fmt='d'
    )

    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    plt.savefig(save_path)
    plt.close()
