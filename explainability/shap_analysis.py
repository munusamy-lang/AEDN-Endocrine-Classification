import shap
import matplotlib.pyplot as plt

def run_shap(model,
             X_train,
             X_test):

    explainer = shap.KernelExplainer(
        model.predict,
        X_train[:100]
    )

    shap_values = explainer.shap_values(
        X_test[:100]
    )

    shap.summary_plot(
        shap_values,
        X_test[:100]
    )

    plt.savefig(
        "figures/shap_summary.png"
    )
