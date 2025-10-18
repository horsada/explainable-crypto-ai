import shap
import matplotlib.pyplot as plt
import os
import pandas as pd

class ModelExplainer:
    def __init__(self, model, features):
        self.model = model
        self.features = features
        self.explainer = shap.Explainer(model)

    def explain_instance(self, X, output_path="plots/latest_shap_explanation.png"):
        shap_values = self.explainer(X)
        shap.plots.bar(shap_values, show=False)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, bbox_inches="tight")
        plt.clf()
        print(f"✅ Saved SHAP bar plot to {output_path}")

    def explain_global(self, df, out_dir="plots"):
        X = df[self.features]
        shap_values = self.explainer(X)
        os.makedirs(out_dir, exist_ok=True)

        shap.summary_plot(shap_values, X, show=False)
        plt.savefig(f"{out_dir}/shap_summary.png", bbox_inches="tight")
        plt.clf()

        shap.plots.bar(shap_values, show=False)
        plt.savefig(f"{out_dir}/shap_bar.png", bbox_inches="tight")
        plt.clf()

        print(f"✅ Saved global SHAP plots to {out_dir}")

        # Save SHAP values to CSV
        shap_df = pd.DataFrame(shap_values.values, columns=self.features, index=X.index)
        shap_df.to_csv(f"{out_dir}/shap_values.csv")
        print(f"✅ Saved SHAP values to {out_dir}/shap_values.csv")


