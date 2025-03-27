
import shap
import lime
import lime.lime_tabular
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

class ModelExplainer:
    def __init__(self, model, X_train):
        self.model = model
        self.X_train = X_train
        self.shap_explainer = None

    def explain_with_shap(self, X_sample=None):
        try:
            if hasattr(self.model, "predict_proba"):
                self.shap_explainer = shap.Explainer(self.model.predict_proba, self.X_train)
            else:
                self.shap_explainer = shap.Explainer(self.model, self.X_train)

            shap_values = self.shap_explainer(X_sample or self.X_train[:100])
            shap.summary_plot(shap_values, X_sample or self.X_train[:100])
        except Exception as e:
            print(f"[SHAP Error] {e}")

    def explain_with_lime(self, X_sample, feature_names=None, class_names=None):
        try:
            explainer = lime.lime_tabular.LimeTabularExplainer(
                training_data=self.X_train.values,
                feature_names=feature_names or self.X_train.columns.tolist(),
                class_names=class_names or ["Class 0", "Class 1"],
                mode="classification"
            )
            explanation = explainer.explain_instance(
                data_row=X_sample.values[0],
                predict_fn=self.model.predict_proba
            )
            explanation.show_in_notebook()
            return explanation
        except Exception as e:
            print(f"[LIME Error] {e}")
