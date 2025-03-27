
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from jinja2 import Environment, FileSystemLoader
import os

class Reporter:
    def __init__(self, output_dir='reports'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.env = Environment(loader=FileSystemLoader(searchpath='./templates'))

    def generate_classification_report(self, model, X_test, y_test, report_name="classification_report.html"):
        try:
            predictions = model.predict(X_test)
            report_text = classification_report(y_test, predictions, output_dict=True)
            cm = confusion_matrix(y_test, predictions)

            plt.figure(figsize=(6, 4))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title("Confusion Matrix")
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            plot_path = os.path.join(self.output_dir, "confusion_matrix.png")
            plt.savefig(plot_path)
            plt.close()

            template = self.env.get_template("classification_template.html")
            html_out = template.render(
                classification=report_text,
                confusion_matrix_image="confusion_matrix.png"
            )
            output_path = os.path.join(self.output_dir, report_name)
            with open(output_path, "w") as f:
                f.write(html_out)

            print(f"[✔] Report saved to {output_path}")
        except Exception as e:
            print(f"[✘] Failed to generate classification report: {e}")

    def suggest_further_analysis(self, report_name="suggestions.html"):
        try:
            suggestions = [
                "Use SHAP or LIME to explain model predictions.",
                "Apply PCA or t-SNE to explore data structure.",
                "Try different imputation strategies to improve data quality.",
                "Consider AutoML to optimize your model pipeline.",
                "Analyze time trends if the data has timestamps."
            ]
            template = self.env.get_template("suggestion_template.html")
            html_out = template.render(suggestions=suggestions)
            output_path = os.path.join(self.output_dir, report_name)
            with open(output_path, "w") as f:
                f.write(html_out)

            print(f"[✔] Suggestions saved to {output_path}")
        except Exception as e:
            print(f"[✘] Failed to generate suggestion report: {e}")
