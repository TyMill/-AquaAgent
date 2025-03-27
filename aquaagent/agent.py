
from aquaagent.modules.data_handler import DataHandler
from aquaagent.modules.analyzer import Analyzer
from aquaagent.modules.imputation_engine import ImputationEngine
from aquaagent.modules.ml_selector import MLSelector
from aquaagent.modules.auto_optimizer import AutoOptimizer
from aquaagent.modules.visualizer import Visualizer
from aquaagent.modules.reporter import Reporter
from aquaagent.modules.model_explainer import ModelExplainer

class AquaAgent:
    def __init__(self, data_path, file_type='csv', mode='autonomous'):
        self.mode = mode  # 'autonomous' or 'interactive'
        self.data_handler = DataHandler(data_path, file_type)
        self.data = self.data_handler.data
        self.cleaned_data = None
        self.imputed_data = None
        self.analyzer = None
        self.visualizer = None
        self.model = None
        self.X = None
        self.y = None

    def prepare_data(self, target_column, impute_strategy='mean'):
        self.cleaned_data = self.data_handler.clean_data()
        imputer = ImputationEngine(self.cleaned_data)
        if self.mode == 'interactive':
            print("[INTERACTIVE] Imputation strategy selected:", impute_strategy)
        if impute_strategy == 'mean':
            self.imputed_data = imputer.mean_imputation()
        elif impute_strategy == 'median':
            self.imputed_data = imputer.median_imputation()
        elif impute_strategy == 'knn':
            self.imputed_data = imputer.knn_imputation()
        elif impute_strategy == 'regression':
            self.imputed_data = imputer.regression_imputation()
        elif impute_strategy == 'autoencoder':
            self.imputed_data = imputer.autoencoder_imputation()
        else:
            self.imputed_data = self.cleaned_data.fillna(0)

        self.analyzer = Analyzer(self.imputed_data)
        self.visualizer = Visualizer(self.imputed_data)

        self.X = self.imputed_data.drop(columns=[target_column])
        self.y = self.imputed_data[target_column]

    def train_model(self, method='ml_selector'):
        if self.mode == 'interactive':
            print(f"[INTERACTIVE] Model training method selected: {method}")
        if method == 'ml_selector':
            selector = MLSelector()
            best_model_name, results = selector.train_and_select(self.X, self.y)
            self.model = selector.train_best_model(self.X, self.y)
            print(f"[AquaAgent] Best Model: {best_model_name} — Accuracy: {results[best_model_name]}")
        elif method == 'xgboost':
            self.model = MLSelector().use_xgboost(self.X, self.y)
        elif method == 'autosklearn':
            self.model = MLSelector().use_automl_autosklearn(self.X, self.y)
        elif method == 'tpot':
            self.model = MLSelector().use_automl_tpot(self.X, self.y)
        elif method == 'optuna':
            self.model, params = AutoOptimizer('xgboost').optimize_with_optuna(self.X, self.y)
            print(f"[AquaAgent] Optuna Params: {params}")

    def visualize(self):
        if self.mode == 'interactive':
            print("[INTERACTIVE] Running visualizations...")
        self.visualizer.plot_correlation_heatmap()
        self.visualizer.plot_pca(self.X.columns.tolist())
        self.visualizer.plot_tsne(self.X.columns.tolist())

    def explain_model(self):
        if self.mode == 'interactive':
            print("[INTERACTIVE] Explaining model with SHAP and LIME...")
        explainer = ModelExplainer(self.model, self.X)
        explainer.explain_with_shap()
        explainer.explain_with_lime(self.X.sample(1))

    def generate_reports(self):
        if self.mode == 'interactive':
            print("[INTERACTIVE] Generating reports...")
        reporter = Reporter()
        X_train = self.X
        y_train = self.y
        reporter.generate_classification_report(self.model, X_train, y_train)
        reporter.suggest_further_analysis()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="AquaAgent CLI - Intelligent Water Quality Analysis")
    parser.add_argument("file", help="Path to the dataset (CSV/Excel/JSON)")
    parser.add_argument("--target", default="target", help="Target column for ML")
    parser.add_argument("--mode", choices=["autonomous", "interactive"], default="autonomous", help="Run mode")
    parser.add_argument("--file_type", choices=["csv", "excel", "json"], default="csv", help="Type of input file")
    parser.add_argument("--model", choices=["ml_selector", "xgboost", "autosklearn", "tpot", "optuna"], default="ml_selector", help="Model training strategy")
    parser.add_argument("--impute", choices=["mean", "median", "knn", "regression", "autoencoder"], default="mean", help="Missing value strategy")
    
    args = parser.parse_args()

    print(f"
[ AquaAgent ] Starting in {args.mode.upper()} mode...
")
    agent = AquaAgent(data_path=args.file, file_type=args.file_type, mode=args.mode)
    agent.prepare_data(target_column=args.target, impute_strategy=args.impute)
    agent.train_model(method=args.model)
    agent.visualize()
    agent.explain_model()
    agent.generate_reports()
    print("
[ AquaAgent ] Analysis complete. Reports saved.
")

if __name__ == '__main__':
    main()
