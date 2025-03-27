
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import xgboost as xgb
import warnings
warnings.filterwarnings("ignore")

# AutoML
from tpot import TPOTClassifier
import autosklearn.classification

class MLSelector:
    def __init__(self):
        self.models = {
            'logistic_regression': LogisticRegression(max_iter=1000),
            'random_forest': RandomForestClassifier(),
            'svm': SVC(probability=True),
            'xgboost': xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
        }
        self.best_model = None
        self.best_score = -np.inf
        self.results = {}

    def train_and_select(self, X, y, scoring='accuracy', cv=5):
        for name, model in self.models.items():
            try:
                score = cross_val_score(model, X, y, cv=cv, scoring=scoring).mean()
                self.results[name] = score
                if score > self.best_score:
                    self.best_score = score
                    self.best_model = name
            except Exception as e:
                self.results[name] = f"Error: {e}"
        return self.best_model, self.results

    def train_best_model(self, X, y):
        if self.best_model:
            model = self.models[self.best_model]
            model.fit(X, y)
            return model
        return None

    def predict(self, model, X_test):
        return model.predict(X_test)

    def use_xgboost(self, X, y):
        model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
        model.fit(X, y)
        return model

    def use_automl_autosklearn(self, X, y, time_left=60, per_run_time=30):
        automl = autosklearn.classification.AutoSklearnClassifier(
            time_left_for_this_task=time_left,
            per_run_time_limit=per_run_time,
            metric=autosklearn.metrics.accuracy
        )
        automl.fit(X, y)
        self.models['autosklearn'] = automl
        self.best_model = 'autosklearn'
        return automl

    def use_automl_tpot(self, X, y, generations=5, population_size=20):
        tpot = TPOTClassifier(generations=generations, population_size=population_size, verbosity=2, random_state=42, disable_update_check=True)
        tpot.fit(X, y)
        self.models['tpot'] = tpot.fitted_pipeline_
        self.best_model = 'tpot'
        return tpot
