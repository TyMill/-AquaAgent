
import optuna
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import xgboost as xgb
import numpy as np

class AutoOptimizer:
    def __init__(self, model_name='random_forest'):
        self.model_name = model_name
        self.best_model = None
        self.best_params = {}

    def optimize_with_optuna(self, X, y, n_trials=30, scoring='accuracy', cv=5):
        def objective(trial):
            if self.model_name == 'random_forest':
                n_estimators = trial.suggest_int('n_estimators', 50, 200)
                max_depth = trial.suggest_int('max_depth', 2, 20)
                model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
            elif self.model_name == 'svm':
                C = trial.suggest_loguniform('C', 1e-3, 1e2)
                gamma = trial.suggest_loguniform('gamma', 1e-4, 1e-1)
                model = SVC(C=C, gamma=gamma)
            elif self.model_name == 'xgboost':
                eta = trial.suggest_float('eta', 0.01, 0.3)
                max_depth = trial.suggest_int('max_depth', 3, 10)
                subsample = trial.suggest_float('subsample', 0.5, 1.0)
                model = xgb.XGBClassifier(eta=eta, max_depth=max_depth, subsample=subsample, use_label_encoder=False, eval_metric='logloss')
            else:
                raise ValueError("Unsupported model")

            score = cross_val_score(model, X, y, cv=cv, scoring=scoring).mean()
            return score

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)
        self.best_params = study.best_params

        if self.model_name == 'random_forest':
            self.best_model = RandomForestClassifier(**self.best_params)
        elif self.model_name == 'svm':
            self.best_model = SVC(**self.best_params)
        elif self.model_name == 'xgboost':
            self.best_model = xgb.XGBClassifier(**self.best_params, use_label_encoder=False, eval_metric='logloss')

        self.best_model.fit(X, y)
        return self.best_model, self.best_params

    def optimize_with_gridsearch(self, model, param_grid, X, y, scoring='accuracy', cv=5):
        grid = GridSearchCV(model, param_grid, scoring=scoring, cv=cv)
        grid.fit(X, y)
        self.best_model = grid.best_estimator_
        self.best_params = grid.best_params_
        return self.best_model, self.best_params
