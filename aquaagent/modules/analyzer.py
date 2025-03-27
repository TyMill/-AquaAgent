
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind, pearsonr
from sklearn.feature_selection import mutual_info_classif, SelectKBest, f_classif
from sklearn.ensemble import IsolationForest

class Analyzer:
    def __init__(self, data):
        self.data = data

    def basic_statistics(self):
        desc = self.data.describe().T
        desc['range'] = desc['max'] - desc['min']
        return desc[['mean', '50%', 'std', 'range']]

    def correlation_matrix(self, method='pearson'):
        return self.data.corr(method=method)

    def visualize_correlation(self):
        plt.figure(figsize=(10, 6))
        sns.heatmap(self.correlation_matrix(), annot=True, cmap='coolwarm')
        plt.title("Correlation Matrix")
        plt.show()

    def feature_ranking(self, target_column, method='mutual_info'):
        X = self.data.drop(columns=[target_column])
        y = self.data[target_column]
        if method == 'mutual_info':
            scores = mutual_info_classif(X, y)
        elif method == 'f_classif':
            selector = SelectKBest(score_func=f_classif, k='all')
            selector.fit(X, y)
            scores = selector.scores_
        else:
            raise ValueError("Invalid method for feature ranking.")
        return pd.Series(scores, index=X.columns).sort_values(ascending=False)

    def t_test(self, group_column, target_column):
        groups = self.data[group_column].unique()
        if len(groups) != 2:
            raise ValueError("T-test requires exactly two groups.")
        g1 = self.data[self.data[group_column] == groups[0]][target_column]
        g2 = self.data[self.data[group_column] == groups[1]][target_column]
        stat, p = ttest_ind(g1, g2)
        return {"t_statistic": stat, "p_value": p}

    def detect_outliers(self, contamination=0.05):
        iso = IsolationForest(contamination=contamination, random_state=42)
        labels = iso.fit_predict(self.data.select_dtypes(include=np.number))
        outliers = self.data[labels == -1]
        return outliers

    def analyze_distributions(self, columns=None):
        if columns is None:
            columns = self.data.select_dtypes(include=np.number).columns
        for col in columns:
            plt.figure(figsize=(8, 4))
            sns.histplot(self.data[col], kde=True)
            plt.title(f"Distribution of {col}")
            plt.show()
