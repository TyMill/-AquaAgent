
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import plotly.express as px

class Visualizer:
    def __init__(self, data):
        self.data = data

    def plot_distribution(self, column):
        plt.figure(figsize=(8, 5))
        sns.histplot(self.data[column], kde=True)
        plt.title(f'Distribution of {column}')
        plt.xlabel(column)
        plt.ylabel('Frequency')
        plt.show()

    def plot_correlation_heatmap(self):
        corr = self.data.corr()
        plt.figure(figsize=(10, 6))
        sns.heatmap(corr, annot=True, cmap='coolwarm')
        plt.title("Correlation Heatmap")
        plt.show()

    def plot_scatter(self, x, y, hue=None):
        plt.figure(figsize=(8, 5))
        sns.scatterplot(data=self.data, x=x, y=y, hue=hue)
        plt.title(f'Scatter Plot: {y} vs {x}')
        plt.xlabel(x)
        plt.ylabel(y)
        plt.show()

    def plot_pca(self, features, hue=None):
        pca = PCA(n_components=2)
        components = pca.fit_transform(self.data[features])
        df_pca = pd.DataFrame(components, columns=['PC1', 'PC2'])
        if hue:
            df_pca[hue] = self.data[hue].values
        plt.figure(figsize=(8, 6))
        sns.scatterplot(data=df_pca, x='PC1', y='PC2', hue=hue)
        plt.title("PCA Plot")
        plt.show()

    def plot_tsne(self, features, hue=None, perplexity=30, learning_rate=200):
        tsne = TSNE(n_components=2, perplexity=perplexity, learning_rate=learning_rate)
        components = tsne.fit_transform(self.data[features])
        df_tsne = pd.DataFrame(components, columns=['Dim1', 'Dim2'])
        if hue:
            df_tsne[hue] = self.data[hue].values
        plt.figure(figsize=(8, 6))
        sns.scatterplot(data=df_tsne, x='Dim1', y='Dim2', hue=hue)
        plt.title("t-SNE Plot")
        plt.show()

    def plot_interactive_bubble(self, x, y, size, hover):
        fig = px.scatter(self.data, x=x, y=y, size=size, hover_name=hover, size_max=60)
        fig.show()
