
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.linear_model import LinearRegression
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model

class ImputationEngine:
    def __init__(self, data):
        self.data = data.copy()

    def mean_imputation(self):
        return self.data.fillna(self.data.mean())

    def median_imputation(self):
        return self.data.fillna(self.data.median())

    def knn_imputation(self, n_neighbors=5):
        imputer = KNNImputer(n_neighbors=n_neighbors)
        imputed_data = pd.DataFrame(imputer.fit_transform(self.data), columns=self.data.columns)
        return imputed_data

    def regression_imputation(self):
        imputer = IterativeImputer(estimator=LinearRegression(), max_iter=10, random_state=0)
        imputed_data = pd.DataFrame(imputer.fit_transform(self.data), columns=self.data.columns)
        return imputed_data

    def autoencoder_imputation(self, epochs=50, batch_size=32):
        input_dim = self.data.shape[1]

        input_layer = Input(shape=(input_dim,))
        encoded = Dense(64, activation='relu')(input_layer)
        encoded = Dense(32, activation='relu')(encoded)
        decoded = Dense(64, activation='relu')(encoded)
        output_layer = Dense(input_dim, activation='linear')(decoded)

        autoencoder = Model(inputs=input_layer, outputs=output_layer)
        autoencoder.compile(optimizer='adam', loss='mean_squared_error')

        # Replace missing with mean temporarily for training
        train_data = self.data.copy().fillna(self.data.mean())

        autoencoder.fit(train_data, train_data, epochs=epochs, batch_size=batch_size, verbose=0)

        imputed_array = autoencoder.predict(train_data)
        imputed_data = pd.DataFrame(imputed_array, columns=self.data.columns)
        return imputed_data
