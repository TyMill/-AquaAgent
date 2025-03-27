
import pandas as pd
import os

class DataHandler:
    def __init__(self):
        self.data = None

    def load_data(self, file_path, file_type='csv', **kwargs):
        try:
            if file_type == 'csv':
                self.data = pd.read_csv(file_path, **kwargs)
            elif file_type == 'excel':
                self.data = pd.read_excel(file_path, **kwargs)
            elif file_type == 'json':
                self.data = pd.read_json(file_path, **kwargs)
            elif file_type == 'parquet':
                self.data = pd.read_parquet(file_path, **kwargs)
            else:
                raise ValueError(f"Unsupported file type: {file_type}")
            return self.data
        except Exception as e:
            print(f"Error loading data: {e}")
            return None

    def remove_exact_duplicates(self):
        # Remove completely identical rows
        if self.data is not None:
            before = len(self.data)
            self.data = self.data.drop_duplicates()
            after = len(self.data)
            print(f"Removed {before - after} exact duplicates.")
        return self.data

    def identify_semantic_duplicates(self, id_columns):
        # Detect potential semantic duplicates based on ID-like columns
        if self.data is not None:
            grouped = self.data.groupby(id_columns).size().reset_index(name='counts')
            suspected = grouped[grouped['counts'] > 1]
            return suspected
        return pd.DataFrame()

    def standardize_column_names(self):
        if self.data is not None:
            self.data.columns = [col.strip().lower().replace(' ', '_') for col in self.data.columns]
        return self.data

    def get_info(self):
        if self.data is not None:
            print("Data Shape:", self.data.shape)
            print("Columns:", self.data.columns.tolist())
            print("Missing Values:", self.data.isnull().sum())
            print("Sample:")
            print(self.data.head())
