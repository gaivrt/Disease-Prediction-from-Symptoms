import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder

class DataLoader:
    def __init__(self, data_dir='./dataset'):
        self.data_dir = data_dir
        self.train_path = os.path.join(data_dir, 'training_data.csv')
        self.test_path = os.path.join(data_dir, 'test_data.csv')
        self.encoder = LabelEncoder()
        
    def load_data(self):
        """
        Loads training and testing data, cleans it, and encodes labels.
        Returns:
            X_train (pd.DataFrame): Training features
            y_train (np.array): Encoded training labels
            X_test (pd.DataFrame): Test features
            y_test (np.array): Encoded test labels
            encoder (LabelEncoder): Fitted label encoder
        """
        # Load Raw Data
        train_df = pd.read_csv(self.train_path)
        test_df = pd.read_csv(self.test_path)
        
        # Clean Data (Main.py logic: drop last empty column if exists)
        # Original simple logic: cols = cols[:-2] 
        # But let's be more robust: drop 'prognosis' and any 'Unnamed' columns
        
        X_train = self._clean_features(train_df)
        y_train_raw = train_df['prognosis']
        
        X_test = self._clean_features(test_df)
        y_test_raw = test_df['prognosis']
        
        # Encode Labels
        # Fit on training data
        y_train = self.encoder.fit_transform(y_train_raw)
        
        # Transform test data (handle unseen labels if necessary, though unlikely here)
        try:
            y_test = self.encoder.transform(y_test_raw)
        except ValueError:
            print("Warning: Unseen labels in test set. This shouldn't happen in this dataset.")
            # Fallback or error handling could go here
            y_test = np.zeros(len(y_test_raw)) # Placeholders
            
        return X_train, y_train, X_test, y_test, self.encoder

    def _clean_features(self, df):
        """Removes target and garbage columns."""
        # Drop prognosis
        if 'prognosis' in df.columns:
            df = df.drop('prognosis', axis=1)
            
        # Drop Unnamed columns (common artifact in this dataset)
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        
        return df

if __name__ == "__main__":
    loader = DataLoader()
    X_tr, y_tr, X_te, y_te, enc = loader.load_data()
    print(f"Train Shape: {X_tr.shape}, {y_tr.shape}")
    print(f"Test Shape: {X_te.shape}, {y_te.shape}")
    print(f"Classes: {len(enc.classes_)}")
