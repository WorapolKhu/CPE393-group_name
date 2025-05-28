import pandas as pd
import numpy as np
import pickle

class DataProcessor:
    def __init__(self, load_encoder=False, scaler_path=None, encoder_path=None):
        self.load_encoder = load_encoder
        self.scaler_path = scaler_path
        self.encoder_path = encoder_path
        self.scaler = None
        self.encoder = None
        self.categorical_features = ['Area Type', 'City', 'Furnishing Status', 'Tenant Preferred', 'Point of Contact']
        self.numerical_features = ['BHK', 'Size', 'Bathroom', 'CurrentFloor', 'TotalFloors']

        if self.load_encoder:
            self.load_encoders()

    def load_encoders(self):
        try:
            with open(self.scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            with open(self.encoder_path, 'rb') as f:
                self.encoder = pickle.load(f)
        except Exception as e:
            raise Exception(f"Error loading encoders: {e}")

    def categorical_encoding(self, data):
        
        if self.encoder is None or self.load_encoder==False: 
            self.encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            self.encoder.fit(data[self.categorical_features])
            with open(self.encoder_path, 'wb') as f:
                pickle.dump(self.encoder, f)
        # Print number of categories
        encoded_data = self.encoder.transform(data[self.categorical_features])
        return encoded_data

    def numerical_scaling(self, data):
        
        if self.scaler is None or self.load_encoder==False:
            self.scaler = StandardScaler()
            self.scaler.fit(data[self.numerical_features])
            with open(self.scaler_path, 'wb') as f:
                pickle.dump(self.scaler, f)
        
        scaled_data = self.scaler.transform(data[self.numerical_features])
        
        return scaled_data

        
    def clean_input(self, data):
        df = pd.DataFrame([data])

        df[self.categorical_features] = df[self.categorical_features].astype(str)
        df[self.numerical_features] = df[self.numerical_features].astype(float)

        encoded_cat = self.encoder.transform(df[self.categorical_features])
        scaled_num = self.scaler.transform(df[self.numerical_features])

        final_data = np.concatenate([scaled_num, encoded_cat], axis=1)
        return final_data
