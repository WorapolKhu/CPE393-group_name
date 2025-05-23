import pandas as pd
import numpy as np
import os
import pickle
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
import re

class DataProcessor:
    def __init__(self, load_encoder=False):
        self.load_encoder = load_encoder
        self.scaler = None
        self.encoder = None
        self.categorical_features = ['Area Type', 'City', 'Furnishing Status', 'Tenant Preferred', 'Point of Contact']
        self.numerical_features = ['BHK', 'Size', 'Bathroom', 'CurrentFloor','TotalFloors']
        if load_encoder:
            self.load_encoders()
            
    def load_encoders(self):
        try:
            with open('../models/standard_scaler.pkl', 'rb') as f:
                self.scaler = pickle.load(f)
            with open('../models/onehot_encoder.pkl', 'rb') as f:
                self.encoder = pickle.load(f)
            print("Encoders loaded successfully.")
        except FileNotFoundError:
            print("Encoder files not found. Please fit and save them first.")
            
    def categorical_encoding(self, data):
        
        if self.encoder is None or self.load_encoder==False: 
            self.encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            self.encoder.fit(data[self.categorical_features])
            with open('../models/onehot_encoder.pkl', 'wb') as f:
                pickle.dump(self.encoder, f)
        # Print number of categories
        encoded_data = self.encoder.transform(data[self.categorical_features])
        return encoded_data

    def numerical_scaling(self, data):
        
        if self.scaler is None or self.load_encoder==False:
            self.scaler = StandardScaler()
            self.scaler.fit(data[self.numerical_features])
            with open('../models/standard_scaler.pkl', 'wb') as f:
                pickle.dump(self.scaler, f)
        
        scaled_data = self.scaler.transform(data[self.numerical_features])
        
        return scaled_data

    def extract_floor_info(self,df):
        """
        Extracts current floor and total floors from the 'Floor' column,
        replaces non-numeric labels, converts to int, and drops the original 'Floor' column.
        """
        pattern = r'(?P<CurrentFloor>\w+)\s*out\s*of\s*(?P<TotalFloors>\d+)'
        df[['CurrentFloor', 'TotalFloors']] = df['Floor'].str.extract(pattern)
        floor_replacements = {
            'Ground': 0,
            'Basement': -1
        }
        
        # Replace na with the value of 'Floor' column
        df['CurrentFloor'] = df['CurrentFloor'].fillna(df['Floor'])
        df['TotalFloors'] = df['TotalFloors'].fillna(df['Floor'])
        
        df['CurrentFloor'] = df['CurrentFloor'].replace(floor_replacements)
        df['TotalFloors'] = df['TotalFloors'].replace(floor_replacements)

        df['CurrentFloor'] = pd.to_numeric(df['CurrentFloor'], errors='coerce').astype('Int64')
        df['TotalFloors'] = pd.to_numeric(df['TotalFloors'], errors='coerce').astype('Int64')

        df = df.drop(columns=['Floor'])
        return df


    
    def data_process_train(self, data_path):
        """Load and preprocess data"""
        # Load data
        data = pd.read_csv(data_path)
        
        # Split data into features and target
        X = data.drop('Rent', axis=1)
        y = data['Rent']
        y = np.array(y)
        
        # Drop unnecessary columns
        unused_columns = ['Posted On','Area Locality']
        X.drop(columns=unused_columns, inplace=True, errors='ignore')
        
        # Handle floor data to 'Floor' 
        X = self.extract_floor_info(X)
        print(X.columns)
        # Split categorical and numerical features
        categorical_data = self.categorical_encoding(X[self.categorical_features])
        numerical_data = self.numerical_scaling(X[self.numerical_features])
        
        X = np.concatenate((categorical_data, numerical_data), axis=1)           
        
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        return X_train, X_test, y_train, y_test
    
    def clean_input(self, input_data):
        # Convert to DataFrame if it's a dictionary
        if isinstance(input_data, dict):
            input_df = pd.DataFrame([input_data])
        else:
            input_df = input_data
        
        # Encode categorical features
        input_encoded = self.categorical_encoding(input_df[self.categorical_features])
        
        # Scale numerical features
        input_scaled = self.numerical_scaling(input_df[self.numerical_features])
        
        X = np.concatenate((input_encoded, input_scaled), axis=1)

        return X