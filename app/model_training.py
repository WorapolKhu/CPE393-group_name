from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import pandas as pd
import numpy as np
from data_processing import DataProcessor

preprocessor = DataProcessor()
# Load data

def train_model(model, X_train, y_train, X_test, y_test):
    # Train the model
    model.fit(X_train, y_train)

    # Make predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Evaluate the model
    mse_train = mean_squared_error(y_train, y_pred_train)
    mse_test = mean_squared_error(y_test, y_pred_test)
    
    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)
    
    mae_train = mean_absolute_error(y_train, y_pred_train)
    mae_test = mean_absolute_error(y_test, y_pred_test)
    
    print(f"Train MSE: {mse_train:.2f}, R2: {r2_train:.2f}, MAE: {mae_train:.2f}")
    print(f"Test MSE: {mse_test:.2f}, R2: {r2_test:.2f}, MAE: {mae_test:.2f}")
    
    # Save the model
    model_path = f'../models/{model.__class__.__name__}.pkl'
    joblib.dump(model, model_path)

    return model,{ 
        'mse_train': mse_train,
        'mse_test': mse_test,
        'r2_train': r2_train,
        'r2_test': r2_test,
        'mae_train': mae_train,
        'mae_test': mae_test
    }
    
if __name__ == "__main__":
    X_train, X_test, y_train, y_test = preprocessor.data_process_train('../data/House_Rent_Dataset.csv')
    random_forest_model = RandomForestRegressor(random_state=42)
    linear_reg_model = LinearRegression()

    rf_model, rf_metrics = train_model(random_forest_model, X_train, y_train, X_test, y_test)
    lr_model, lr_metrics = train_model(linear_reg_model, X_train, y_train, X_test, y_test)


