from flask import Flask, request, jsonify, render_template
import joblib
from data_processing import DataProcessor
import numpy as np
import pandas as pd
app = Flask(__name__)

# Load model
model = joblib.load('../models/RandomForestRegressor.pkl')

def model_predict(data):
    prediction = model.predict(data)
    return prediction[0] if len(prediction) == 1 else prediction

def predict_with_confidence(data):
    preprocess = DataProcessor(load_encoder=True)
    
    processed_data = preprocess.clean_input(data)

    if hasattr(model, 'estimators_'):
        # For ensemble models like Random Forest
        predictions = []
        for estimator in model.estimators_:
            pred = estimator.predict(processed_data)
            predictions.append(pred[0])
        
        mean_pred = np.mean(predictions)
        std_pred = np.std(predictions)
        
        return {
            'prediction': mean_pred,
            'confidence_interval': {
                'lower': mean_pred - 1.96 * std_pred,
                'upper': mean_pred + 1.96 * std_pred
            }
        }
    else:
        return {'prediction': model_predict(processed_data)}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from request
        data = request.get_json()
        
        # Required fields
        required_fields = ['BHK', 'Size', 'Area Type', 'City', 'Furnishing Status', 
                          'Tenant Preferred', 'Bathroom', 'Point of Contact', 'Floor', 'ofFloor']
        
        # Validate input
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400

        # Make prediction
        prediction = predict_with_confidence(data)

        return jsonify({
            'success': True,
            'prediction': prediction
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy'})

@app.route('/model_info', methods=['GET'])
def model_info():
    return jsonify({
        'model_type': 'Random Forest Regressor',
        'version': '1.0',
        'features': ['BHK', 'Size','Floor','ofFloor', 'Area Type', 'City', 'Furnishing Status', 
                    'Tenant Preferred', 'Bathroom', 'Point of Contact']
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

