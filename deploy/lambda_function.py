import awsgi
from flask import Flask, render_template, request, jsonify
import boto3
import joblib
import numpy as np
import logging
from data_processing import DataProcessor
import os 

app = Flask(__name__)

logger = logging.getLogger()
logger.setLevel(logging.INFO)

s3 = boto3.client('s3')

model = None

def download_file_from_s3(bucket, key, local_path):
    if not os.path.exists(local_path):
        logger.info(f"Downloading {key} from s3://{bucket}")
        s3.download_file(bucket, key, local_path)
        logger.info(f"Downloaded {key} to {local_path}")
    else:
        logger.info(f"File already exists at {local_path}, skipping download.")

def load_model_and_encoders():
    global model

    bucket = 'osp-ai-inventrack-nonprod'
    model_key = 'model/RandomForestRegressor.pkl'
    scaler_key = 'model/standard_scaler.pkl'
    encoder_key = 'model/onehot_encoder.pkl'

    model_path = '/tmp/RandomForestRegressor.pkl'
    scaler_path = '/tmp/standard_scaler.pkl'
    encoder_path = '/tmp/onehot_encoder.pkl'

    try:
        download_file_from_s3(bucket, model_key, model_path)
        download_file_from_s3(bucket, scaler_key, scaler_path)
        download_file_from_s3(bucket, encoder_key, encoder_path)

        if not os.path.exists(model_path):
            logger.error(f"Model file not found at {model_path}")
            model = None
            return

        with open(model_path, 'rb') as f:
            model_loaded = joblib.load(f)
        
        model = model_loaded
        logger.info("Model loaded successfully")

    except Exception as e:
        logger.error(f"Failed to load model or encoders: {e}")
        model = None

load_model_and_encoders()

def predict_with_confidence(data):
    if model is None:
        raise Exception("Model is not loaded")

    processor = DataProcessor(
        load_encoder=True,
        encoder_path='/tmp/onehot_encoder.pkl',
        scaler_path='/tmp/standard_scaler.pkl'
    )
    processed_data = processor.clean_input(data)

    if hasattr(model, 'estimators_'):
        predictions = [est.predict(processed_data)[0] for est in model.estimators_]
        mean_pred = float(np.mean(predictions))
        std_pred = float(np.std(predictions))

        return {
            'prediction': mean_pred,
            'confidence_interval': {
                'lower': mean_pred - 1.96 * std_pred,
                'upper': mean_pred + 1.96 * std_pred
            }
        }
    else:
        pred = float(model.predict(processed_data)[0])
        return {'prediction': pred}

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        required_fields = ['BHK', 'Size', 'Area Type', 'City', 'Furnishing Status', 
                           'Tenant Preferred', 'Bathroom', 'Point of Contact', 'CurrentFloor','TotalFloors']

        logger.info(f"Received data: {data}")

        if not data:
            return jsonify({'error': 'No input data provided'}), 400

        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400

        prediction = predict_with_confidence(data)
        return jsonify({'success': True, 'prediction': prediction})

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/')
def index():
    return render_template('index.html')

def lambda_handler(event, context):
    return awsgi.response(app, event, context, base64_content_types={"image/png"})