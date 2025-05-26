FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
COPY data/ ./data/
COPY models/ ./models/

# Train the model
RUN cd app && python model_training.py

WORKDIR /app/app

EXPOSE 5000

# Run the application
CMD ["python", "app.py"]