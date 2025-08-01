FROM python:3.9

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/

# Train the model during image build
RUN python src/train.py

# Train the model during image build
RUN python src/quantize.py

# Set default entry point to run predictions
CMD ["python", "src/predict.py"]