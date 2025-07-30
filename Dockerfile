# Use Python 3.9 full image
FROM python:3.9

# Set working directory
WORKDIR /app

# Copy and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy model and scripts
COPY src/ ./src/
COPY model.pth .

# Default command to run
CMD ["python", "src/predict.py"]