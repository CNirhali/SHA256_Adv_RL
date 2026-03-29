FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src /app/src

# Set environment variables for AWS
ENV AWS_DEFAULT_REGION=us-east-1

# Run the evaluator entrypoint
CMD ["python", "-m", "src.cloud.sagemaker_entrypoint"]
