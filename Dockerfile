# Use a lightweight Python image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy project files to the working directory
COPY . /app

# Install application dependencies
RUN pip install --no-cache-dir -r requirements_inference.txt

# Install DVC with S3 support
RUN pip install --no-cache-dir "dvc[s3]"

# Initialize DVC without Git (no SCM)
RUN dvc init -f --no-scm

# Configure DVC to use the S3 remote storage
RUN dvc remote add -d storage s3://cola-classification/dvc-files

# Expose the port your application runs on
EXPOSE 8000

# Command to pull the model and start the application
CMD ["sh", "-c", "dvc pull dvcfiles/trained_model.dvc && uvicorn app:app --host 0.0.0.0 --port 8000"]