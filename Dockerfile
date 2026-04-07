# Use Python 3.11
FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# Install system dependencies
# We replaced libgl1-mesa-glx with libgl1 and libglx-mesa0
RUN apt-get update && apt-get install -y \
    freeglut3-dev \
    libgl1-mesa-dev \
    libglu1-mesa-dev \
    libgl1 \
    libglx-mesa0 \
    && rm -rf /var/lib/apt/lists/*

# Copy all files
COPY . .

# Install Python libraries
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 7860
# Run the inference script
CMD ["python", "inference.py"]
