# Use Python 3.11
FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# Install system dependencies for OpenGL and Mesa (Required for HighwayEnv/Gymnasium)
RUN apt-get update && apt-get install -y \
    freeglut3-dev \
    libgl1-mesa-dev \
    libglu1-mesa-dev \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Copy all files (including inference.py and highway_brain.pth) to the container
COPY . .

# Install Python libraries from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# IMPORTANT: Meta OpenEnv expects the entry point to be the inference script.
# We no longer run Streamlit as the main command.
CMD ["python", "inference.py"]