FROM python:3.9-slim-bullseye

# Install system dependencies for pygame/gymnasium
RUN apt-get update && apt-get install -y --fix-missing \
    libgl1-mesa-glx \
    libglib2.0-0 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Upgrade pip to ensure latest wheel tags are supported
RUN pip install --no-cache-dir --upgrade pip

# Explicitly construct the CPU-only PyTorch environment
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all other sources
COPY . .

# Default command can be overridden, but run train.py mostly
CMD ["python", "train.py"]
