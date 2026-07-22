FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# Install system dependencies for OpenCV and other packages
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    git \
    g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install python packages
COPY requirements.txt requirements-cli.txt ./
RUN pip install --no-cache-dir -r requirements.txt -r requirements-cli.txt
RUN pip install --no-cache-dir fastapi uvicorn requests pydantic httpx

# Copy engine codebase
COPY . .

EXPOSE 8001

CMD ["python", "engine_api.py"]
