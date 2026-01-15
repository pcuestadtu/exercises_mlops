# Base image
FROM python:3.12-slim
#Install some essentials
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*
# Copy files
COPY requirements.txt requirements.txt
COPY src/ src/
COPY data/ data/
# Install python libs
WORKDIR /
RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements.txt --no-cache-dir
# Name our training script as the entrypoint for our Docker image
ENTRYPOINT ["python", "-u", "src/my_model/train.py"]
