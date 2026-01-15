# 1. Use the same base
FROM python:3.12-slim

# 2. Install same system tools
RUN apt-get update && \
    apt-get install --no-install-recommends -y build-essential gcc && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# 3. Setup workspace
WORKDIR /

# 4. Copy code
# We don't strictly need requirements.txt for this dummy script, 
# but in real life you would COPY and install it here.
COPY src/ src/

# 5. ENTRYPOINT
# We set the entrypoint to the evaluate script.
ENTRYPOINT ["python", "-u", "src/my_model/evaluate.py"]
