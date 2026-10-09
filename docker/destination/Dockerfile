FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        iproute2 \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --no-cache-dir \
    fastapi==0.111.0 \
    uvicorn[standard]==0.29.0 \
    pillow==10.3.0 \
    torch==2.3.0+cpu \
    torchvision==0.18.0+cpu \
    --extra-index-url https://download.pytorch.org/whl/cpu

COPY destination_node.py .

ENV LOG_LEVEL=INFO

EXPOSE 8001

CMD ["python", "-u", "destination_node.py"]
