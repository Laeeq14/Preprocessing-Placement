FROM python:3.11-slim

# System deps for image processing and network tools
RUN apt-get update && apt-get install -y --no-install-recommends \
        iproute2 \
        iputils-ping \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
RUN pip install --no-cache-dir \
    fastapi==0.111.0 \
    uvicorn[standard]==0.29.0 \
    pillow==10.3.0 \
    torch==2.3.0+cpu \
    torchvision==0.18.0+cpu \
    requests==2.32.3 \
    --extra-index-url https://download.pytorch.org/whl/cpu

COPY source_node.py .

# Shared policy will be mounted at /shared at runtime
ENV DEST_URL=http://destination:8001
ENV LOG_LEVEL=INFO
ENV SWITCHING_MARGIN=0.05
ENV EMA_ALPHA=0.3

EXPOSE 8000

CMD ["python", "-u", "source_node.py"]
