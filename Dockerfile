FROM python:3.11-slim-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
COPY cv/ cv/
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir -e cv/

COPY . .
ENV POLYFUT_DATA_DIR=/data
ENV OMP_NUM_THREADS=2
EXPOSE 5000

CMD ["python", "server.py"]
