FROM ghcr.io/osgeo/gdal:ubuntu-small-3.8.4

RUN apt-get update && apt-get install -y \
    python3.11 python3.11-dev python3-pip \
    libexpat1 libgeos-dev libproj-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN python3.11 -m pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
#rebuild