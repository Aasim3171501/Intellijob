# 1. Base image with Python runtime
FROM python:3.11-slim

# 2. Prevent Python from writing .pyc files and enable real-time log streaming
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080

# 3. Set the active directory inside the container
WORKDIR /app

# 4. Install essential Linux build tools for C-extensions (like HDBSCAN and psycopg2)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 5. Copy requirements first to leverage Docker layer caching
COPY requirements.txt .

# Tell the compiler to treat incompatible pointer types as warnings, not fatal errors
ENV CFLAGS="-Wno-incompatible-pointer-types"

# 6. Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt gunicorn

# 7. Copy the rest of your application code
COPY . .

# 8. Document the container port
EXPOSE 8080

# 9. Start Gunicorn serving Django WSGI
CMD exec gunicorn core.wsgi:application \
    --bind 0.0.0.0:${PORT} \
    --workers 2 \
    --threads 4 \
    --timeout 120