FROM python:3.11-slim AS backend

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend code
COPY backend/ backend/
COPY ai_strategist.py .
COPY enhanced_flip_finder.py .
COPY flip_finder.py .
COPY flip_predictor.py .
COPY quant_analyzer.py .
COPY user_config.py .

EXPOSE 8001

CMD ["python", "-c", "import os; port = os.environ.get('PORT', '8001'); os.execvp('uvicorn', ['uvicorn', 'backend.app:app', '--host', '0.0.0.0', '--port', port])"]
