# Stage 1: build the frontend to static files
FROM node:22-slim AS web
WORKDIR /web
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY frontend/ ./
RUN npm run build

# Stage 2: the Python app (API + the built site + the pipeline/ML code)
FROM python:3.11-slim AS app
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /app
# libgomp is the OpenMP runtime LightGBM needs
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
COPY requirements.txt ./
RUN pip install -r requirements.txt
COPY pipeline ./pipeline
COPY ml ./ml
COPY backend ./backend
COPY --from=web /web/out ./backend/app/static
RUN useradd --create-home app && mkdir -p /app/data && chown -R app /app
USER app
ENV HOST=0.0.0.0 PORT=8000
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s CMD python -c "import os,urllib.request as u; u.urlopen('http://127.0.0.1:%s/api/health' % os.environ.get('PORT','8000'), timeout=4)"
# The in-process scheduler (ENABLE_SCHEDULER, default on) downloads data and trains in the background on
# first start, so the port opens immediately. Docker Compose turns it off and uses init + worker services.
CMD ["sh", "-c", "uvicorn backend.app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
