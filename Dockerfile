# Serving image: the production model exported from the registry, the API, and the solver as fallback.
# Build after the pipeline has written serving/bundle (python -m cooling_surrogate.pipeline ...).
FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    COOLING_BUNDLE=/app/serving/bundle \
    COOLING_TORCH_THREADS=1
WORKDIR /app
COPY requirements-serve.txt ./
RUN pip install --no-cache-dir -r requirements-serve.txt
COPY cooling_surrogate ./cooling_surrogate
COPY serving/bundle ./serving/bundle
RUN useradd --system --no-create-home app
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz')"
CMD ["uvicorn", "--factory", "cooling_surrogate.service:create_app", "--host", "0.0.0.0", "--port", "8000"]
