FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install . && useradd --create-home --uid 1000 litagent
COPY data ./data

USER litagent
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=4)"

# ANTHROPIC_API_KEY is required at run time; NCBI_API_KEY, NCBI_EMAIL and LITAGENT_API_KEY are optional.
CMD ["uvicorn", "litagent.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
