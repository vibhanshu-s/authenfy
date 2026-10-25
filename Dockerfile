FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    AUTHENFY_FUSION_DIR=/app/artifacts/efficientnet-v1 \
    AUTHENFY_DATABASE=/app/instance/receipts.sqlite3 \
    AUTHENFY_SIGNING_KEY_PATH=/app/instance/receipts.key \
    AUTHENFY_UPLOAD_FOLDER=/app/uploads

RUN apt-get update && apt-get install -y --no-install-recommends libglib2.0-0 libgl1 libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --system --gid 10001 authenfy \
    && useradd --system --uid 10001 --gid authenfy --no-create-home authenfy \
    && mkdir -p /app/instance /app/uploads \
    && chown -R authenfy:authenfy /app/instance /app/uploads

WORKDIR /app
COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY --chown=authenfy:authenfy app.py ./
COPY --chown=authenfy:authenfy authenfy ./authenfy
COPY --chown=authenfy:authenfy routes ./routes
COPY --chown=authenfy:authenfy templates ./templates
COPY --chown=authenfy:authenfy static ./static
COPY --chown=authenfy:authenfy reports/v1-evidence ./reports/v1-evidence

USER authenfy
EXPOSE 5000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:5000/health', timeout=4)"
CMD ["waitress-serve", "--listen=0.0.0.0:5000", "--threads=4", "app:app"]
