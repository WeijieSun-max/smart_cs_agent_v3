FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get -o Acquire::Retries=5 update \
    && apt-get -o Acquire::Retries=5 install --yes --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 10001 app \
    && useradd --uid 10001 --gid app --create-home --shell /usr/sbin/nologin app

COPY requirements.txt ./requirements.txt
RUN python -m pip install --upgrade pip \
    && python -m pip install --requirement requirements.txt

COPY --chown=app:app . .
RUN mkdir -p /app/logs /app/data/vector_store \
    && chown -R app:app /app/logs /app/data

USER app

EXPOSE 8000
STOPSIGNAL SIGTERM

CMD ["python", "main.py"]
