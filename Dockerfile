# app/Dockerfile

FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    python3-venv \
    libcairo-dev \
    && rm -rf /var/lib/apt/lists/*

ENV PATH="${PATH}:/root/.local/bin"

RUN python -m ensurepip
RUN python -m pip install librelane
RUN python3 -m venv .venv
RUN .venv/bin/pip install loguru librelane streamlit gdstk cairosvg

ADD ./app.py /app

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT [".venv/bin/streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
