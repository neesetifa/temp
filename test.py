FROM python:3.11-slim

RUN apt-get update \
 && apt-get install -y --no-install-recommends util-linux \
 && rm -rf /var/lib/apt/lists/* \
 && pip install --no-cache-dir numpy==2.1.3 pandas==2.2.3 scipy==1.14.1 scikit-learn==1.5.2 joblib==1.4.2 \
 && useradd --create-home --uid 10001 --shell /bin/sh oeagent \
 && mkdir -p /workspace /output /logs/verifier \
 && chown -R oeagent:oeagent /workspace /output /logs/verifier /home/oeagent

COPY data/train.csv /app/train.csv
RUN chmod 0555 /app && chmod 0444 /app/train.csv
WORKDIR /workspace
ENV HOME=/home/oeagent
USER oeagent
