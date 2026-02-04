FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN useradd -m -u 10001 ultron && mkdir -p /app/data && chown -R ultron:ultron /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY ultronpro /app/ultronpro
COPY ui /app/ui

USER ultron
EXPOSE 8000

CMD ["uvicorn", "ultronpro.main:app", "--host", "0.0.0.0", "--port", "8000"]
