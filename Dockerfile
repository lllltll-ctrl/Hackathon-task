FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Створюємо директорії для даних та користувача
RUN mkdir -p data results && \
    useradd --create-home appuser && \
    chown -R appuser:appuser /app
USER appuser

ENTRYPOINT ["python"]
