FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Створюємо директорії для даних
RUN mkdir -p data results

ENTRYPOINT ["python"]
