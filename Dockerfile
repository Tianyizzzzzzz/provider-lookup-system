FROM python:3.11-slim

WORKDIR /app

COPY provider_lookup_web/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY provider_lookup_web/ .

EXPOSE 8000

CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]
