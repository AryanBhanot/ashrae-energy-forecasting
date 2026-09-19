FROM python:3-slim

ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["sh", "-c", "streamlit run apps/streamlit/app.py --server.port=${PORT:-8080} --server.address=0.0.0.0"]
