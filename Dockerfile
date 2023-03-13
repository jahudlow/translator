FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .

RUN apt-get update && apt-get install -y \
    gcc libpq-dev python3-dev && \
    rm -rf /var/lib/apt/lists/* && \
    pip install -r requirements.txt

COPY iso-639-3.tab /app/iso-639-3.tab
COPY app.py /app/app.py
COPY startup.sh /app/startup.sh

EXPOSE 8501

WORKDIR /app

CMD ["./startup.sh"]