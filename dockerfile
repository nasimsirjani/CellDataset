FROM python:3.8-slim-buster

COPY src/requirements.txt /requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

WORKDIR /app

COPY . /app





