FROM python:3.8-slim-buster

# Increase shared memory limit
RUN sysctl -w kernel.shmmax=134217728

RUN sysctl -w kernel.shmall=134217728

RUN apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0

COPY src/requirements.txt /requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

WORKDIR /app

COPY . /app





