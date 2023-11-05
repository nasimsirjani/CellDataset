FROM python:3.8-slim-buster

# Increase shared memory limit
# Set kernel parameters using echo
RUN echo 'kernel.shmmax = 134217728' > /etc/sysctl.conf

# Apply the changes
RUN sysctl -p

# Set kernel parameters using echo
RUN echo 'kernel.shmall = 134217728' > /etc/sysctl.conf

# Apply the changes
RUN sysctl -p

RUN apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0

COPY src/requirements.txt /requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

WORKDIR /app

COPY . /app





