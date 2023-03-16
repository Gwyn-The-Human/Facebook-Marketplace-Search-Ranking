FROM python:3.8

WORKDIR /app

RUN apt-get update
RUN apt-get install \
    'ffmpeg'\
    'libsm6'\
    'libxext6'  -y
RUN pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .

RUN pip3 install -r requirements.txt

COPY . .

EXPOSE 8080

CMD ["python3", "api.py"]


#dont use docker compose, just use docker build. can come back to back.