FROM python:3.11

WORKDIR /app

COPY . /app

RUN apt-get update && apt-get install -y gdal-bin libgdal-dev
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
