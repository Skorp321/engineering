FROM python:3.10


WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip
RUN apt-get update \
    && apt-get install -y libgl1-mesa-glx
RUN pip install --no-cache-dir -r requirements.txt

COPY save_video.py /app

ENTRYPOINT [ "python3", "save_video.py" ]
CMD ["arg1", "arg2"]