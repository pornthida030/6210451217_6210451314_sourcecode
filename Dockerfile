FROM python:3.7.9
RUN apt-get -y update
RUN apt-get -y upgrade
RUN apt-get install -y ffmpeg libavcodec-extra
COPY . .
RUN pip install -r requirements.txt
CMD ["python","wsgi.py"]