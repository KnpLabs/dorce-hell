FROM tensorflow/tensorflow:latest-py3

RUN pip install \
    keras \
    tensorflow-hub \
    tensorflow-datasets \
    lxml \
    requests \
    googletrans

WORKDIR /usr/src/app

COPY . /usr/src/app

CMD ["python", "src/main.py"]
