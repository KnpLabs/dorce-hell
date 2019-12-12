FROM tensorflow/tensorflow:latest-py3

RUN pip install \
    keras \
    tensorflow-hub \
    tensorflow-datasets 

WORKDIR /usr/src/app

COPY . /usr/src/app

CMD ["python", "src/main.py"]
