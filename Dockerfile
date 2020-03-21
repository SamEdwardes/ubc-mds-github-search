FROM python:3.7

COPY requirements.txt /
RUN pip install -r /requirements.txt

RUN mkdir /ubc-github-search
WORKDIR /ubc-github-search
COPY ./ ./

CMD ["python", "app.py"]