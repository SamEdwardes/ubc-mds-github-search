# to build:
# docker image build -t ubc_github_search:latest .

# to run:
# docker run -d --rm -p 5000:5000 ubc_github_search

# to run interactively:
# docker run -it --rm ubc_github_search bash

FROM python:3.7

COPY requirements.txt /
RUN pip install -r /requirements.txt

RUN mkdir /ubc-github-search
WORKDIR /ubc-github-search
COPY ./ ./

RUN apt-get update
RUN apt-get install tree

CMD ["python", "app.py"]