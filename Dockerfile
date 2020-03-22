# to build:
# docker image build -t ubc_github_search:latest .

# to run:
# docker run -d --rm -p 5000:5000 ubc_github_search

# to run interactively:
# docker run -it --rm -p 5000:5000 ubc_github_search bash

FROM python:3.7

COPY requirements.txt /
RUN pip install -r /requirements.txt

# get NLTK data
RUN mkdir /usr/local/nltk_data
RUN python -m nltk.downloader -d /usr/local/nltk_data stopwords
RUN python -m textblob.download_corpora

RUN apt-get update
RUN apt-get install tree
RUN apt-get -y install vim

RUN mkdir /ubc-github-search
WORKDIR /ubc-github-search
COPY ./ ./

CMD ["python", "app.py"]