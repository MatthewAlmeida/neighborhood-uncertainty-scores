FROM matthewalmeida/tkddnbscorebase:latest

ENV PYTHONUNBUFFERED 1

COPY . /Workspace
WORKDIR /Workspace
