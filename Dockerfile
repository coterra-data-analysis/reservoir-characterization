# For more information, please refer to https://aka.ms/vscode-docker-python
FROM python:3.8.12 as base
ENV PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100 \
    POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_CREATE=false \
    POETRY_VERSION=1.2.2 \
    PYTHON_INTERPRETER_PATH=/usr/local/bin/python \
    # Keeps Python from generating .pyc files in the container\
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONFAULTHANDLER=1 \
    # Turns off buffering for easier container logging
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=0 \
    # Poetry install location
    PATH="/root/.local/bin:$PATH" 
RUN apt-get update \
    && apt-get --yes install curl
RUN curl -sSL https://install.python-poetry.org | python -
COPY pyproject.toml /
RUN poetry install --no-dev

FROM base as development
ENV ENVIRONMENT=DEV
RUN poetry install
# install this for data viewer extension, and Azure CLI 
RUN apt-get update && apt-get install -y libstdc++6
RUN SNIPPET="export PROMPT_COMMAND='history -a' && export HISTFILE=/commandhistory/.bash_history" \
    && echo $SNIPPET >> "/root/.bashrc"
