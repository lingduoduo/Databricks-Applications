FROM public.ecr.aws/lts/ubuntu:20.04

# Defining some variables used at build time to install Python3
ARG PYTHON=python3
ARG PYTHON_PIP=python3-pip
ARG PIP=pip3
ARG PYTHON_VERSION=3.7.10

# Install some handful libraries like curl, wget, git, build-essential, zlib
RUN apt-get update && apt-get install -y --no-install-recommends software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa -y && \
    apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
        wget \
        git \
        libopencv-dev \
        openssh-client \
        openssh-server \
        vim \
        zlib1g-dev && \
    rm -rf /var/lib/apt/lists/*

# Installing Python3

RUN apt-get update \
   && apt-get install -y --no-install-recommends \
   libbz2-dev \
   libc6-dev \
   libffi-dev \
   libgdbm-dev \
   liblzma-dev \
   libncursesw5-dev \
   libreadline-gplv2-dev \
   libsqlite3-dev \
   libssl-dev \
   tk-dev \
   ffmpeg \
   libsm6 \
   libxext6 \
   && rm -rf /var/lib/apt/lists/* \
   && apt-get clean

RUN wget https://www.python.org/ftp/python/$PYTHON_VERSION/Python-$PYTHON_VERSION.tgz \
   && tar -xvf Python-$PYTHON_VERSION.tgz \
   && cd Python-$PYTHON_VERSION \
   && ./configure && make && make install \
   && rm -rf ../Python-$PYTHON_VERSION*

RUN ${PIP} --no-cache-dir install --upgrade pip

RUN ln -s $(which ${PYTHON}) /usr/local/bin/python \
   && ln -s $(which ${PIP}) /usr/bin/pip

WORKDIR /

# Installing numpy, pandas, scikit-learn, scipy
RUN ${PIP} install --no-cache --upgrade \
numpy==1.19.2 \
pandas==0.25.3 \
scikit-learn==0.24.0

# Setting some environment variables.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/local/lib" \
    PYTHONIOENCODING=UTF-8 \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

RUN ${PIP} install --no-cache --upgrade \
    sagemaker-training

# Copies code under /opt/ml/code where sagemaker-containers expects to find the script to run
COPY train.py /opt/ml/code/

# Defines train.py as script entry point
ENV SAGEMAKER_PROGRAM train.py