FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# if you are in mainland china, use aliyun apt source
RUN mv /etc/apt/sources.list /etc/apt/sources.list.bak && \
    echo "deb http://mirrors.aliyun.com/ubuntu/ jammy main restricted universe multiverse" > /etc/apt/sources.list && \
    echo "deb http://mirrors.aliyun.com/ubuntu/ jammy-security main restricted universe multiverse" >> /etc/apt/sources.list && \
    echo "deb http://mirrors.aliyun.com/ubuntu/ jammy-updates main restricted universe multiverse" >> /etc/apt/sources.list && \
    echo "deb http://mirrors.aliyun.com/ubuntu/ jammy-backports main restricted universe multiverse" >> /etc/apt/sources.list

# basic system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    wget \
    curl \
    vim \
    gcc \
    g++ \
    cmake \
    ca-certificates \
    libxml2-dev \
    libboost-all-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# proxy for mainland china, run your proxy on 7890 port outside the container
ENV http_proxy='http://172.17.0.1:7890'
ENV https_proxy='http://172.17.0.1:7890'

# install micromamba (static binary) to /usr/local/bin
RUN curl -L https://micro.mamba.pm/api/micromamba/linux-64/latest | \
    tar -xvj -C /usr/local/bin bin/micromamba --strip-components=1

# configure micromamba root directory and PATH
ENV MAMBA_ROOT_PREFIX=/opt/conda
ENV PATH=/opt/conda/bin:$PATH

# working directory
WORKDIR /workspace/probayes

# copy conda configuration (optional for mainland china) and environment file
COPY probayes_env.yml /tmp/env.yml


# use micromamba to create environment (environment name: probayes)
SHELL ["/bin/bash", "-lc"]
RUN mkdir -p "$MAMBA_ROOT_PREFIX" && \
    micromamba create -y -n probayes -f /tmp/env.yml && \
    micromamba clean -a -y -v

# activate environment and install additional pip dependencies
ENV PATH=/opt/conda/envs/probayes/bin:$PATH
RUN eval "$(micromamba shell hook -s bash)" && micromamba activate probayes && \
    pip config set global.index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple && \
    pip install -v --progress-bar=on --no-cache-dir git+https://github.com/pandegroup/pdbfixer.git && \
    pip install -v --progress-bar=on --no-cache-dir torch-scatter==2.1.2+pt20cu117 -f https://data.pyg.org/whl/torch-2.0.0+cu117.html

# download and install rosetta
RUN curl -O https://graylab.jhu.edu/download/PyRosetta4/archive/release/PyRosetta4.Release.python310.linux.wheel/pyrosetta-2024.35+release.45abd6a-cp310-cp310-linux_x86_64.whl
RUN pip install pyrosetta-2024.35+release.45abd6a-cp310-cp310-linux_x86_64.whl
RUN rm pyrosetta-2024.35+release.45abd6a-cp310-cp310-linux_x86_64.whl

# clone project source code and install in editable mode
RUN git clone https://github.com/GenSI-THUAIR/ProBayes
RUN cd ProBayes && pip install -v --progress-bar=on --no-cache-dir -e .

# move ProBayes to workspace
RUN mv ProBayes /workspace/ProBayes && cd /workspace/ProBayes && rm -rf /workspace/probayes

# auto-activate environment when entering container
RUN micromamba shell init -s bash && \
    echo "micromamba activate probayes" >> /root/.bashrc && \
    echo 'alias conda="micromamba"' >> /root/.bashrc

WORKDIR /workspace/ProBayes
# usage:
# build image: docker build -t probayes:latest . --progress=plain
# run container: docker run --gpus all -it probayes:latest


