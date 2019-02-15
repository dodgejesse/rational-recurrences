FROM cupy/cupy:v5.2.0-python3

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
ENV PATH /usr/local/nvidia/bin/:$PATH

ENV LD_LIBRARY_PATH /usr/local/nvidia/lib64

WORKDIR /stage

# Install base packages.
RUN apt-get update --fix-missing && apt-get install -y \
    bzip2 \
    ca-certificates \
    curl \
    gcc \
    git \
    libc-dev \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    wget \
    libevent-dev && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install --upgrade pip

RUN pip3 install --no-cache-dir -q http://download.pytorch.org/whl/cu92/torch-0.4.1-cp35-cp35m-linux_x86_64.whl
RUN pip3 install torchvision

COPY requirements.txt requirements.txt

RUN pip3 install -r requirements.txt

RUN git clone https://github.com/dodgejesse/rational-recurrences

WORKDIR rational-recurrences/

COPY classification/train_classifier.py classification/train_classifier.py
COPY classification/run_local_experiment.py classification/run_local_experiment.py
COPY classification/save_learned_structure.py classification/save_learned_structure.py
COPY language_model/train_lm.py language_model/train_lm.py

RUN echo "hab"
RUN pip3 freeze

#RUN cat classification/experiment_params.py

CMD ["python3", "-u", "classification/run_local_experiment.py", "-d", "/data/", "-a", "original_mix", "--logging_dir", "/output/logging/", "--model_save_dir", "/output"]