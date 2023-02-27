# Edit the base image here, e.g., to use 
# TENSORFLOW (https://hub.docker.com/r/tensorflow/tensorflow/) 
# or a PYTORCH (https://hub.docker.com/r/pytorch/pytorch/) base image

# FROM python:3.9.2
# FROM pytorch/pytorch
FROM pytorch/pytorch

# Needed for opencv
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN apt-get install -y git

RUN apt-get update -y

RUN apt install libgl1-mesa-glx -y





RUN groupadd -r algorithm && useradd -m --no-log-init -r -g algorithm algorithm


RUN mkdir -p /opt/algorithm /input /output \
    && chown algorithm:algorithm /opt/algorithm /input /output

USER algorithm

WORKDIR /opt/algorithm

ENV PATH="/home/algorithm/.local/bin:${PATH}"

RUN python -m pip install --user -U pip
# RUN python -m pip install pip

# Install required packages
# e.g. `python -m pip install sklearn==...`

COPY --chown=algorithm:algorithm requirements.txt /opt/algorithm/
COPY --chown=algorithm:algorithm requirements_yolo.txt /opt/algorithm/
RUN python -m pip install --user --upgrade pip
RUN python -m pip install --user -rrequirements.txt
RUN python -m pip install --user -rrequirements_yolo.txt
RUN python -m pip show imagecodecs

COPY --chown=algorithm:algorithm process.py /opt/algorithm/
COPY --chown=algorithm:algorithm weights_full_image.pth /opt/algorithm/weights_full_image.pth
COPY --chown=algorithm:algorithm weights_cropped_image.pth /opt/algorithm/weights_cropped_image.pth
COPY --chown=algorithm:algorithm best.pt /opt/algorithm/best.pt
COPY --chown=algorithm:algorithm classifier_models /opt/algorithm/classifier_models
COPY --chown=algorithm:algorithm yolov5_1 /opt/algorithm/yolov5_1 
COPY --chown=algorithm:algorithm weights /opt/algorithm/weights

# Copy additional files, such as model weights
# e.g. `COPY --chown=algorithm:algorithm weights.pth /opt/algorithm/weights.pth`

ENTRYPOINT python -m process $0 $@

## ALGORITHM LABELS ##

# These labels are required
LABEL nl.diagnijmegen.rse.algorithm.name=airogs_algorithm

# These labels are required and describe what kind of hardware your algorithm requires to run.
LABEL nl.diagnijmegen.rse.algorithm.hardware.cpu.count=1
LABEL nl.diagnijmegen.rse.algorithm.hardware.cpu.capabilities=()
LABEL nl.diagnijmegen.rse.algorithm.hardware.memory=1G
LABEL nl.diagnijmegen.rse.algorithm.hardware.gpu.count=0
LABEL nl.diagnijmegen.rse.algorithm.hardware.gpu.cuda_compute_capability=
LABEL nl.diagnijmegen.rse.algorithm.hardware.gpu.memory=


