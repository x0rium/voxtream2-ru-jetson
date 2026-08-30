# PyTorch 2.7 fixes the Jetson SDPA/CUDA Graph path used by VoXtream's
# single-token autoregressive decode.  The older PyTorch 2.5 image routes that
# shape through a cuDNN attention path which fails during graph capture.
ARG BASE_IMAGE=dustynv/pytorch:2.7-r36.4.0
FROM ${BASE_IMAGE}

# VoXtream2-RU vendors its inference code with the model.  This image only
# supplies the compatible Jetson CUDA runtime and Python/system dependencies.
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    build-essential \
    cargo \
    espeak-ng \
    libasound2-dev \
    libopus-dev \
    libsndfile1 \
    pkg-config \
    rustc \
    && rm -rf /var/lib/apt/lists/*

COPY docker/voxtream2-ru/requirements.txt /tmp/voxtream2-ru-requirements.txt
# Some Jetson base images inject NVIDIA's optional PyPI mirror.  Keep this
# runtime build deterministic and avoid long DNS retries when that mirror is
# unavailable; every dependency below is available from the primary index.
ENV PIP_EXTRA_INDEX_URL=https://pypi.org/simple
RUN python3 -m pip install --no-cache-dir --index-url https://pypi.org/simple \
    -r /tmp/voxtream2-ru-requirements.txt
