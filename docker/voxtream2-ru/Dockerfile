# Official Jetson deployment image with CUDA 12.6 and TensorRT 10.3.  PyTorch
# is deliberately absent: model export happens offline, outside this image.
ARG BASE_IMAGE=nvcr.io/nvidia/l4t-tensorrt:r10.3.0-runtime
FROM ${BASE_IMAGE}

# The runtime writes PCM with Python's wave module.  Only espeak-ng is needed
# from the former audio/build dependency set.
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    ca-certificates \
    espeak-ng \
    && rm -rf /var/lib/apt/lists/*

COPY docker/voxtream2-ru/requirements.lock /tmp/voxtream2-ru-requirements.lock
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1
RUN python3 -m pip install --no-cache-dir --index-url https://pypi.org/simple \
    -r /tmp/voxtream2-ru-requirements.lock

# TensorRT itself is imported after launch with `--runtime nvidia`, which
# mounts Jetson platform libraries such as libnvdla_compiler.so.
RUN python3 -m pip check \
    && python3 -c "import importlib.util as u; blocked = ('torch', 'torchaudio', 'torchvision', 'torchao', 'torchtune'); assert all(u.find_spec(name) is None for name in blocked); assert u.find_spec('tensorrt') is not None; import numpy, onnxruntime; from cuda import cudart"
