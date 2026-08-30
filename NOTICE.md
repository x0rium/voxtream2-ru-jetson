# Third-party components

This repository contains integration code and experiments around projects that
retain their own licenses:

- [VoXtream2](https://github.com/herimor/voxtream) code: MIT.
- The Depth Transformer component used by VoXtream2: Apache-2.0; see the
  upstream `LICENSE-APACHE` and `NOTICE` files.
- [VoXtream2-RU](https://huggingface.co/simba9/voxtream2-ru) model weights:
  OpenRAIL-M, with the restrictions and attribution requirements stated in
  the model repository.
- Mimi/Moshi, RUAccent, ru-normalizr, TensorRT, CUDA and other dependencies:
  their respective upstream licenses.
- The deployment image is based on NVIDIA
  `l4t-tensorrt:r10.3.0-runtime` and is subject to the NVIDIA Deep Learning
  Container License included in that image.

Model weights, TensorRT engines, voice prompts and generated audio are not
distributed in this Git repository.
