# Torch-free runtime image

Образ предназначен только для синтеза. Он не содержит PyTorch, экспортёры
модели или инструменты сборки TensorRT engines.

Основа — официальный Jetson runtime
`nvcr.io/nvidia/l4t-tensorrt:r10.3.0-runtime` с CUDA 12.6 и TensorRT 10.3.

## Сборка

```bash
docker build \
  -t voxtream2-ru-jetson:runtime \
  -f docker/voxtream2-ru/Dockerfile \
  .
```

`.dockerignore` передаёт сборщику только Dockerfile и lock-файл. Модели,
engines, голосовые prompts, WAV и локальные данные в build context не входят.

## Проверка окружения

TensorRT нужно импортировать после запуска с NVIDIA runtime: во время
`docker build` платформа ещё не монтирует `libnvdla_compiler.so`.

```bash
docker run --rm \
  --runtime nvidia \
  --network none \
  --entrypoint python3 \
  voxtream2-ru-jetson:runtime \
  -c 'import importlib.util as u, sys, tensorrt; from cuda import cudart; assert u.find_spec("torch") is None; assert "torch" not in sys.modules; print(tensorrt.__version__)'
```

На проверенной сборке полный text-to-WAV regression дважды совпал со старым
окружением побитово. Размер образа снизился с 15.12 до 3.27 GB. Эти значения
относятся к конкретной ARM64-сборке для JetPack 6.2.3 / Jetson Linux R36.5.2.
