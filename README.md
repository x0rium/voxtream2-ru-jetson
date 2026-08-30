# VoXtream2-RU for NVIDIA Jetson

Исследовательский PyTorch-less runtime для запуска
[VoXtream2-RU](https://huggingface.co/simba9/voxtream2-ru) на NVIDIA Jetson.
Текстовый frontend работает на CPU и ONNX Runtime, а нейросетевой тракт — в
BF16 через TensorRT и CUDA.

Репозиторий содержит runtime, CUDA kernels, инструменты экспорта и сборки,
а также диагностические эксперименты. Веса модели, голосовой prompt,
TensorRT engines и сгенерированный звук не публикуются.

Это исследовательский прототип, а не готовый TTS-сервис. TensorRT plans нужно
собирать на целевом Jetson: готовый engine с RTX/x86 нельзя перенести на
Jetson/ARM.

## Тракт синтеза

```text
русский текст
    │
    ├─ ru-normalizr: числа, даты, единицы и сокращения
    ├─ RUAccent ONNX: ударения и омографы
    └─ espeak-ng: фонемы
             │
             ▼
     phone encoder (TensorRT BF16)
             │
             ▼
     temp_former (TensorRT BF16 + CUDA Graph)
             │
             ▼
     dep_former (TensorRT BF16 + CUDA Graph)
             │
             ▼
     Mimi decoder (TensorRT BF16)
             │
             ▼
        PCM 24 kHz mono
```

Процесс синтеза использует TensorRT, CUDA, NumPy и ONNX Runtime, но не
импортирует PyTorch. Офлайн-инструменты экспорта могут использовать PyTorch.

## Что проверено

Прототип запускался на Jetson Orin Nano 8 GB с JetPack 6.2.1:

- полный runtime завершает генерацию при `torch_imported=false`;
- phone encoder, `temp_former`, `dep_former` и Mimi исполняются через TensorRT;
- CUDA Graph сохранил принятую траекторию генерации и побитовое совпадение WAV
  в проведённой проверке;
- CUDA lookup для audio embeddings побитово совпал с PyTorch-эталоном;
- проверочная фраза с датой, временем и процентами дала 0 неизвестных фонем;
- для неё получены RTF 0.926, TTFA ядра 0.271 секунды и peak RSS 2.30 GiB.
- torch-free deployment image занимает 3.27 GB вместо 15.12 GB у прежнего
  PyTorch-base; две контрольные реплики дали побитово прежние WAV.

Эти числа описывают конкретную конфигурацию и не являются универсальным
бенчмарком VoXtream2-RU.

## Ограничения

- Голос и prompt cache готовятся заранее.
- Cold start RUAccent занимает около 13 секунд; для живого диалога нужен
  резидентный процесс.
- Русская нормализация пока ошибается в некоторых падежах, версиях программ,
  телефонных номерах и единицах измерения.
- Генератор работает по кадрам, но публичный API пока сохраняет итоговый WAV;
  выдача PCM-чанков потребителю ещё не оформлена.
- Sink-attention compaction после позиции 624 не реализован.

## Структура

- `src/voxtream2_ru_jetson/` — рабочий PyTorch-less runtime;
- `native/cuda/` — CUDA kernels горячего цикла;
- `tools/` — экспорт assets, правка ONNX и сборка TensorRT engines;
- `experiments/tts/` — диагностические программы и исторические проверки;
- `docker/voxtream2-ru/` — воспроизводимое Jetson-окружение;
- `docs/architecture.md` — границы текущего результата;
- `docs/development.md` — цикл экспорта, сборки и проверки.

## Быстрая проверка

Проверки, не требующие Jetson:

```bash
python3 -m compileall -q src tools experiments tests
PYTHONPATH=src python3 -m unittest discover -s tests -v
ruff check src tools experiments tests
```

Параметры runtime в подготовленном Jetson-окружении:

```bash
PYTHONPATH=src python3 -m voxtream2_ru_jetson --help
```

Пример запуска после подготовки engines и assets находится в
[`docs/development.md`](docs/development.md).

## Лицензии

Код этого репозитория опубликован под MIT. Сторонние компоненты и веса модели
сохраняют собственные лицензии; подробности перечислены в [`NOTICE.md`](NOTICE.md).
