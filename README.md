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
- резидентный runtime дважды синтезировал одну реплику в одном процессе и
  получил одинаковый WAV. При втором запросе первый PCM-чанк был готов через
  0.231 секунды после получения текста;
- потоковый интерфейс выдаёт mono PCM `s16le` по 1920 сэмплов, то есть один
  3840-байтовый чанк каждые 80 мс звука.

Эти числа описывают конкретную конфигурацию и не являются универсальным
бенчмарком VoXtream2-RU.

## Ограничения

- Голос и prompt cache готовятся заранее.
- Полная загрузка резидентного runtime занимает около 14.5 секунды. Она
  выполняется один раз при старте устройства; повторный запрос не загружает
  RUAccent и TensorRT engines заново.
- Резидентный процесс занимает около 3.50 GiB RSS вместо 2.30 GiB у
  одноразового запуска: RUAccent остаётся в памяти между репликами.
- Русская нормализация пока ошибается в некоторых падежах, версиях программ,
  телефонных номерах и единицах измерения.
- Текущий resident-протокол обрабатывает запросы последовательно. Отмена
  реплики, несколько клиентов и готовый USB audio sink ещё не реализованы.
- Sink-attention rebuild после позиции 624 реализован по upstream-политике
  prompt + recent tail. Crossing 624→625 детерминирован, но текущий q=1 replay
  занимает около 7.4 секунды; для длинного live-аудио нужен batched-prefill
  TensorRT engine и отдельный слуховой gate после границы.

## Структура

- `src/voxtream2_ru_jetson/` — рабочий PyTorch-less runtime;
- `native/cuda/` — CUDA kernels горячего цикла;
- `tools/` — экспорт assets, правка ONNX и сборка TensorRT engines;
- `experiments/tts/` — диагностические программы и исторические проверки;
- `docker/voxtream2-ru/` — воспроизводимое Jetson-окружение;
- `docs/architecture.md` — границы текущего результата;
- `docs/development.md` — цикл экспорта, сборки и проверки.
- `docs/resident-runtime.md` — PCM API и протокол резидентного процесса.

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
