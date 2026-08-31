# VoXtream2-RU for NVIDIA Jetson

Исследовательский PyTorch-less runtime для запуска
[VoXtream2-RU](https://huggingface.co/simba9/voxtream2-ru) на NVIDIA Jetson.
Текстовый frontend работает на CPU и ONNX Runtime, а нейросетевой тракт — в
BF16 через TensorRT и CUDA.

Репозиторий содержит runtime, CUDA kernels, инструменты экспорта и сборки,
а также диагностические эксперименты. Исходный checkpoint, голосовой prompt,
TensorRT engines и сгенерированный звук здесь не хранятся.

Переносимые ONNX и их manifest публикуются отдельно в
[`x0rium/voxtream2-ru-jetson-onnx@v0.2.1`](https://huggingface.co/x0rium/voxtream2-ru-jetson-onnx/tree/v0.2.1).

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

Прототип запускался на Jetson Orin Nano 8 GB с JetPack 6.2.3 и Jetson Linux
R36.5.2:

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
- один dynamic `temp_former` plan содержит q=1 hot path и q=420 sink replay с
  общими весами. Replay занимает 0.112 секунды вместо 7.317, а RTF длинной
  реплики снизился с 1.060 до 0.852. В слепом A/B различий не услышали;
- `voxtream2-ru-jetson-playback` передаёт PCM в ALSA до завершения генерации.
  End-to-end smoke передал 75 чанков, первый появился через 0.417 секунды.

Эти числа описывают конкретную конфигурацию и не являются универсальным
бенчмарком VoXtream2-RU.

## Ограничения

- Голос и prompt cache готовятся заранее.
- Полная загрузка резидентного runtime занимает около 14.5 секунды. Она
  выполняется один раз при старте устройства; повторный запрос не загружает
  RUAccent и TensorRT engines заново.
- Резидентный процесс занимает около 3.50 GiB RSS вместо 2.30 GiB у
  одноразового запуска. Unified q=420 context добавляет около 70 MiB
  относительно q=1-контроля, не дублируя temporal weights.
- Русская нормализация пока ошибается в некоторых падежах, версиях программ,
  телефонных номерах и единицах измерения.
- Текущий resident-протокол обрабатывает запросы последовательно. PCM уже
  передаётся в ALSA потоково; отмена реплики, несколько клиентов и проверка с
  физической USB-колонкой ещё не реализованы.
- Sink-attention rebuild после позиции 624 реализован по upstream-политике
  prompt + recent tail. Для текущего prompt длиной 108 кадров q=420 profile
  восстанавливает prompt и последние 312 temporal-входов. Если в едином plan
  нет точного профиля q=420, runtime останавливается при запуске и не выдаёт
  частично корректный длинный звук.

## Структура

- `src/voxtream2_ru_jetson/` — рабочий PyTorch-less runtime;
- `native/cuda/` — CUDA kernels горячего цикла;
- `tools/` — экспорт assets, правка ONNX и сборка TensorRT engines;
- `experiments/tts/` — диагностические программы и исторические проверки;
- `docker/voxtream2-ru/` — воспроизводимое Jetson-окружение;
- `docs/architecture.md` — границы текущего результата;
- `docs/development.md` — цикл экспорта, сборки и проверки.
- `docs/jetson-install.md` — установка опубликованного релиза с чистого
  checkout;
- `docs/resident-runtime.md` — PCM API и протокол резидентного процесса.

## Быстрая проверка

Проверки, не требующие Jetson:

```bash
python3 -m compileall -q src tools experiments tests
python3 -m pytest -q
ruff check src tools experiments tests
```

Параметры runtime в подготовленном Jetson-окружении:

```bash
PYTHONPATH=src python3 -m voxtream2_ru_jetson --help
```

Полная установка, сборка всех четырёх engines и первый запуск находятся в
[`docs/jetson-install.md`](docs/jetson-install.md).

## Лицензии

Код этого репозитория опубликован под MIT. Сторонние компоненты и веса модели
сохраняют собственные лицензии; подробности перечислены в [`NOTICE.md`](NOTICE.md).
