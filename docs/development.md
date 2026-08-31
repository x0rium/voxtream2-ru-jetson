# Цикл разработки

Работа разделена на два процесса. Офлайн-конвертеры могут использовать PyTorch;
процесс синтеза — нет. Такое разделение делает результат проверяемым и не
заставляет держать тяжёлый framework на устройстве во время разговора.

## 1. Экспорт

`tools/` извлекает веса и fixed-prompt state, строит ONNX и сохраняет
framework-neutral assets. Экспорт запускается только после изменения модели,
графа или голоса.

## 2. Сборка TensorRT

Engines собираются на целевом Jetson. План, собранный на RTX 4090/x86, нельзя
перенести на Jetson/ARM как готовый бинарник.

```bash
python3 tools/build_tensorrt_engine.py --help
```

Основной `temp_former` экспортируется одним dynamic ONNX с диапазоном q=1..420.
Для воспроизводимого ABI cache используются pins из
`experiments/tts/requirements-export.txt`: более новые версии torchtune меняют
число KV heads и несовместимы с проверенным runtime ABI.

```bash
PYTHONPATH=experiments/tts python3 \
  experiments/tts/voxtream_tensorrt_temp_unified_probe.py \
  --checkpoint /path/to/model.safetensors \
  --output-dir /path/to/export

python3 tools/patch_onnx_append_sem_head.py \
  --input /path/to/export/temp-unified-dynamic-q1-q420.onnx \
  --output /path/to/export/temp-unified-q1-q420-sem-head.onnx \
  --checkpoint /path/to/model.safetensors
```

Готовый ONNX нужно перенести на Jetson и собрать там:

```bash
docker run --rm --runtime nvidia --network none \
  -v /path/to/worktree:/work -w /work \
  voxtream2-ru-jetson:runtime \
  python3 tools/build_tensorrt_engine.py \
  --onnx artifacts/temp-unified-q1-q420-sem-head.onnx \
  --engine artifacts/temp-unified-q1-q420.engine \
  --sequence-profiles 1 420 \
  --workspace-mib 512 \
  --optimization-level 1 \
  --metrics artifacts/temp-unified-q1-q420-build.json
```

Engine нужно собирать внутри того же runtime-образа, в котором он будет
запускаться. Даже patch-сборки TensorRT `10.3.0.26` и `10.3.0.30` создают
несовместимые plan-файлы. Короткая строка `tensorrt.__version__ == "10.3.0"`
этого различия не показывает; точную версию проверяют через `dpkg-query`.

Если длина prompt изменится, второй TensorRT profile тоже нужно изменить на
сумму длины prompt и 312 сохранённых входов. Runtime проверяет наличие точного
profile и ABI state-буферов при запуске.

## 3. Runtime-образ без PyTorch

Готовые engines запускаются в отдельном deployment-окружении. PyTorch там не
установлен и нужен только офлайн-экспортёрам из предыдущего шага.

```bash
docker build \
  -t voxtream2-ru-jetson:runtime \
  -f docker/voxtream2-ru/Dockerfile \
  .
```

Полная инструкция и проверка отсутствия PyTorch находятся в
[`docker/voxtream2-ru/README.md`](../docker/voxtream2-ru/README.md).

## 4. Быстрая проверка кода

```bash
python3 -m compileall -q src tools experiments
PYTHONPATH=src python3 -m voxtream2_ru_jetson --help
```

Вторая команда требует Jetson-окружение с TensorRT и CUDA Python.

## 5. Генерация

После подготовки engines и assets runtime запускается как модуль:

```bash
PYTHONPATH=src python3 -m voxtream2_ru_jetson \
  --assets /path/to/prompt-assets.json \
  --text "Сегодня 30 августа 2026 года, время 21:45." \
  --ruaccent-assets /path/to/ruaccent \
  --phone-map /path/to/phoneme_to_token.json \
  --temp-engine /path/to/temp-unified-q1-q420.engine \
  --dep-engine /path/to/dep.engine \
  --phone-engine /path/to/phone.engine \
  --mimi-engine /path/to/mimi.engine \
  --mimi-state /path/to/mimi-state.json \
  --audio-embedding-weight /path/to/audio_embeddings.bf16 \
  --audio-embedding-cubin /path/to/audio_embedding.cubin \
  --cuda-acoustic-control-cubin /path/to/acoustic_control.cubin \
  --cuda-dep-graph --cuda-temp-graph \
  --output output.wav
```

## 6. Проверки перед принятием изменения

1. Процесс не импортировал `torch`.
2. ONNX/TensorRT прошли численное или побитовое сравнение с эталоном там, где
   оно возможно.
3. Не появились неизвестные фонемы.
4. RTF, TTFA и peak RSS записаны до и после изменения.
5. Изменение, способное повлиять на звук, прошло слепое или хотя бы
   рандомизированное A/B-прослушивание.

Разовые probes после завершения исследования остаются в `experiments/`, а
выводы переносятся в документацию. Так не приходится угадывать, какой из
десятков файлов является рабочим runtime.

План публикации переносимых ONNX находится в
[`huggingface-release.md`](huggingface-release.md).

## 7. Резидентный запуск

Для диалога создайте один `SynthesisRuntime` и вызывайте
`synthesize_stream()` для каждой реплики. Runtime сбрасывает состояние
генераторов, но не выгружает RUAccent, TensorRT engines и CUDA Graph.

Готовый stdio-процесс запускается теми же аргументами путей, что и обычный
runtime, но без `--text` и `--output`:

```bash
PYTHONPATH=src python3 -m voxtream2_ru_jetson.resident \
  --assets /path/to/prompt-assets.json \
  --ruaccent-assets /path/to/ruaccent \
  --phone-map /path/to/phoneme_to_token.json \
  --temp-engine /path/to/temp-unified-q1-q420.engine \
  --dep-engine /path/to/dep.engine \
  --phone-engine /path/to/phone.engine \
  --mimi-engine /path/to/mimi.engine \
  --mimi-state /path/to/mimi-state.json \
  --audio-embedding-weight /path/to/audio_embeddings.bf16 \
  --audio-embedding-cubin /path/to/audio_embedding.cubin \
  --cuda-acoustic-control-cubin /path/to/acoustic_control.cubin \
  --cuda-dep-graph --cuda-temp-graph
```

Описание JSONL-запросов и бинарных PCM-записей находится в
[`resident-runtime.md`](resident-runtime.md).

Runtime создаёт два execution context над одним temporal plan: q=1 для hot
path и q=420 для sink replay. Второго temporal engine в runtime API нет.

Для воспроизведения чанков сразу по мере генерации:

```bash
voxtream2-ru-jetson-playback \
  --text "Привет! Звук уже идёт, пока реплика ещё считается." \
  --device hw:2,0 -- \
  voxtream2-ru-jetson-resident <те же аргументы путей>
```

При запуске resident внутри Docker обязательно используйте `-i` и
`--entrypoint python3`: текст штатного entrypoint не должен попадать в
бинарный stdout.
