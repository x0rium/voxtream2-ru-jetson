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

Для текущей sink-attention схемы fixed-shape prefill содержит 420 позиций:
108 кадров голосового prompt и 312 последних temporal-входов. ONNX можно
экспортировать на машине с PyTorch и CUDA. Для воспроизводимого ABI cache нужны
`torchtune==0.4.0` и `torchao==0.9.0`: более новый `torchtune 0.6.1` создаёт для
этого вызова четыре KV-головы вместо шестнадцати и несовместим с runtime-engine.

```bash
PYTHONPATH=experiments/tts python3 \
  experiments/tts/voxtream_tensorrt_temp_prefill_probe.py \
  --checkpoint /path/to/model.safetensors \
  --sequence-length 420 \
  --output-dir /path/to/export
```

Готовый ONNX нужно перенести на Jetson и собрать там:

```bash
sudo systemctl stop gdm3
docker run --rm --runtime nvidia --network none \
  -v /path/to/worktree:/work -w /work \
  voxtream2-ru-jetson:runtime \
  python3 tools/build_tensorrt_engine.py \
  --onnx artifacts/temp-prefill-explicit-kv-q420.onnx \
  --engine artifacts/temp-prefill-q420.engine \
  --workspace-mib 512 \
  --optimization-level 0 \
  --metrics artifacts/temp-prefill-q420-build.json
```

Engine нужно собирать внутри того же runtime-образа, в котором он будет
запускаться. Даже patch-сборки TensorRT `10.3.0.26` и `10.3.0.30` создают
несовместимые plan-файлы. Короткая строка `tensorrt.__version__ == "10.3.0"`
этого различия не показывает; точную версию проверяют через `dpkg-query`.

Если длина prompt изменится, `--sequence-length` тоже нужно изменить на сумму
длины prompt и 312 сохранённых входов. Runtime проверяет это соответствие до
начала синтеза.

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
sudo systemctl stop gdm3
```

q=420 конфигурация проверена на Jetson 8 GB в headless-режиме. Для временной
отладки desktop возвращается командой `sudo systemctl start gdm3`; постоянно
отключать его имеет смысл только после оформления TTS как системного сервиса.

```bash
PYTHONPATH=src python3 -m voxtream2_ru_jetson \
  --assets /path/to/prompt-assets.json \
  --text "Сегодня 30 августа 2026 года, время 21:45." \
  --ruaccent-assets /path/to/ruaccent \
  --phone-map /path/to/phoneme_to_token.json \
  --temp-engine /path/to/temp.engine \
  --temp-prefill-engine /path/to/temp-prefill-q420.engine \
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
  --temp-engine /path/to/temp.engine \
  --temp-prefill-engine /path/to/temp-prefill-q420.engine \
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
