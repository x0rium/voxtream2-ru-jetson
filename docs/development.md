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
  --temp-engine /path/to/temp.engine \
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
