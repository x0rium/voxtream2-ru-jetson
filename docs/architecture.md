# Архитектура текущего TTS-прототипа

Цель текущей ветки — качественный русский TTS на Jetson Orin Nano 8 GB без
PyTorch в процессе синтеза. Это исследовательский runtime, а не готовый сервис.

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

`src/voxtream2_ru_jetson/` содержит только рабочий PyTorch-less путь. Модельные веса,
голос, fixed-prompt cache и TensorRT plans готовятся заранее. Для этого служат
`tools/` и диагностические программы из `experiments/tts/`.

## Что уже проверено

- Полный runtime завершает генерацию при `torch_imported=false`.
- Phone encoder, `temp_former`, `dep_former` и Mimi работают через TensorRT.
- CUDA Graph не изменил принятую траекторию и WAV в проведённой побитовой
  проверке.
- Фраза с датой, временем и процентами дала 0 неизвестных фонем. На Jetson она
  сгенерировала 10.96 секунды аудио за 10.15 секунды: RTF 0.926, TTFA ядра
  0.271 секунды, peak RSS 2.30 GiB.

## Где проходит граница результата

- Голос и prompt cache пока фиксированы заранее.
- Cold start RUAccent занимает около 13 секунд. Для живого устройства нужен
  резидентный процесс, но его ещё нет.
- Нормализация русского текста не закрыта по качеству: известны ошибки падежей,
  версий программ, телефонов и некоторых единиц.
- Sink-attention compaction после позиции 624 ещё не реализован.
- Streaming сейчас есть внутри генератора, но наружу runtime пишет WAV. API для
  выдачи PCM-чанков потребителю ещё не оформлен.
