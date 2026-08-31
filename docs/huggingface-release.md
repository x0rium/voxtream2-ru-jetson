# Публикация ONNX для Jetson

Цель релиза — дать повторяемый путь от готовых framework-neutral assets до
PyTorch-less VoXtream2-RU runtime на Jetson. Код, экспортёры и инструкции живут
в GitHub; крупные ONNX и сопутствующие метаданные — в Hugging Face.

Зафиксированный набор публичных репозиториев:

- GitHub: `x0rium/voxtream2-ru-jetson`;
- Hugging Face: `x0rium/voxtream2-ru-jetson-onnx`.

В README GitHub будет прямая ссылка на HF assets, а в model card HF — ссылка на
исходники и точный git tag релиза.

## Что выкладываем на Hugging Face

- проверенные ONNX для phone encoder, единого dynamic `temp_former` q=1/q=420,
  `dep_former` и Mimi;
- таблицу audio embeddings в raw BF16 и подготовленное состояние одного
  демонстрационного голоса, чтобы релиз можно было запустить без PyTorch;
- manifest с формами и dtype всех входов, выходов и state tensors;
- SHA-256 каждого публикуемого runtime-артефакта;
- версии PyTorch, ONNX и torchtune, использованные только при экспорте;
- эталонные входы и численные метрики сравнения с исходной моделью;
- лицензию и атрибуцию upstream-модели.

TensorRT `.engine` не объявляем переносимым артефактом: plan зависит от
архитектуры GPU, версии TensorRT и JetPack/L4T. На HF можно хранить проверенные
plans как явно помеченные примеры для одной конфигурации, но основной путь —
скачать ONNX и собрать engine на целевом Jetson.

Текущий воспроизводимый export stack зафиксирован в
[`experiments/tts/requirements-export.txt`](../experiments/tts/requirements-export.txt):
PyTorch 2.7.0, torchtune 0.4.0, ONNX 1.17.0, ONNX Script 0.3.2 и ONNX IR 0.1.9.
Это функциональные pins: torchtune 0.5 изменил KV-cache ABI с 16 attention
heads на 4 KV heads, а ONNX IR 1.0 несовместим со старым version-conversion
pass ONNX Script 0.3.2. Экспортёр проверяет ожидаемую форму KV до записи ONNX.

## Что должна закрывать инструкция

1. Поддерживаемая конфигурация Jetson, JetPack/L4T, Docker и свободное место.
2. Скачивание assets с фиксированной revision и проверка SHA-256.
3. Сборка каждого TensorRT engine на устройстве.
4. Запуск с готовым демонстрационным голосом и отдельная граница процесса
   подготовки нового голоса.
5. Запуск resident runtime без импорта PyTorch.
6. Короткий smoke test, длинный тест через границу sink-attention и ожидаемые
   диапазоны TTFA, RTF и памяти.
7. Диагностика несовместимого TensorRT plan, нехватки памяти и неизвестных
   фонем.

## Gate перед публикацией

- [x] единый `temp_former` q=1/q=420 выбран по побитовому short regression,
  полному длинному прогону и слепому A/B после compaction: q=420 принят без
  слышимой разницы относительно q=1;
- [x] релизная инструкция воспроизведена в новом каталоге: manifest, четыре
  engines, CUDA kernels, RUAccent и итоговый WAV;
- [x] ONNX checker 1.17.0 и TensorRT parser проходят на всех четырёх файлах;
- [x] все ссылки используют release tag или HF revision, а не плавающий `main`;
- [x] в публичных файлах нет локальных IP-адресов, ключей, паролей и путей из
  домашней сети.
