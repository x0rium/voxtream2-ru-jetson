# Инструменты сборки

Эти программы создают воспроизводимые промежуточные артефакты, которые нужны
PyTorch-less runtime:

- `export_voxtream_torchless_assets.py` — fixed-prompt cache и
  framework-neutral binary ABI;
- `export_voxtream_audio_embeddings.py` — таблица audio embeddings в raw BF16;
- `patch_onnx_*.py` — добавление проекционных heads и исправление выходов ONNX;
- `build_tensorrt_engine.py` — сборка TensorRT plan с метриками;
- `download_ruaccent_assets.py` — зафиксированный набор RUAccent для frontend;
- `verify_release.py` — проверка размеров и SHA-256 файлов Hugging Face;
- `inspect_*.py` и `compare_voxtream_captures.py` — диагностика планов и
  сравнение захватов.

Инструменты экспорта могут импортировать PyTorch. Это допустимо: они работают
офлайн. Процесс синтеза в `src/voxtream2_ru_jetson/` PyTorch не импортирует.
