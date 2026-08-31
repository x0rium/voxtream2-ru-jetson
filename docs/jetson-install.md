# Установка релиза на Jetson

Эта инструкция собирает рабочий PyTorch-less тракт из исходного кода GitHub и
переносимых артефактов Hugging Face. Она рассчитана на Jetson Orin с JetPack
6.2.3, Jetson Linux R36.5.2 и TensorRT `10.3.0.30-1+cuda12.5`.

TensorRT plan-файлы собираются на том Jetson, где будут запускаться. Между RTX
и Jetson, а также между разными patch-версиями TensorRT они не переносятся.

## Что потребуется

- Jetson Orin Nano 8 GB или другой Orin с GPU `sm_87`;
- JetPack 6.2.3 / Jetson Linux R36.5.2;
- Docker с NVIDIA runtime;
- `/usr/local/cuda/bin/nvcc` из JetPack;
- не менее 8 GB свободного места, лучше 12 GB с запасом на Docker и временные
  файлы сборки;
- сеть только для клонирования репозиториев и скачивания RUAccent. Сам синтез
  работает без сети.

Проверьте платформу и точную версию TensorRT:

```bash
cat /etc/nv_tegra_release
dpkg-query -W nvidia-jetpack 'libnvinfer*' 2>/dev/null
/usr/local/cuda/bin/nvcc --version
```

Проверенная версия кода — `v0.2.2`:

```bash
git clone --branch v0.2.2 --depth 1 \
  https://github.com/x0rium/voxtream2-ru-jetson.git
cd voxtream2-ru-jetson
```

## 1. Runtime-образ

```bash
docker build \
  -t voxtream2-ru-jetson:runtime \
  -f docker/voxtream2-ru/Dockerfile \
  .

docker run --rm \
  --runtime nvidia \
  --network none \
  --entrypoint python3 \
  voxtream2-ru-jetson:runtime \
  -c 'import importlib.util as u, tensorrt; assert u.find_spec("torch") is None; print(tensorrt.__version__)'
```

Последняя команда должна вывести `10.3.0` и завершиться без ошибки.

## 2. Артефакты Hugging Face

Установите CLI в отдельное лёгкое окружение и скачайте зафиксированный релиз:

```bash
python3 -m venv .venv-hf
. .venv-hf/bin/activate
python3 -m pip install 'huggingface_hub>=1.0,<2'
hf download x0rium/voxtream2-ru-jetson-onnx \
  --revision v0.2.2 \
  --local-dir release
deactivate

python3 tools/verify_release.py release
```

Проверка должна напечатать `OK` для каждого файла из `manifest.json`. В релиз
входят четыре ONNX, таблица audio embeddings, начальное состояние Mimi,
таблица фонем и подготовленное состояние демонстрационного голоса.

## 3. TensorRT engines

Создайте каталоги для plan-файлов и отчётов сборки:

```bash
mkdir -p artifacts/engines artifacts/build-metrics artifacts/cuda
```

Соберите phone encoder с диапазоном длины фонем от 2 до 640:

```bash
docker run --rm --runtime nvidia --network none \
  -v "$PWD:/work" -w /work \
  --entrypoint python3 \
  voxtream2-ru-jetson:runtime \
  tools/build_tensorrt_engine.py \
  --onnx release/models/phone-encoder.onnx \
  --engine artifacts/engines/phone-encoder.engine \
  --sequence-range 2 128 640 \
  --workspace-mib 256 \
  --optimization-level 1 \
  --metrics artifacts/build-metrics/phone-encoder.json
```

Максимум 640 выбран как конечная верхняя граница одного TensorRT optimization
profile на Jetson, а не как предел длительности TTS. Phone encoder обрабатывает
каждый сегмент целиком, поэтому TensorRT требует конкретный `MAX` для выбора
тактик и планирования памяти. Если нормализованный текст не помещается в один
сегмент, runtime автоматически делит его по естественным границам и продолжает
выдавать один PCM-поток. Пересобирать engine для обычных длинных текстов не
нужно.

Соберите единый temporal plan. Профиль q=1 обслуживает обычный шаг генерации,
q=420 — восстановление sink-attention после заполнения окна:

```bash
docker run --rm --runtime nvidia --network none \
  -v "$PWD:/work" -w /work \
  --entrypoint python3 \
  voxtream2-ru-jetson:runtime \
  tools/build_tensorrt_engine.py \
  --onnx release/models/temp-former-q1-q420.onnx \
  --engine artifacts/engines/temp-former.engine \
  --sequence-profiles 1 420 \
  --workspace-mib 512 \
  --optimization-level 1 \
  --metrics artifacts/build-metrics/temp-former.json
```

Соберите acoustic dep-former с двумя одношаговыми профилями и профилем q=2:

```bash
docker run --rm --runtime nvidia --network none \
  -v "$PWD:/work" -w /work \
  --entrypoint python3 \
  voxtream2-ru-jetson:runtime \
  tools/build_tensorrt_engine.py \
  --onnx release/models/dep-former-q1-q2.onnx \
  --engine artifacts/engines/dep-former.engine \
  --sequence-profiles 1 1 2 \
  --workspace-mib 256 \
  --optimization-level 1 \
  --metrics artifacts/build-metrics/dep-former.json
```

Mimi decoder имеет статические формы:

```bash
docker run --rm --runtime nvidia --network none \
  -v "$PWD:/work" -w /work \
  --entrypoint python3 \
  voxtream2-ru-jetson:runtime \
  tools/build_tensorrt_engine.py \
  --onnx release/models/mimi-decoder-step.onnx \
  --engine artifacts/engines/mimi-decoder.engine \
  --workspace-mib 256 \
  --optimization-level 1 \
  --metrics artifacts/build-metrics/mimi-decoder.json
```

Успешная сборка каждого движка заканчивается JSON-отчётом без ошибок parser и
с ненулевым `engine_bytes`.

## 4. CUDA kernels

Два небольших kernel собираются штатным `nvcc` для Orin (`sm_87`):

```bash
/usr/local/cuda/bin/nvcc --cubin --gpu-architecture=sm_87 -O3 \
  native/cuda/audio_embedding.cu \
  -o artifacts/cuda/audio-embedding.cubin

/usr/local/cuda/bin/nvcc --cubin --gpu-architecture=sm_87 -O3 \
  native/cuda/torchless_acoustic_control.cu \
  -o artifacts/cuda/acoustic-control.cubin
```

На проверенной CUDA 12.6 получаются файлы размером 5800 и 4896 байт
соответственно. Хеш может измениться при другой версии компилятора.

## 5. Русский frontend

RUAccent хранится в отдельном публичном репозитории и не дублируется в нашем
релизе. Вспомогательная команда скачивает только нужные словари, ONNX и
tokenizer-файлы с зафиксированной revision:

```bash
docker run --rm --network bridge \
  -v "$PWD:/work" -w /work \
  --entrypoint python3 \
  voxtream2-ru-jetson:runtime \
  tools/download_ruaccent_assets.py artifacts/ruaccent
```

После этого каталог `artifacts/ruaccent` можно использовать без сети.

## 6. Первый WAV

```bash
docker run --rm --runtime nvidia --network none \
  -v "$PWD:/work" -w /work \
  -e PYTHONPATH=/work/src \
  --entrypoint python3 \
  voxtream2-ru-jetson:runtime \
  -m voxtream2_ru_jetson \
  --assets /work/release/voices/example-f1.json \
  --text 'Сегодня 1 сентября 2026 года, заряд — 87 процентов.' \
  --ruaccent-assets /work/artifacts/ruaccent \
  --phone-map /work/release/assets/phoneme-to-token.json \
  --phone-engine /work/artifacts/engines/phone-encoder.engine \
  --temp-engine /work/artifacts/engines/temp-former.engine \
  --dep-engine /work/artifacts/engines/dep-former.engine \
  --mimi-engine /work/artifacts/engines/mimi-decoder.engine \
  --mimi-state /work/release/assets/mimi-decoder-initial-state.json \
  --audio-embedding-weight /work/release/assets/audio-embeddings.bf16 \
  --audio-embedding-cubin /work/artifacts/cuda/audio-embedding.cubin \
  --cuda-acoustic-control-cubin /work/artifacts/cuda/acoustic-control.cubin \
  --cuda-dep-graph --cuda-temp-graph \
  --metrics /work/artifacts/smoke.json \
  --output /work/artifacts/smoke.wav
```

Проверка результата:

```bash
python3 - <<'PY'
import json
m = json.load(open('artifacts/smoke.json'))
assert m['torch_imported'] is False
assert not m['frontend']['unknown_phones']
assert m['audio_frames'] > 0
print({k: m[k] for k in ('audio_seconds', 'rtf', 'ttfa_seconds', 'max_rss_mib')})
PY
```

Чтобы послушать WAV на Mac, скопируйте его с Jetson и используйте системный
консольный проигрыватель:

```bash
scp jetson:/path/to/voxtream2-ru-jetson/artifacts/smoke.wav .
afplay smoke.wav
```

Сырой PCM-файл требует явных параметров формата:

```bash
brew install ffmpeg
ffplay -nodisp -autoexit -f s16le -ar 24000 -ac 1 output.pcm
```

Вывод `resident` нельзя сохранять как `.wav`: это бинарный протокол с
заголовками и PCM-записями. Для него нужен `voxtream2-ru-jetson-playback` или
собственный клиент протокола из `resident-runtime.md`.

Для живого диалога не запускайте этот одноразовый процесс на каждую фразу.
Используйте резидентный процесс и поток PCM из
[`resident-runtime.md`](resident-runtime.md): тогда модели и RUAccent остаются
в памяти, а первые аудиочанки выдаются до окончания синтеза всей реплики.

## Что ещё не автоматизировано

Релиз включает один готовый демонстрационный голос. Подготовка нового голоса
пока остаётся офлайн-операцией с PyTorch: нужно получить prompt cache и
пересобрать fixed-prompt state. Рабочий процесс синтеза от этого PyTorch не
зависит, но однокнопочного экспортёра нового голоса в текущем релизе нет.
