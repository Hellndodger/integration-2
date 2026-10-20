# Лабораторна робота №2 — Docker та розгортання ML-моделі

Проєкт пакує аудіокласифікатор із ЛР1 (Google Speech Commands, класи `yes`, `no`, `up`, `down`) у відтворюваний FastAPI-сервіс. Модель навчається під час збирання першого етапу Dockerfile; фінальний runtime-образ містить лише код інференсу, зафіксовані залежності та артефакти `model.pt` і `metrics.json`.

## Структура

```text
lab2/
├── src/
│   ├── data.py                 # завантаження датасету і Mel-спектрограми
│   ├── model.py                # архітектура SpeechCNN
│   ├── train.py                # навчання і збереження артефактів
│   ├── evaluate.py             # незалежна оцінка тестової вибірки
│   └── api.py                  # REST API
├── scripts/benchmark.py        # mean/p50/p95 серверної і клієнтської latency
├── requirements-api.txt
├── requirements-train.txt
├── Dockerfile                  # multi-stage trainer -> runtime
└── Dockerfile.single-stage     # контрольний образ для порівняння
```

## Збирання і запуск

Потрібен Docker Engine/Desktop. Навчальний етап завантажує офіційний Google Speech Commands v0.02 у тимчасовий шар build-контейнера; вихідний датасет не потрапляє у фінальний образ.

```powershell
docker build -t kws-api:1.0.0 -f lab2/Dockerfile lab2
docker run -d --name kws -p 8000:8000 kws-api:1.0.0
docker ps
docker logs kws
```

Навчання за замовчуванням використовує 4 епохи, seed 42 і до 50 файлів кожного класу в кожному split. Для коротшого CI-збирання або експериментів можна змінити параметри:

```powershell
docker build --build-arg EPOCHS=2 --build-arg MAX_PER_CLASS=25 `
  --build-arg MODEL_VERSION=1.0.1 -t kws-api:1.0.1 -f lab2/Dockerfile lab2
```

Перевірте готовність і схему:

```powershell
curl.exe http://localhost:8000/health
curl.exe http://localhost:8000/model-info
curl.exe -X POST -F "file=@lab1/samples/yes.wav" http://localhost:8000/predict
```

Swagger UI: <http://localhost:8000/docs>. `/health` повертає `{"status":"ok"}`; модель та метрики завантажуються один раз під час старту. `/model-info` повертає архітектуру, класи, розмір моделі в КБ, accuracy у відсотках та версію. `/predict` повертає клас, впевненість, ймовірності кожного класу, версію та latency, що охоплює декодування/попередню обробку й інференс. Невалідне розширення, порожній або недекодовуваний файл повертає HTTP 400.

## Порівняння multi-stage та single-stage

Зберіть контрольний образ з тими самими навчальними аргументами:

```powershell
docker build --build-arg EPOCHS=4 --build-arg MODEL_VERSION=1.0.0 `
  -t kws-api:single-stage -f lab2/Dockerfile.single-stage lab2
docker images kws-api
docker history kws-api:1.0.0
docker history kws-api:single-stage
```

Multi-stage не робить сам PyTorch/API runtime магічно меншим: він усуває з фінального шару всі дані навчання, train-only залежності та build-кеш. Знімки `docker images` і `docker history` потрібно додати до звіту після виконання команд.

## Порівняння локально / у контейнері

Для локального запуску встановіть один і той самий API-набір залежностей у віртуальне середовище, навчіть модель на локальному CPU, а потім запустіть API. Ці команди використовують PowerShell:

```powershell
cd lab2
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-train.txt
python -m src.train --data-root ..\lab1\data\speechcommands --output-dir artifacts
$env:MODEL_PATH = "$PWD\artifacts\model.pt"
$env:METRICS_PATH = "$PWD\artifacts\metrics.json"
python -m src.api
```

Якщо архів датасету ще не розпакований у `lab1/data/speechcommands`, тренувальний скрипт завантажить його. В окремому терміналі з каталогу `lab2`:

```powershell
python scripts/benchmark.py --url http://127.0.0.1:8000 --requests 100 --warmup 10
python -m src.evaluate --data-root ..\lab1\data\speechcommands --model-path artifacts\model.pt
```

Для повторного підрахунку accuracy саме в Docker змонтуйте датасет лише на час виконання `evaluate.py`; звичайний runtime-образ його не містить:

```powershell
docker run --rm --entrypoint python `
  -v "${PWD}\lab1\data\speechcommands:/data:ro" `
  kws-api:1.0.0 -m src.evaluate --data-root /data --model-path /app/artifacts/model.pt
```

Порівняйте ті самі запити з контейнером:

```powershell
python lab2/scripts/benchmark.py --url http://127.0.0.1:8000 --requests 100 --warmup 10 --sample lab1/samples/yes.wav
docker stats --no-stream kws
docker images kws-api
```

Скрипт benchmark виводить mean/p50/p95 окремо для серверної latency з JSON-відповіді та end-to-end HTTP latency клієнта. Холодний старт вимірюйте секундоміром від `docker run -d` до першої успішної відповіді `/health`. Для локальної RAM зафіксуйте RSS процесу API (наприклад, через Диспетчер завдань); для контейнера — значення `docker stats --no-stream`.

## Артефакти і конфігурація

| Змінна | Значення за замовчуванням | Призначення |
|---|---|---|
| `MODEL_PATH` | `/app/artifacts/model.pt` | шлях до ваг моделі |
| `METRICS_PATH` | поряд із `MODEL_PATH`, `metrics.json` | метадані й accuracy |
| `MODEL_VERSION` | `1.0.0` | версія моделі в API |
| `PORT` | `8000` | порт HTTP-сервісу |
| `TORCH_NUM_THREADS` | `1` | кількість CPU-потоків PyTorch |

У Dockerfile задані OCI labels із версією, git revision і датою збирання. Модель у штатній конфігурації запечена в образ; для контрольованого оновлення можна змонтувати каталог зовнішніх артефактів read-only та передати `MODEL_PATH` і `METRICS_PATH`:

```powershell
docker run -d --name kws-volume -p 8000:8000 `
  -v "${PWD}\lab2\artifacts:/models:ro" `
  -e MODEL_PATH=/models/model.pt -e METRICS_PATH=/models/metrics.json `
  kws-api:1.0.0
```

## Безпека та CI

Фінальний образ запускається від непривілейованого `appuser`, має `HEALTHCHECK`, а `.dockerignore` вилучає локальні дані, артефакти й кеші з build context. Додаткова перевірка образу (Trivy, якщо встановлений):

```powershell
trivy image kws-api:1.0.0
docker run -d --name kws-secure -p 8001:8000 --read-only --tmpfs /tmp `
  --cap-drop=ALL --security-opt=no-new-privileges kws-api:1.0.0
curl.exe http://localhost:8001/health
docker rm -f kws-secure
```

Workflow `../.github/workflows/ci.yml` запускає Ruff, будує multi-stage образ (3 епохи замість стандартних 4), стартує контейнер, перевіряє health, `/model-info` і реальний WAV із ЛР1. Quality gate встановлено на 45% accuracy — на 20 відсоткових пунктів вище випадкового рівня 25% для чотирьох класів; це gate базового smoke-набору, а не заміна повноцінної оцінки моделі. CI не публікує образ у зовнішній реєстр.

## Таблиця результатів для звіту

Не підставляйте оцінкові значення: запустіть вимірювання на своїй машині та внесіть виводи команд. У CI workflow додайте посилання на успішний run і, за наявності, на run із навмисно завищеним порогом accuracy.

| Середовище | Модель | Розмір моделі, КБ | Accuracy, % | Сервер mean/p95, мс | Клієнт mean/p95, мс | Образ / середовище, МБ | RAM, МБ | Холодний старт, с |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Локально (venv) | SpeechCNN | — | — | — | — | — | — | — |
| Docker (single-stage) | SpeechCNN | — | — | — | — | — | — | — |
| Docker (multi-stage) | SpeechCNN | — | — | — | — | — | — | — |

## Контрольні питання

1. **Образ, контейнер, шар.** Образ — незмінний шаблон із шарами; контейнер — запущений екземпляр образу з процесами та ізольованою файловою системою. Шар — кешований незмінний результат інструкції збірки.
2. **`.dockerignore`.** Виключає файли з build context, зменшуючи його розмір, час передавання і ризик випадково включити секрети чи дані.
3. **Порядок інструкцій і кеш.** Стабільні файли залежностей копіюють та інсталюють до вихідного коду. Тоді зміна коду не інвалідує кеш дорогого `pip install`.
4. **Multi-stage.** Розділяє інструменти навчання і runtime. У фінальний образ копіюються лише модельні артефакти та потрібний для інференсу код.
5. **`CMD`/`ENTRYPOINT`, `EXPOSE`/`-p`.** `CMD` задає команду за замовчуванням, `ENTRYPOINT` — основний виконуваний процес; `EXPOSE` документує порт, а `-p` публікує його на хості.
6. **CPU PyTorch.** Для невеликої моделі CPU-only пакет уникає CUDA runtime і суттєво зменшує образ без вимоги до GPU.
7. **Модель у образі чи volume.** Запечена модель дає відтворюваний immutable release; volume полегшує оновлення ваг без перебудови, але потребує окремого версіонування/контролю відповідності моделі та метрик.
8. **`HEALTHCHECK`.** Періодично перевіряє готовність контейнера; Docker та оркестратори можуть позначити несправний сервіс і не спрямовувати на нього трафік.
9. **Запуск без root.** Компрометація процесу має менші привілеї на хості та всередині контейнера.
10. **Latency.** Контейнер додає незначну віртуалізаційну/мережеву вартість; основне розходження часто спричиняють CPU-ліміти, cold start і HTTP/серіалізація. Окремо вимірюйте серверний та клієнтський час.
11. **Docker у CI/CD ML.** CI збирає версійований образ і перевіряє API та quality gates до публікації/деплою, відтворюючи середовище навчання й інференсу.
