# Звіт з лабораторної роботи №2

**Дисципліна:** Проєкт з інтеграції інформаційних систем
**Тема:** Контейнеризація та розгортання ML-моделі з Docker
**Виконав(ла):** ____________________
**Група:** ____________________
**Дата:** ____________________

## Мета роботи

Опанувати збирання Docker-образів і запуск контейнерів, розгорнути сервіс інференсу аудіомоделі з ЛР1, відокремити етап навчання від runtime за допомогою multi-stage build та виміряти характеристики моделі й середовища.

## Модель і дані з ЛР1

Використовується підмножина Google Speech Commands із чотирма класами `yes`, `no`, `up`, `down`. Аудіо приводиться до mono та 16 кГц, обрізається або доповнюється до однієї секунди. Ознаки — Mel-Spectrogram із 64 mel-смугами, `n_fft=400`, `hop_length=160`, переведений у dB.

Модель — компактна CNN `SpeechCNN`: три згорткові блоки (16, 32 і 64 канали) з ReLU, BatchNorm та pooling, Adaptive Average Pooling і лінійний класифікатор на 4 класи. Артефакти збирання — `artifacts/model.pt` та `artifacts/metrics.json`.

## Архітектура контейнера

Збірка описана у [Dockerfile](./Dockerfile):

1. **trainer:** `python:3.11-slim`, встановлює зафіксовані train-залежності й `libsndfile1`, завантажує Google Speech Commands, навчає модель із фіксованим seed і записує ваги та метрики. Кількість епох і ліміт прикладів налаштовуються через `EPOCHS` та `MAX_PER_CLASS`.
2. **runtime:** окремий `python:3.11-slim`, містить лише API-залежності, вихідний код та артефакти, скопійовані з trainer. Датасет і train-only бібліотеки у фінальний образ не копіюються. Сервіс працює від `appuser`; задані `HEALTHCHECK`, `EXPOSE`, `CMD` та OCI labels.

Контрольний одноетапний варіант наведений у [Dockerfile.single-stage](./Dockerfile.single-stage). У ньому датасет, навчальне середовище й runtime лишаються в одному образі.

`.dockerignore` виключає локальні дані, артефакти, кеш Python, віртуальні середовища та ноутбуки з build context. У [requirements-api.txt](./requirements-api.txt) та [requirements-train.txt](./requirements-train.txt) залежності зафіксовані версіями; для PyTorch використовується CPU wheel.

## Запуск і перевірка

```powershell
docker build -t kws-api:1.0.0 -f lab2/Dockerfile lab2
docker run -d --name kws -p 8000:8000 kws-api:1.0.0
curl.exe http://localhost:8000/health
curl.exe http://localhost:8000/model-info
curl.exe -X POST -F "file=@lab1/samples/yes.wav" http://localhost:8000/predict
```

API завантажує модель один раз під час старту. `/health` призначений для перевірки готовності; `/model-info` повертає розмір моделі, архітектуру, класи, версію та test accuracy у відсотках; `/predict` приймає WAV і повертає клас, впевненість, імовірності класів і latency попередньої обробки разом з інференсом. Для некоректних WAV API повертає HTTP 400.

Сервіс також доступний у Swagger UI: <http://localhost:8000/docs>.

## Вимірювання

Сценарій [benchmark.py](./scripts/benchmark.py) виконує 10 розігрівних і 100 вимірювальних HTTP-запитів, обчислює mean/p50/p95 для серверної та end-to-end клієнтської затримки. Використовуйте той самий WAV і параметри в локальному середовищі та контейнері. Холодний старт вимірюється від `docker run` до першої успішної відповіді `/health`; RAM контейнера — через `docker stats --no-stream`, локального процесу — за його RSS.

| Середовище | Модель | Розмір моделі, КБ | Accuracy, % | Сервер mean/p95, мс | Клієнт mean/p95, мс | Розмір образу / середовища, МБ | RAM, МБ | Холодний старт, с |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Локально (venv, Windows) | SpeechCNN | — | — | — | — | — | — | — |
| Docker (single-stage) | SpeechCNN | 99.84 | 50.5 | 4.15 / 5.58 | 8.63 / 10.79 | 12 400 | 159.9 | 7.03 |
| Docker (multi-stage) | SpeechCNN | 99.84 | 50.5 | 7.12 / 11.52 | 13.75 / 19.43 | 1 460 | 167.7 | 26.99 |

Обидва Docker-образи зібрані з `epochs=4`, `limit_per_class=50`, seed 42: по 200 прикладів у train, validation і test splits. Для кожного API benchmark використано ті самі WAV, 10 запитів розігріву та 100 вимірювальних HTTP-запитів. Розмір образу округлено за показом `docker images` (1.46 GB та 12.4 GB). RAM — показ `docker stats --no-stream`. Cold start виміряно від `docker run` до першої відповіді `/health`; це не час до переходу Docker health status у `healthy`.

Локальний рядок не заповнений: на Windows `pip install -r requirements-train.txt` не знайшов зафіксований `torch==2.5.1+cpu` у доступному індексі; після невдалого встановлення навчання й API не стартували через відсутність PyTorch.

Артефакт із точними метриками Docker-збірки доступний у контейнері за шляхом `/app/artifacts/metrics.json` і в ендпоінті `/model-info`. Щоб скопіювати його з запущеного контейнера на хост, виконайте `docker cp kws:/app/artifacts/metrics.json .\metrics.json`. Для локального навчання метрики записуються у `lab2/artifacts/metrics.json`. Щоб повторно порахувати accuracy всередині Docker, запустіть `src.evaluate` з read-only volume датасету, як описано в [README](./README.md).

Щоб виміряти розміри образів і проаналізувати шари, окремо зберіть контрольний варіант та виконайте:

```powershell
docker build --build-arg EPOCHS=4 -t kws-api:single-stage -f lab2/Dockerfile.single-stage lab2
docker images kws-api
docker history kws-api:1.0.0
docker history kws-api:single-stage
```

За `docker history`, найбільший шар single-stage — команда навчання, близько 5.87 GB; шар встановлення train-залежностей — близько 1.22 GB. У multi-stage ці train-шари та датасет лишаються у стадії `trainer` і не переходять у runtime; runtime містить приблизно 1 GB шару API-залежностей. У вимірюванні повний multi-stage образ — близько 1.46 GB, тоді як single-stage — 12.4 GB: зменшення приблизно на 88.2% (близько 10.94 GB). Обидва образи все ще містять CPU PyTorch, тому runtime не є малим абсолютним розміром.

## CI та якість моделі

Workflow [ci.yml](../.github/workflows/ci.yml) запускає Ruff, збирає multi-stage образ (3 епохи в CI проти 4 за замовчуванням), запускає контейнер, перевіряє `/health`, `/model-info` і запит `/predict` із тестовим WAV. Quality gate вимагає не менше 45% accuracy. Для чотирьох класів випадковий орієнтир становить 25%; поріг 45% — на 20 відсоткових пунктів вищий. У перевіреному локальному прогоні 3 епох accuracy склала 47.5%. Це поріг smoke-gate, а не заміна повноцінної оцінки на більшому наборі.

**Посилання на запуск CI:** [run №1](https://github.com/Hellndodger/integration-2/actions/runs/39353513906) — workflow було запущено подією push, але job не стартував через billing lock GitHub-акаунта. Це не результат помилки lint, build чи quality gate.
**Невдалий run через порушення quality gate:** відсутній; job не дійшов до перевірки моделі через блокування запуску runner.

## Ілюстрації для фінальної здачі

Додайте власні знімки з фактичного запуску — результати не вигадуються наперед:

1. Вивід `docker build` з етапами `trainer` і `runtime`.
2. `docker images` з обома варіантами образу.
3. `docker ps` зі статусом `healthy`.
4. Відповіді `/health`, `/model-info`, `/predict` або запит у Swagger UI.
5. `docker stats --no-stream kws`.
6. GitHub Actions run після усунення billing lock; на момент підготовки звіту runner не запускався.
7. `docker history` для single-stage та multi-stage з аналізом найбільших шарів.

## Опціональні завдання

| Завдання | Стан | Реалізація / результат |
|---|---|---|
| Оптимізація образу | Виконано частково | Використано `python:3.11-slim` і CPU-only PyTorch; виміряно single-stage 12.4 GB та multi-stage 1.46 GB. Варіанти CUDA, ONNX, Alpine/distroless і квантизацію не порівнювали. |
| Версіонування моделі | Виконано частково | API повертає `MODEL_VERSION`; Dockerfile задає OCI labels для version, revision і build date. MLflow не інтегрували. |
| Зовнішня модель через volume | Реалізовано | API підтримує `MODEL_PATH` і `METRICS_PATH`; зовнішні артефакти read-only монтували для тесту API. Окрему швидкість оновлення та розмір образу без вбудованої моделі не вимірювали. |
| CI/CD | Налаштовано, але запуск заблокований | Workflow доданий, однак GitHub не виділив runner через billing lock акаунта. |
| Безпека образу | Виконано частково | Контейнер запускається не від root (UID 10001), має `HEALTHCHECK`. Trivy та комбінацію `--read-only`, `--cap-drop=ALL`, `--security-opt=no-new-privileges` не перевіряли. |
| Compose UI/API, MLflow, Prometheus/Grafana, навантажувальне тестування, GPU | Не виконувались | Окремі сервіси й графіки для цих розширень не створювалися. |

Для невиконаних опціональних розширень графіки не наводяться, щоб не підміняти виміряні результати демонстраційними даними.

## Висновки

У роботі побудовано відтворюваний pipeline, що навчає SpeechCNN у build-стадії та запускає REST API в окремому непривілейованому runtime-контейнері. Обидва варіанти досягли 50.5% test accuracy. Multi-stage образ приблизно на 88.2% менший за single-stage, оскільки не містить train-залежностей і даних датасету. У вимірюваннях multi-stage cold start був довшим (26.99 с проти 7.03 с), а RAM подібною (167.7 MiB проти 159.9 MiB); latency відрізнялась, тому порівняння виконано на тому самому хості й WAV. CI workflow додано, однак фактичний run наразі заблоковано проблемою білінгу GitHub-акаунта, а не кодом.

**Репозиторій:** <https://github.com/Hellndodger/integration-2>
**Коміт із кодом:** `df181ca5cac87048e88cac446c372e3e584c39c1`
