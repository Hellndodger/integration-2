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
| Локально (venv) | SpeechCNN | ___ | ___ | ___ / ___ | ___ / ___ | ___ | ___ | ___ |
| Docker (single-stage) | SpeechCNN | ___ | ___ | ___ / ___ | ___ / ___ | ___ | ___ | ___ |
| Docker multi-stage runtime (3-epoch model via volume) | SpeechCNN | 99.84 | 47.5 | 4.36 / 6.17 | 8.56 / 10.95 | ___ | ___ | ___ |

Зафіксовані вимірювання для останнього рядка: навчання в Docker з `epochs=3`, `limit_per_class=50`, seed 42 (200 train, 200 validation і 200 test прикладів); артефакт тестувався через multi-stage runtime API зі змінними `MODEL_PATH`/`METRICS_PATH`, 10 запитів розігріву та 100 вимірювальних HTTP-запитів. Таблиця розміру образу, RAM і cold start лишена для окремого вимірювання саме фінального образу з цією моделлю; не підмінюйте їх значеннями від тестового образу з іншими вагами.

Артефакт із точними метриками Docker-збірки доступний у контейнері за шляхом `/app/artifacts/metrics.json` і в ендпоінті `/model-info`. Щоб скопіювати його з запущеного контейнера на хост, виконайте `docker cp kws:/app/artifacts/metrics.json .\metrics.json`. Для локального навчання метрики записуються у `lab2/artifacts/metrics.json`. Щоб повторно порахувати accuracy всередині Docker, запустіть `src.evaluate` з read-only volume датасету, як описано в [README](./README.md).

Щоб виміряти розміри образів і проаналізувати шари, окремо зберіть контрольний варіант та виконайте:

```powershell
docker build --build-arg EPOCHS=4 -t kws-api:single-stage -f lab2/Dockerfile.single-stage lab2
docker images kws-api
docker history kws-api:1.0.0
docker history kws-api:single-stage
```

Очікуване пояснення: у multi-stage runtime-образ не включає датасет, train-only залежності й результати проміжних build-шарів. Сам runtime усе ще містить CPU PyTorch та бібліотеки інференсу, тому вони складають значну частку його розміру.

## CI та якість моделі

Workflow [ci.yml](../.github/workflows/ci.yml) запускає Ruff, збирає multi-stage образ (3 епохи в CI проти 4 за замовчуванням), запускає контейнер, перевіряє `/health`, `/model-info` і запит `/predict` із тестовим WAV. Quality gate вимагає не менше 45% accuracy. Для чотирьох класів випадковий орієнтир становить 25%; поріг 45% — на 20 відсоткових пунктів вищий. У перевіреному локальному прогоні 3 епох accuracy склала 47.5%. Це поріг smoke-gate, а не заміна повноцінної оцінки на більшому наборі.

**Посилання на успішний CI run:** ____________________
**Приклад невдалого run із порушенням quality gate (за наявності):** ____________________

## Ілюстрації для фінальної здачі

Додайте власні знімки з фактичного запуску — результати не вигадуються наперед:

1. Вивід `docker build` з етапами `trainer` і `runtime`.
2. `docker images` з обома варіантами образу.
3. `docker ps` зі статусом `healthy`.
4. Відповіді `/health`, `/model-info`, `/predict` або запит у Swagger UI.
5. `docker stats --no-stream kws`.
6. Успішний запуск GitHub Actions.
7. `docker history` для single-stage та multi-stage з аналізом найбільших шарів.

## Висновки

У роботі відтворюваний pipeline навчає модель у build-стадії та розгортає REST API у окремому непривілейованому runtime-контейнері. Multi-stage build відокремлює середовище навчання від середовища інференсу, а health check та CI smoke tests перевіряють готовність і базову якість сервісу. Після фактичних запусків доповніть таблицю виміряними метриками та поясніть різницю latency, RAM, холодного старту й розміру образів.
