# ЗВІТ
## до лабораторної роботи №3
### «Побудова CI/CD-пайплайна для ML-сервісу»

**Заклад вищої освіти:** [назва закладу]  
**Кафедра:** [назва кафедри]  
**Дисципліна:** Інтеграція інформаційних систем  
**Виконав(ла):** [ПІБ], група [номер групи]  
**Перевірив(ла):** [ПІБ викладача]  
**Місто:** [місто], 2026

## Мета роботи

Автоматизувати збирання, навчання, перевірку й пакування ML-сервісу засобами GitHub Actions та Docker; передавати артефакти між етапами, вимірювати якість і затримку API, а також публікувати перевірений runtime-образ до GitHub Container Registry за визначеними умовами.

## 1. Вихідна модель і датасет

У роботі використано аудіокласифікатор із лабораторних робіт №1–2. Датасет — Google Speech Commands v0.02, чотири класи: `yes`, `no`, `up`, `down`. Аудіо приводиться до монофонічного сигналу частотою 16 кГц, доповнюється або обрізається до однієї секунди й перетворюється на Mel-спектрограму. Модель — компактна CNN `SpeechCNN` із трьома згортковими блоками та класифікатором на чотири виходи.

Для контрольного локального запуску використано seed `42`, чотири епохи та не більше 50 прикладів на клас у кожному наборі. Отже, у train, validation і test було по 200 аудіозаписів. Для перевірки API застосовано окремі WAV-файли з офіційного testing split; скрипт підготовки перевіряє їхню приналежність до цього split і не додає самі записи до Git-репозиторію.

## 2. Структура проєкту

```text
.github/workflows/ci.yml       # CI/CD: lint → pipeline (train → verify → build-push)
.dockerignore                  # виключення даних і локальних середовищ із Docker context
lab2/
├── requirements-api.txt       # залежності API та інференсу
├── requirements-train.txt     # train-залежності поверх API-набору
├── src/
│   ├── api.py                 # FastAPI endpoints
│   ├── data.py                # датасет і Mel-препроцесинг
│   ├── model.py               # SpeechCNN
│   └── train.py               # train/evaluate та збереження артефактів
└── scripts/benchmark.py       # вимірювання server/client latency
lab3/
├── Dockerfile                 # окремі trainer і runtime targets
├── golden/expected.json       # еталонні класи та джерела test WAV
└── scripts/
    ├── prepare_golden.py      # перевірка й підготовка held-out WAV
    ├── verify.py              # API, schema, golden checks і quality gates
    └── write_summary.py       # Markdown-звіт запуску та Job Summary
```

## 3. Dockerfile та залежності

У [Dockerfile ЛР3](./Dockerfile) визначено два незалежні targets.

1. **`trainer`** базується на `python:3.11-slim-bookworm`, встановлює `libsndfile1` і залежності з `requirements-train.txt`, копіює код навчання та запускає `python -m src.train`. Датасет і каталог артефактів підключаються як volumes, тому датасет не запікається в образ.
2. **`runtime`** встановлює runtime-залежності, додає FastAPI-код і три необхідні артефакти (`model.pt`, `labels.json`, `metrics.json`), створює непривілейованого `appuser`, відкриває порт 8000 і перевіряє готовність через Docker `HEALTHCHECK`.

Фінальний runtime-образ збирається після навчання й не містить train-only середовища та вихідного датасету. `.dockerignore` виключає `.git`, Python cache/venv, дані ЛР1 і попередні локальні артефакти ЛР2. Каталог `lab3/artifacts` зберігається у workspace pipeline job-а та доступний для наступних кроків без завантаження до Actions artifact storage.

Залежності зафіксовані у двох файлах:

- `lab2/requirements-api.txt`: CPU-збірки PyTorch 2.5.1 і torchaudio 2.5.1, FastAPI, Uvicorn, soundfile, NumPy та requests.
- `lab2/requirements-train.txt`: включає `requirements-api.txt` і scikit-learn 1.5.2.

## 4. CI/CD-процес

Workflow запускається під час push у `main`, pull request до `main` і вручну через `workflow_dispatch`. Ручний запуск приймає кількість епох, ліміт прикладів на клас і перемикач публікації.

```text
lint → pipeline (train → verify → build-push)
```

- **lint:** Ruff і перевірка синтаксису Python.
- **pipeline:** послідовно збирає trainer target, навчає й оцінює модель у контейнері, створює `model.pt`, `labels.json`, `metrics.json`, `train.log`, готує held-out WAV, збирає runtime target, перевіряє `/health`, `/model-info`, `/predict`, чотири golden samples, accuracy та p95 latency, а потім збирає final image, записує `docker history` і формує GitHub Job Summary. Файли передаються між кроками у спільному workspace, без `upload-artifact`/`download-artifact`. Публікація в GHCR виконується для перевіреного push у `main`; для ручного запуску її потрібно окремо ввімкнути.

Оскільки репозиторій публічний, pull request із fork пропускається: неперевірений fork-код не запускається на приватному self-hosted runner. Для довіри до runner-а обробляються лише push, ручний запуск і pull request із гілок цього репозиторію.

## 5. Результати локальної перевірки

Trainer і runtime image було зібрано та запущено в Docker Desktop. Наведені нижче показники отримано локально; вони не є вимірами GitHub runner-а. Latency виміряно для 30 запитів після 5 warm-up запитів.

| Датасет / модальність | Модель | Розмір ваг | Test accuracy | Server latency, mean | Client latency, p50 / p95 | Час навчання |
|---|---|---:|---:|---:|---:|---:|
| Google Speech Commands / аудіо | SpeechCNN, 4 класи | 102 235 B (99,84 KiB) | 50,5% (101/200) | 7,68 ms | 15,38 / 17,20 ms | 73,55 s |

| Docker-показник | Локальний результат |
|---|---:|
| Runtime image `speech-lab3-api:local` (`docker images`) | 1,46 GB |
| Trainer image `speech-lab3-trainer:local` (`docker images`) | 1,74 GB |
| API-контейнер під час benchmark (`docker stats --no-stream`) | 156,7 MiB RAM; 0,27% CPU |
| Quality gate | Passed: accuracy ≥ 45%, client p95 ≤ 5000 ms |

API повернув успішний `/health`. Усі чотири записи з testing split класифіковано відповідно до їхніх міток: `yes → yes`, `no → no`, `up → up`, `down → down`. Суми ймовірностей у відповідях були перевірені на рівність 1 з допуском 0,02.

Як історичний орієнтир контейнеризації ЛР2 у локальному Docker збережені образи `kws-api:single-stage` (12,4 GB) і `kws-api:1.0.0` (1,46 GB). За колонкою `SIZE` команди `docker images` multi-stage runtime-образ приблизно на 88,2% менший за single-stage. Це попередньо зібраний локальний baseline ЛР2, а не повторне збирання single-stage в цьому запуску ЛР3.

**GitHub Actions:** перший запуск виявив несумісність завантажуваного Python із CPU ISA віртуальної машини; наступний — відсутній Docker CLI/доступ до Docker socket у runner-користувача. Після усунення цих перешкод запуск дійшов до завантаження артефактів, але GitHub відхилив `training-results` через вичерпану квоту Actions artifact storage. Workflow змінено: навчання, перевірка API й пакування образу виконуються в одному pipeline job із передаванням файлів через спільний workspace, без завантаження Actions-артефактів. Успішний запуск цієї зміненої версії ще потрібно підтвердити окремим запуском.

## 6. Скріншоти результатів

Зробити скриншоти після завершення workflow у GitHub → **Actions** → **ML CI/CD pipeline** → потрібний запуск. У звіт вставити зображення з підписами:

1. **Рисунок 1 — Успішне виконання job `lint`.** Відкрити job і зафіксувати зелені кроки Ruff та compileall.
2. **Рисунок 2 — Навчання моделі в Docker.** У job `pipeline` показати запуск контейнера, фінальні метрики `metrics.json` і перелік створених файлів.
3. **Рисунок 3 — Результат перевірки REST API.** У job `pipeline` показати успішний `/health`, чотири golden predictions, accuracy gate і latency gate.
4. **Рисунок 4 — Вимірювання latency.** У логах кроку `Measure latency` показати кількість запитів, warm-up і значення mean/p50/p95.
5. **Рисунок 5 — Підсумок GitHub Actions.** На вкладці **Summary** показати таблицю метрик та підсумок golden-sample перевірки.
6. **Рисунок 6 — Runtime Docker image та його шари.** У job `pipeline` показати вивід `docker images` і `docker history`.
7. **Рисунок 7 — Публікація в GHCR.** Додати лише якщо workflow справді виконав publish; показати пакет і тег у розділі **Packages**.

Окремо збережіть сторінку run із видимими commit SHA та фінальним статусом. Не використовуйте скриншот YAML як доказ успішного запуску. Fork pull request у цьому workflow має бути пропущений за правилом безпеки, а не показаний як успішно виконаний.

## 7. Висновки

Реалізовано відтворюваний Docker-based процес навчання й розгортання аудіомоделі: lint, train, передавання артефактів, інтеграційна перевірка API, вимірювання затримки, quality gates і підготовка GHCR image. Локально підтверджено успішне навчання в контейнері, точність 50,5%, чотири правильні перевірочні прогнози та client p95 latency 17,2 ms.

Накладні витрати контейнеризації проявляються насамперед у розмірі образу та стартовому використанні пам’яті: runtime містить Python, PyTorch, torchaudio й системну бібліотеку для аудіо. Trainer image більший, бо додає навчальні залежності. Розділення на targets і виключення датасету з фінального образу не прибирають великі runtime-залежності, але не переносять до сервісу навчальний набір даних і зайві артефакти. Після завершення GitHub run до звіту слід додати його фактичний статус і, якщо відрізняються, метрики, виміряні саме на self-hosted runner.
