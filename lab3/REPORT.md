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

Автоматизувати перевірку коду, навчання та оцінювання ML-моделі, тестування REST API й збирання Docker-образу засобами GitHub Actions. Виміряти точність моделі та затримку API, застосувати quality gates і опублікувати перевірений runtime-образ до GitHub Container Registry.

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

## 6. Результати GitHub Actions

Фінальний запуск №9 виконано успішно на self-hosted runner `ubuntu-runner` під Linux x64. Усі етапи `lint` і `pipeline` завершилися зі статусом success. Навчання та перевірка виконувалися всередині Docker-контейнерів. Через вичерпання квоти Actions artifact storage результати зберігалися у спільному workspace одного job-а та відображаються в логах і GitHub Job Summary.

| Показник | Результат CI |
|---|---:|
| Статус workflow | Success |
| GitHub Actions run | №9 |
| Commit | `183e5dfb75d034118bb317b0ea294fdf1a153a89` |
| Runner | `ubuntu-runner`, Linux x64, self-hosted |
| Версія моделі | `3.0.9` |
| Датасет | Google Speech Commands |
| Класи | `yes`, `no`, `up`, `down` |
| Архітектура | `SpeechCNN` |
| Епохи / seed | 4 / 42 |
| Навчальна / validation / test вибірки | 200 / 200 / 200 аудіозаписів |
| Test accuracy | 51,0% |
| Розмір файлу ваг | 102 235 B (99,84 KiB) |
| Час навчання | 28,53 s |
| Server latency, mean | 5,34 ms |
| Client latency, p50 / p95 | 9,06 / 10,74 ms |
| Runtime Docker image | `ghcr.io/hellndodger/integration-2:183e5dfb75d034118bb317b0ea294fdf1a153a89` |
| Розмір runtime image | 1,46 GB |
| Публікація до GHCR | Так |
| Quality gates | Пройдені |

У benchmark виконано 5 warm-up запитів і 30 вимірюваних запитів. Перевірка API підтвердила доступність `/health`, відповідність метаданих моделі тренувальним метрикам, коректну схему прогнозу та проходження порогів accuracy ≥45% і client p95 ≤5000 ms.

| Golden WAV | Очікуваний клас | Прогноз | Confidence | Latency |
|---|---|---|---:|---:|
| `yes.wav` | yes | yes | 0,3296 | 5,46 ms |
| `no.wav` | no | no | 0,2953 | 5,27 ms |
| `up.wav` | up | up | 0,3734 | 6,47 ms |
| `down.wav` | down | down | 0,2948 | 4,74 ms |

Усі 4 контрольні приклади з офіційного testing split класифіковано правильно. Перевірку виконано в API, розгорнутому з runtime-образу.

## 7. Скріншоти результатів

Зробити скриншоти після завершення workflow у GitHub → **Actions** → **ML CI/CD pipeline** → потрібний запуск. У звіт вставити зображення з підписами:

1. **Рисунок 1 — Успішне виконання перевірки коду в GitHub Actions.** На сторінці run №9 показати job `lint` зі статусом success та успішним кроком Ruff і compileall.
2. **Рисунок 2 — Навчання SpeechCNN у Docker-контейнері.** У job `pipeline` показати завершення `Train and evaluate inside Docker` та підсумкові метрики: accuracy 51,0%, 4 епохи, час навчання 28,53 s.
3. **Рисунок 3 — Перевірка REST API та quality gates.** У логах job `pipeline` показати успішні перевірки `/health`, чотирьох golden predictions, accuracy gate і latency gate.
4. **Рисунок 4 — Вимірювання затримки API.** У кроці `Measure latency in the runtime image` показати 5 warm-up і 30 запитів, server mean 5,34 ms та client p50/p95 9,06/10,74 ms.
5. **Рисунок 5 — Підсумок CI/CD у GitHub Job Summary.** На вкладці **Summary** запуску №9 показати таблицю метрик і результати golden-sample перевірки.
6. **Рисунок 6 — Розмір runtime-образу та історія шарів Docker.** У логах job `pipeline` показати команди `docker images` і `docker history`.
7. **Рисунок 7 — Опублікований Docker-образ у GitHub Container Registry.** У job `pipeline` показати успішний крок `Publish verified image`; у розділі **Packages** репозиторію показати образ і тег commit SHA.

Окремо збережіть сторінку run №9 із видимими commit SHA та фінальним статусом **Success**. У workflow артефакти не завантажуються до Actions artifact storage, тому окремої секції з `training-results` або `pipeline-report` немає. Не використовуйте скриншот YAML як доказ успішного запуску. Fork pull request має бути пропущений за правилом безпеки, а не показаний як успішно виконаний.

## 8. Висновки

Реалізовано та успішно виконано Docker-based CI/CD-процес для аудіокласифікатора: перевірку коду, навчання, оцінювання, підготовку контрольних аудіофайлів, інтеграційне тестування API, вимірювання latency, застосування quality gates, збирання та публікацію runtime-образу в GHCR. У фінальному запуску №9 модель досягла test accuracy 51,0%; усі 4 golden-приклади класифіковано правильно, client p95 latency становила 10,74 ms, а quality gates пройдено.

Накладні витрати контейнеризації проявляються насамперед у розмірі образу та стартовому використанні пам’яті: runtime містить Python, PyTorch, torchaudio й системну бібліотеку для аудіо. Trainer image більший, бо додає навчальні залежності. Розділення на targets і виключення датасету з фінального образу не прибирають великі runtime-залежності, але не переносять до сервісу навчальний набір даних і зайві артефакти.

Фінальний runtime-образ має розмір 1,46 GB і опублікований у GHCR. Для порівняння, історичний single-stage baseline ЛР2 мав розмір 12,4 GB, тоді як multi-stage runtime-образ — 1,46 GB, тобто приблизно на 88,2% менше. Це порівняння наведено як попередній локальний результат ЛР2, а не як повторне вимірювання в CI ЛР3.
