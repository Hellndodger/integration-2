# Лабораторна робота №3 — CI/CD для аудіокласифікатора

Робочий процес GitHub Actions розгортає модель із ЛР1/ЛР2 у два job-и:

1. `lint` — Ruff і перевірка синтаксису.
2. `pipeline` — послідовне навчання в Docker, підготовка golden samples, перевірка API й latency, quality gates, збірка та аналіз runtime-образу, формування Job Summary і, за потреби, публікація в GHCR.

Навчальні й verification-файли залишаються у workspace одного job-а; workflow не використовує GitHub Actions artifact storage. Це усуває залежність від квоти артефактів і зайві передачі файлів між job-ами.

## Безпека self-hosted runner-а

Репозиторій публічний, тому jobs для pull request із fork автоматично пропускаються: довільний код fork не виконується на приватному self-hosted runner-і. Pull request із гілки цього самого репозиторію може виконати pipeline. Запускайте runner лише для довірених змін, не зберігайте на ньому секрети чи доступ до чутливих систем. Тригер `workflow_dispatch` дозволяє запуск вручну з GitHub Actions.

## Локальне виконання у Docker

Команди нижче для PowerShell із кореня репозиторію; потрібні Docker Desktop із Linux containers і розпакований датасет `lab1/data/speechcommands`. Образи базуються на `python:3.11-slim-bookworm`, щоб не залежати від CPU-вимог Python tool-cache self-hosted runner-а.

```powershell
New-Item -ItemType Directory -Force lab3\artifacts | Out-Null
docker build --target trainer -t speech-lab3-trainer:local -f lab3\Dockerfile .
docker run --rm `
  -v "${PWD}\lab1\data\speechcommands:/data:ro" `
  -v "${PWD}\lab3\artifacts:/artifacts" `
  speech-lab3-trainer:local `
  --data-root /data --output-dir /artifacts --epochs 4 --limit-per-class 50 `
  --seed 42 --model-version 1.0.0 --git-commit local
```

`lab3/golden/expected.json` перелічує невикористані в навчанні аудіофайли з офіційного testing split. WAV-файли не додаються до репозиторію: скрипт перевіряє їхню приналежність до testing split і копіює до каталогу артефактів.

```powershell
python lab3\scripts\prepare_golden.py `
  --data-root lab1\data\speechcommands `
  --output lab3\artifacts\golden
docker build --target runtime -t speech-lab3-api:local -f lab3\Dockerfile .
docker run --rm -d --name speech-lab3-api -p 8000:8000 speech-lab3-api:local
python -m pip install requests
python lab2\scripts\benchmark.py --url http://127.0.0.1:8000 --n 30 --warmup 5 `
  --sample lab3\artifacts\golden\yes.wav --out lab3\artifacts\latency.json
python lab3\scripts\verify.py --samples lab3\artifacts\golden
docker rm -f speech-lab3-api
```

## Отримання скриншотів для звіту

Після успішного workflow відкрийте GitHub → **Actions** → **ML CI/CD pipeline** → потрібний запуск:

1. **Lint:** відкрийте job `lint`; зніміть успішні Ruff і compile кроки.
2. **Навчання:** відкрийте job `pipeline`; зніміть Docker build/run та заключний JSON із метриками.
3. **Перевірка API:** у job `pipeline` зафіксуйте успішні health/golden/quality checks і p95 latency з логів.
4. **Job Summary:** на сторінці запуску відкрийте **Summary**; зніміть таблицю метрик та golden predictions.
5. **Образ:** у job `pipeline` зніміть вивід `docker images`/розмір і `docker history`; якщо `Published to GHCR` дорівнює true, додатково відкрийте **Packages** у репозиторії та зніміть тег образу.
6. **Тригери:** для звіту зробіть окремі успішні запуски від push у `main`, `workflow_dispatch` і довіреного same-repository pull request. Не демонструйте fork PR як виконаний: він має бути skipped згідно з правилом безпеки.

GitHub Actions artifact storage не використовується; результати показуються у логах і Job Summary. Не підмінюйте відсутні запуски скриншотами workflow-файлу: у звіті вказуйте лише фактично завершені події.
