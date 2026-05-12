# Бакалаврский диплом: phi-OTDR

Репозиторий для дипломной работы по теме:
**«Исследование методов обработки данных в фазочувствительной рефлектометрии на длинных оптических линиях с усилителями»**.

`dataset audit` и `baseline pipeline` — воспроизводимые этапы первичной обработки и контроля качества всего массива данных.

## Сам диплом

- [thesis/main.pdf](thesis/main.pdf) — собранный PDF (финальная версия, 36 страниц).
- [thesis/main.tex](thesis/main.tex) — исходник на XeLaTeX (Times New Roman, поля 30/15/20/20 мм, шрифт 12 пт, межстрочный 1.5 — по Положению о ВКР МФТИ, Приложение 2).
- [thesis/figures/](thesis/figures/) — рисунки для главы 3-4 (рефлектограмма, водопадная диаграмма, спектральная карта, компоненты оценки $S(g)$, гистограммы и сравнительные диаграммы).
- [thesis/THESIS_OUTLINE.md](thesis/THESIS_OUTLINE.md), [thesis/THESIS_PLAN.md](thesis/THESIS_PLAN.md) — план работы и литературный обзор (рабочие документы, в PDF не входят).

### Сборка PDF

```bash
# 1. Перегенерировать рисунки из исходного датасета
#    (требует примонтированной флешки /Volumes/data/phi-OTDR/)
.venv/bin/python scripts/make_thesis_figures.py

# 2. Скомпилировать диплом (XeLaTeX, два прохода для ссылок)
cd thesis
xelatex -interaction=nonstopmode main.tex
xelatex -interaction=nonstopmode main.tex
```

Скрипт [scripts/make_thesis_figures.py](scripts/make_thesis_figures.py) собирает все рисунки диплома: 4 диаграммы из CSV в `reports/tables/` (метрики, гистограмма ошибки локализации, сравнение детекторов, Recall по импульсам) и 7 диаграмм с эталонной размеченной записи `aligned.npz` (рефлектограмма, водопад, фон+MAD, нормированный водопад, FFT-карта, компоненты $S(g)$, финальный детект). Подписи осей и легенды на русском.

### Список литературы

Оформлен по ГОСТ Р 7.0.100-2018 (национальный стандарт РФ для библиографических записей): инвертированный заголовок «Фамилия, И. О.», полная зона ответственности после `/`, предписанная пунктуация `. -`, для 5+ авторов — первые три + `[et al.]`. В списке 25 источников: обзорные работы по DOFS и DAS, foundational phi-OTDR (Posey 2000, Juarez 2005), длиннолинейные демонстрации с EDFA/Raman-усилением (Peng, Martins, Wang Z. N., Tian, Wang Y., Fan), методы обработки сигналов phi-OTDR (CWT, HHT, вейвлет-пакеты, матричное сопоставление), event recognition (Tejedor, Wu, Shi), методическая ссылка на устойчивую к выбросам статистику (Hampel 1974). Все DOI проверены через Crossref API.

## Презентация защиты

В директории [slides/](slides/):
- `presentation_phi_otdr.pdf` / `.tex` — слайды защиты (Beamer).
- `presentation_speaker_notes.pdf` / `.tex` — заметки докладчика построчно к слайдам.

## Данные и ограничения

- `data/raw` — symlink на WebDAV: `/Volumes/webdav.yandex.ru/phi-OTDR`
- Объём датасета большой (десятки ГБ)
- Чтение только ленивое/выборочное
- Для устойчивости используется локальный кэш `cache/`

## Архитектура pipeline

- `src/audit` — аудит и каталог файлов (`catalog.parquet`)
- `src/index/prepare.py` — построение processing-index (`index.parquet`)
- `src/utils/cache.py` — `ensure_local(path, cache_dir)` для локального кеширования файлов
- `src/baseline/run.py` — массовая baseline-аналитика с checkpoint/resume
- `src/baseline/artifacts.py` — генерация графиков/таблиц/markdown summary
- `src/pipeline.py` — единый CLI `run-all`
- `src/utils/logging_config.py` — глобальное логирование в консоль и `logs/pipeline.log`
- `src/parser/detect_from_aligned.py` — пост-детекция возмущений по выровненным рефлектограммам

## Структура директорий

- `cache/` — локальные копии файлов с WebDAV
- `logs/` — `pipeline.log` и `errors.log`
- `data/interim/` — `catalog.parquet`, `index.parquet`
- `data/processed/` — `baseline_metrics.parquet`
- `reports/figures/` — графики baseline
- `reports/tables/` — csv-таблицы baseline

## Установка

```bash
python3 -m venv .venv --system-site-packages
source .venv/bin/activate
# При доступе в интернет:
# pip install -e .[dev]
```

## Основные команды

1. Построить индекс (если нужен вручную):

```bash
python -m src.index build --catalog data/interim/catalog.parquet --out data/interim/dataset_index.parquet
```

2. Полный baseline pipeline:

```bash
python -m src.pipeline run-all
```

CLI поддерживает параметры:
- `--max-workers` (не более 4)
- `--max-bytes` (объём сэмпла на файл)
- `--checkpoint-every` (по умолчанию 50)

## Парсер рефлектограмм

Парсер разбит на модули:
- `src/parser/core.py` — детекция стартов, оценка периода и извлечение рефлектограмм.
- `src/parser/templates.py` — шаблонное доуточнение стартов и кросс-корреляционное выравнивание.
- `src/parser/io.py` — потоковое чтение parquet, построение графиков и запись диагностик.
- `src/parser/one_file.py` — CLI-обёртка для запуска на одном файле.
- `src/parser/batch_usb.py` — пакетный прогон по `data/raw_usb` с сохранением в `data/processed_usb/parser_cache`.

Запуск:

```bash
python -m src.parser.one_file --file <path_to_file.parquet> --outdir reports/figures/parser_run
```

### Математическая схема парсера

Пусть дискретный сигнал `x[n]`, `n=0..N-1`, частота дискретизации `Fs` (в проекте `50 МГц`).

1. Предобработка на прореженной сетке:
   - прореживание: `y[k] = x[kd]`, где `d = decimation`;
   - скользящее среднее длины `w`: `s[k] = (1/w) * Σ_{i=0}^{w-1} y[k-i]`;
   - экспоненциальные огибающие:
     - верхняя `u[k] = max(s[k], αu[k-1] + (1-α)s[k])`;
     - нижняя `l[k] = min(s[k], αl[k-1] + (1-α)s[k])`;
   - центральная линия: `m[k] = 0.5 * (u[k] + l[k])`.

2. Детекция кандидатов стартов:
   - робастный динамический диапазон: `Δ = Q98(m) - Q02(m)`;
   - пороги:
     - `θ_low = Q02(m) + β_low * Δ`;
     - `θ_rise = β_rise * Δ`;
   - градиент: `g[k] = 0.5 * (m[k+1] - m[k-1])`;
   - кандидат старта: `m[k-1] < θ_low` и `g[k] > θ_rise`.

3. Уточнение кандидатов в исходной сетке:
   - проекция `k -> n0 = k*d`;
   - локальный поиск в окне `±r`:  
     `n* = argmax_{n in [n0-r, n0+r]} (x[n+1]-x[n-1])/2`.

4. Автооценка длины рефлектограммы:
   - из отсортированных кандидатов `c_i` берутся разности `δ_i = c_{i+1} - c_i`;
   - для каждой `δ_i` перебираются гармоники `k=1..K`, гипотезы периода `p_{i,k}=δ_i/k`;
   - голоса взвешиваются как `1/sqrt(k)` (штраф за высокие гармоники);
   - максимум взвешенной гистограммы + локальная медиана дают оценку `L_hat` (в точках).

5. Фильтрация и восстановление пропусков:
   - базовый отбор по минимальному шагу: `c_{i+1}-c_i >= γ*L_hat`;
   - якоря `a_j` интерполируются между собой:
     - `k_j = round((a_{j+1}-a_j)/L_hat)`;
     - шаг `h_j = (a_{j+1}-a_j)/k_j`;
     - вставки `a_j + m*h_j`, `m=1..k_j-1`;
   - затем локальное доуточнение каждой вставки тем же `argmax` градиента;
   - финальный spacing-guard: соседние старты не ближе `ρ*L_hat`.

6. Извлечение трасс:
   - `T_i[r] = x[s_i + r]`, `r=0..L_hat-1`.

7. Выравнивание кросс-корреляцией:
   - сначала оцениваются попарные лаги между соседними трассами в окне `[a, a+W)`;
   - лаги робастно регуляризуются (подавление выбросов и дрейфа), затем интегрируются в абсолютные сдвиги;
   - выравнивание применяется только если улучшаются метрики residual jitter/roughness.

8. Временные пересчеты:
   - время точки: `t_us(n) = 1e6 * n / Fs`;
   - длительность рефлектограммы: `T_trace_us = 1e6 * L_hat / Fs`.

### Важные параметры

- `--adc-fs-hz` — частота АЦП (в проекте: `50000000` Гц).
- `--waterfall-cmap` — цветовая схема heatmap (по умолчанию `jet`).
- `--waterfall-exp-alpha` — коэффициент экспоненциального контрастирования (поднимает низкие уровни).

Парсер всегда читает и обрабатывает **весь файл целиком**.
Длина рефлектограммы (`trace_len`) и число извлечённых рефлектограмм (`n_extracted_traces`) вычисляются автоматически из сигнала.

## Пост-детекция по выровненным данным

Модуль: `src/parser/detect_from_aligned.py`

Вход: `aligned.npz` из парсера (`data/processed_usb/parser_cache/records/<record>/aligned.npz`).

Шаги:
1. Робастный фон по времени (`median`) и residual.
2. Нормировка residual по `MAD` для каждого distance-bin.
3. FFT-карта по всей полосе частот (агностично к конкретной частоте воздействия).
4. Комбинированный score: спектральные пики + широкополосная энергия + time-domain energy.
5. Ограничение зоны поиска по `usable_end_km` (автооценка конца полезного сигнала).
6. Экспорт пороговых кандидатов и top-candidate в usable-зоне.

Ключевое поведение по умолчанию:
- обрабатывается **весь файл** (`--max-traces` не задан);
- crop по стабильному сегменту **выключен** (`--use-stable-segment` не указан).

### Ключевые метрики в `parser_diagnostics.md`

- `n_detected_starts` — найдено стартов.
- `expected_traces` и `coverage_ratio` — полнота обнаружения.
- `trace_len` — оцененная длина рефлектограммы в точках.
- `residual_before_abs_mean` / `residual_after_abs_mean` — остаточный джиттер до/после выравнивания.
- `alignment_applied` — применилось ли выравнивание (1/0).

### Примеры

Базовый режим (auto):

```bash
python -m src.parser.one_file \
  --file data/raw/05_10_2024/2024-10-05_00_00.parquet \
  --outdir reports/figures/parser_v1 \
  --adc-fs-hz 50000000
```

Детекция по уже выровненному файлу:

```bash
python -m src.parser.detect_from_aligned \
  --aligned-npz data/processed_usb/parser_cache/records/some_test/2024-11-11_13_06/aligned.npz \
  --outdir data/processed_usb/parser_cache/records/some_test/2024-11-11_13_06/detection_v2_full
```

Опционально (для эксперимента): ограничить обработку самым стабильным временным сегментом:

```bash
python -m src.parser.detect_from_aligned \
  --aligned-npz <path_to_aligned.npz> \
  --outdir <outdir> \
  --use-stable-segment
```

## Поведение run-all

- создаёт нужные директории (`cache`, `logs`, `data/processed`, `reports/*`)
- создаёт `data/interim/index.parquet`, если его нет
- обрабатывает только не обработанные ранее `record_id`
- сохраняет результаты инкрементально в `data/processed/baseline_metrics.parquet`
- пишет ошибки отдельных файлов в `logs/errors.log`, не останавливая pipeline
- генерирует:
  - `reports/figures/hist_file_sizes.png`
  - `reports/figures/hist_std_distribution.png`
  - `reports/figures/format_distribution.png`
  - `reports/figures/scatter_size_vs_std.png`
  - `reports/tables/top20_largest_files.csv`
  - `reports/summary_baseline.md`
