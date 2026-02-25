# Бакалаврский диплом: phi-OTDR

Репозиторий для дипломной работы по теме:
**«Исследование методов обработки данных в фазочувствительной рефлектометрии на длинных оптических линиях с усилителями»**.

`dataset audit` и `baseline pipeline` — воспроизводимые этапы первичной обработки и контроля качества всего массива данных.

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

Парсер для одного файла находится в `src/parser/one_file.py` и запускается через:

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

4. Автооценка длины рефлектограммы (если `--trace-len` не задан):
   - из отсортированных кандидатов `c_i` берутся разности `δ_i = c_{i+1} - c_i`;
   - для каждой `δ_i` перебираются гармоники `k=1..K`, гипотезы периода `p_{i,k}=δ_i/k`;
   - голоса взвешиваются как `1/sqrt(k)` (штраф за высокие гармоники);
   - максимум взвешенной гистограммы + локальная медиана дают оценку `L_hat` (в точках).

5. Фильтрация и восстановление пропусков (`--fill-missing`):
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
   - в окне `[a, a+W)` для каждой трассы `T_i`:
     - `τ_i = argmax_{|τ|<=S} corr(T_0[a:a+W], T_i[a+τ:a+W+τ])`;
   - выравнивание применяется только если метрика residual jitter уменьшается.

8. Временные пересчеты:
   - время точки: `t_us(n) = 1e6 * n / Fs`;
   - длительность рефлектограммы: `T_trace_us = 1e6 * L_hat / Fs`.

### Основные режимы

- `v1` (по умолчанию): максимальная стабильность выравнивания, но может находить меньше стартов.
- `v1 + recovery` (`--fill-missing`): значительно выше полнота по числу стартов при сохранении приемлемого jitter.

### Важные параметры

- `--adc-fs-hz` — частота АЦП (в проекте: `50000000` Гц).
- `--trace-len` — длина одной рефлектограммы в точках (опционально; по умолчанию оценивается из сигнала).
- `--max-samples` — ограничение по числу точек для быстрого теста.
- `--max-traces` — верхняя граница числа извлекаемых трасс.
- `--waterfall-cmap` — цветовая схема heatmap (по умолчанию `jet`, как в классических phi-OTDR waterfall).
- `--waterfall-exp-alpha` — коэффициент экспоненциального контрастирования (поднимает низкие уровни).

### Ключевые метрики в `parser_diagnostics.md`

- `n_detected_starts` — найдено стартов.
- `expected_traces` и `coverage_ratio` — полнота обнаружения.
- `trace_len_source` — источник длины (`manual` или `inferred`).
- `residual_before_abs_mean` / `residual_after_abs_mean` — остаточный джиттер до/после выравнивания.
- `alignment_applied` — применилось ли выравнивание (1/0).

### Примеры

Базовый стабильный режим:

```bash
python -m src.parser.one_file \
  --file data/raw/05_10_2024/2024-10-05_00_00.parquet \
  --outdir reports/figures/parser_v1 \
  --adc-fs-hz 50000000 \
  --max-samples 12000000
```

Режим с повышенной полнотой:

```bash
python -m src.parser.one_file \
  --file data/raw/05_10_2024/2024-10-05_00_00.parquet \
  --outdir reports/figures/parser_v1_plus \
  --adc-fs-hz 50000000 \
  --max-samples 12000000 \
  --fill-missing
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
