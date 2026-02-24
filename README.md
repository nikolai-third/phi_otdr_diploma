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

### Что делает парсер

1. Лениво читает колонку `data` из parquet (потоково, без загрузки всего файла в RAM).
2. Находит старты рефлектограмм:
   - сглаживание + верхняя/нижняя огибающие;
   - поиск участков «низкий уровень -> резкий фронт»;
   - локальное уточнение старта по градиенту.
3. Опционально восстанавливает пропущенные старты (`--fill-missing`) по периодической структуре между надежными якорями.
4. Извлекает трассы фиксированной длины (`--trace-len`).
5. Пробует выравнивание кросс-корреляцией и применяет его только если residual jitter улучшается.
6. Сохраняет графики и диагностику.

### Основные режимы

- `v1` (по умолчанию): максимальная стабильность выравнивания, но может находить меньше стартов.
- `v1 + recovery` (`--fill-missing`): значительно выше полнота по числу стартов при сохранении приемлемого jitter.

### Важные параметры

- `--adc-fs-hz` — частота АЦП (в проекте: `50000000` Гц).
- `--trace-len` — длина одной рефлектограммы в точках.
- `--max-samples` — ограничение по числу точек для быстрого теста.
- `--max-traces` — верхняя граница числа извлекаемых трасс.

### Ключевые метрики в `parser_diagnostics.md`

- `n_detected_starts` — найдено стартов.
- `expected_traces` и `coverage_ratio` — полнота обнаружения.
- `residual_before_abs_mean` / `residual_after_abs_mean` — остаточный джиттер до/после выравнивания.
- `alignment_applied` — применилось ли выравнивание (1/0).

### Примеры

Базовый стабильный режим:

```bash
python -m src.parser.one_file \
  --file data/raw/05_10_2024/2024-10-05_00_00.parquet \
  --outdir reports/figures/parser_v1 \
  --trace-len 55000 \
  --adc-fs-hz 50000000 \
  --max-samples 12000000
```

Режим с повышенной полнотой:

```bash
python -m src.parser.one_file \
  --file data/raw/05_10_2024/2024-10-05_00_00.parquet \
  --outdir reports/figures/parser_v1_plus \
  --trace-len 55000 \
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
