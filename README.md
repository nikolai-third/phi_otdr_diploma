# phi-OTDR Data Audit

Python-проект для ревизии датасета phi-OTDR с безопасным извлечением метаданных без загрузки больших файлов целиком в RAM.

## Структура проекта

- `data/raw` — symlink на WebDAV (`/Volumes/webdav.yandex.ru/phi-OTDR`)
- `data/interim` — каталоги/сводки аудита
- `data/processed` — производные данные
- `reports/figures`, `reports/tables` — артефакты отчётов
- `notebooks` — исследовательские ноутбуки
- `src/audit` — CLI и логика сканера
- `tests` — тесты `pytest`

## Требования

- Python 3.10+
- Зависимости: `numpy`, `pandas`, `pyarrow`, `tqdm`, `h5py`, `PyYAML`, `pytest`

## Установка

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

## Запуск аудита

Базовая команда:

```bash
python -m src.audit scan --root data/raw --out data/interim/catalog.parquet
```

Параметры:

- `--max-files` — ограничить число файлов (например, 200 для быстрой проверки)
- `--sample-bytes` — сколько байт читать для magic header (default: `64`)
- `--qc-max-bytes` — лимит на объём выборки для QC (default: `5000000`)
- `--workers` — число потоков чтения метаданных (default: `4`)
- `--parquet-schema-max-bytes` — лимит размера parquet, после которого schema-read пропускается (default: `2000000`)

Пример быстрой проверки:

```bash
python -m src.audit scan \
  --root data/raw \
  --out data/interim/catalog.parquet \
  --max-files 200 \
  --workers 4
```

## Что создаётся

- `data/interim/catalog.parquet` — основной каталог
- `data/interim/catalog.csv` — облегчённый просмотр
- `data/interim/summary.md` — человекочитаемая сводка

## Поддерживаемые форматы метаданных

- `.h5/.hdf5`: список datasets, shape/dtype, root attrs, безопасный QC-сэмпл
- `.npy`: shape/dtype, безопасный QC-сэмпл
- `.npz`: список массивов с shape/dtype
- `.json/.yaml/.yml`: ключи (для небольших файлов)
- `.txt`: автоопределение кодировки (включая `cp1251/cp866`) + превью текста и key/value-пары
- `.csv`: колонки (header-only)
- `.parquet`: схема и типы колонок
- Прочие бинарники: magic bytes (`sample-bytes`) и MIME guess

## WebDAV: ограничения и рекомендации

- WebDAV имеет высокую латентность на множество мелких запросов.
- Используйте умеренный `--workers` (обычно `2..6`) и `--max-files` для первых прогонов.
- Не увеличивайте `--qc-max-bytes` без необходимости.
- Для повторяющихся экспериментов полезно кэшировать нужные подмножества локально в `data/interim`/`data/processed`.

## Тесты

```bash
pytest
```
