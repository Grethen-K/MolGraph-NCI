# MolGraph-NCI

Molecular Graph Builder with Non-Covalent Interactions

## Описание

MolGraph-NCI — инструмент для построения молекулярных графов с учётом нековалентных взаимодействий (водородные связи и σ-дырочные взаимодействия) из XYZ-файлов.

## Установка

```bash
git clone https://github.com/Grethen-K/MolGraph-NCI.git
cd MolGraph-NCI
pip install -r requirements.txt
```

## Использование

### Анализ одной структуры
```bash
python main.py test/molecule.xyz --mode single --rule B --inter HB
```

### Сравнение правил A/B/C
```bash
python main.py test/molecule.xyz --mode compare --inter HB
```

### Batch-обработка
```bash
python main.py test/ --mode batch --workers 4
```

### Визуализация
```bash
# Из JSON-результата
python visualizer.py results/molecule_HB_B.json --mode structure

# Из XYZ напрямую
python visualizer.py test/molecule.xyz --format xyz --mode structure --rule B

# Дашборд
python visualizer.py results/molecule_HB_B.json --mode dashboard
```

## Структура проекта

- `main.py` — точка входа CLI
- `graph_builder.py` — построение графа
- `hb_detectors.py` — детекция водородных связей
- `sigma_detectors.py` — детекция σ-взаимодействий
- `visualizer.py` — визуализация (Plotly + matplotlib fallback)
- `metrics.py` — статистика
- `batch_processor.py` — batch-обработка

## Лицензия

MIT
