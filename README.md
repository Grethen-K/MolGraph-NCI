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

## Настройка правил A/B/C

Правила A, B и C определяют пороговые значения дистанции и угла при детектировании NCI.  
Изменить их можно в следующих файлах:

### 1. Водородные связи (HB)

**Файл:** `hb_detectors.py` → функция `apply_hb_rules()`

**Как менять:**  
Найдите блок (строки ~65–75):
```python
if rule == 'A':
    d_max = 3.25    # Макс. дистанция H···A
    a_min = 0
elif rule == 'B':
    d_max = vdw_sum    # Макс. дистанция H···A
    a_min = 130        # Мин. угол D–H···A   ()
elif rule == 'C':
    d_max = vdw_sum - 0.2    # Макс. дистанция H···A = `vdw_sum - 0.2` Å 
    a_min = 150

### 2. σ-дырочные связи (XB, ChB, PnB)

**Файл:** `sigma_detectors.py` → функция `apply_sigma_rules()`

**Как менять:** 

### 2.1. Списки элементов-доноров и акцепторов

**Строки ~21–22: можно добавить Ваши элементы**

acceptors = ['O', 'N', 'F', 'S', 'Cl', 'Br', 'I', 'P', 'As', 'Se']
sigma_donors = ['F', 'Cl', 'Br', 'I', 'At', 'O', 'S', 'Se', 'Te', 'N', 'P', 'As', 'Sb', 'Bi']

### 2.2. Пороги дистанции и угла для правил A/B/C

**Строки ~58–70: аналогично HB

if rule == 'A':
    d_max = 4.14
    angle_1_min = 0.0
elif rule == 'B':
    d_max = vdw_sum
    angle_1_min = 110.0
elif rule == 'C':
    d_max = vdw_sum * 0.9
    angle_1_min = 130.0

## Лицензия

MIT
