# 🧠 EEG Movement Classification Pipeline

Система классификации движений по EEG данным с использованием нейронных сетей TensorFlow.

## 📋 Описание

Этот проект реализует полный пайплайн для классификации движений головы/глаз по данным ЭЭГ:

### 🎯 Классы движений:
- **Up** - движение вверх
- **Down** - движение вниз  
- **Left** - движение влево
- **Right** - движение вправо
- **Back** - движение назад
- **Forward** - движение вперед
- **None** - состояние покоя

### 🔧 Архитектура:
- **Входные данные**: 15 эпох × 20 признаков
- **Модель**: Глубокая нейронная сеть TensorFlow
- **Каналы ЭЭГ**: C3, Cz, C4, FCz, Fz, F3, F4, FC3, FC4, F7, F8, T7, T4, TP8, T5, PO7, PO3, PO4, PO8, P3, P4, Pz
- **Метрика**: F1 Score

## 🚀 Быстрый старт

### 1. Установка зависимостей

```bash
pip install -r requirements_eeg.txt
```

### 2. Подготовка данных

Поместите EDF файлы в следующую структуру:
```
data/
└── Russian/
    ├── sub-1/
    │   └── sub-1.edf
    ├── sub-2/
    │   └── sub-2.edf
    └── ...
```

### 3. Запуск полного пайплайна

```bash
python run_eeg_pipeline.py
```

Выберите опцию "1" для запуска полного пайплайна.

## 📁 Структура проекта

```
├── run_eeg_pipeline.py      # 🎮 Главный файл запуска
├── eeg_preprocessing.py     # 🔧 Препроцессинг EDF данных  
├── train_eeg_classifier.py  # 🤖 Обучение нейронной сети
├── predict_eeg.py           # 🔮 Инференс и предсказания
├── requirements_eeg.txt     # 📦 Зависимости
└── best_models/             # 💾 Сохраненные модели
```

## 🔄 Этапы пайплайна

### 1️⃣ Препроцессинг (`eeg_preprocessing.py`)

**Что делает:**
- Загружает EDF файлы из `data/Russian/sub-*/`
- Извлекает данные из нужных каналов ЭЭГ
- Группирует аннотации в целевые классы (UP1/UP2 → Up)
- Извлекает признаки из каждой эпохи:
  - Среднее значение сигнала
  - Медиана/стандартное отклонение
- Создает последовательности по 15 эпох
- Сохраняет в `processed_eeg_data.pkl`

**Признаки (20 штук):**
Для каждого из важных каналов извлекается 2 признака, до общего количества 20.

### 2️⃣ Обучение (`train_eeg_classifier.py`)

**Архитектура модели:**
```python
model = Sequential([
    Flatten(input_shape=(15, 20)),  # 15 эпох × 20 признаков
    Dense(16, activation='relu'),
    Dropout(0.3),
    Dense(32, activation='relu'), 
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(256, activation='relu'),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.3), 
    Dense(16, activation='relu'),
    Dense(7, activation='softmax')  # 7 классов
])
```

**Параметры обучения:**
- Оптимизатор: `nadam`
- Функция потерь: `categorical_crossentropy`
- Метрики: `accuracy`, `precision`, `recall`
- Batch size: 16
- Epochs: 100 (с early stopping)

**Результаты:**
- Модель сохраняется в `best_models/final_eeg_model.h5`
- Объекты препроцессинга в `best_models/preprocessing_objects.pkl`
- Графики обучения и матрица ошибок

### 3️⃣ Предсказания (`predict_eeg.py`)

**Что делает:**
- Загружает обученную модель
- Обрабатывает новые EDF файлы
- Выводит предсказания с уверенностью
- Сравнивает с исходными аннотациями

## 📊 Результаты

После обучения вы получите:

1. **Метрики модели:**
   - F1 Score (macro/micro/weighted)
   - Accuracy, Precision, Recall
   - Confusion Matrix

2. **Визуализации:**
   - `training_history.png` - графики обучения
   - `confusion_matrix.png` - матрица ошибок

3. **Сохраненные файлы:**
   - `best_models/final_eeg_model.h5` - обученная модель
   - `processed_eeg_data.pkl` - предобработанный датасет

## 🔧 Кастомизация

### Изменение архитектуры модели

Отредактируйте метод `create_model()` в `train_eeg_classifier.py`:

```python
def create_model(self):
    model = Sequential([
        # Ваша архитектура здесь
    ])
    return model
```

### Добавление новых признаков

Измените метод `extract_features_from_epoch()` в `eeg_preprocessing.py`:

```python
def extract_features_from_epoch(self, epoch_data):
    features = []
    for channel_data in epoch_data:
        # Добавьте новые признаки здесь
        features.extend([...])
    return np.array(features)
```

### Настройка классов

Измените `class_mapping` в `EEGPreprocessor.__init__()`:

```python
self.class_mapping = {
    'YourClass': ['ANNOTATION1', 'ANNOTATION2'],
    # ...
}
```

## 🐛 Устранение проблем

### Ошибка "No module named..."
```bash
pip install -r requirements_eeg.txt
```

### Ошибка "EDF файлы не найдены"
Убедитесь в правильной структуре папок:
```
data/Russian/sub-N/sub-N.edf
```

### Ошибка "Недостаточно эпох"
Убедитесь, что в EDF файлах есть аннотации и достаточно данных.

### Проблемы с памятью
Уменьшите `batch_size` в `train_eeg_classifier.py`:
```python
batch_size=8  # вместо 16
```

## 📈 Рекомендации по улучшению

1. **Больше данных** - добавьте больше EDF файлов
2. **Аугментация** - добавьте шум, сдвиги времени  
3. **Feature engineering** - попробуйте частотные признаки (FFT, PSD)
4. **Архитектура** - эксперименты с CNN, LSTM, Transformer
5. **Гиперпараметры** - настройка learning rate, dropout, batch size

## 📚 Дополнительная информация

- **MNE Documentation**: https://mne.tools/
- **TensorFlow Guide**: https://www.tensorflow.org/guide
- **EEG Analysis**: https://en.wikipedia.org/wiki/Electroencephalography

---

**Автор**: AI Assistant  
**Версия**: 1.0  
**Лицензия**: MIT
