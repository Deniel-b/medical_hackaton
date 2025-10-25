#!/usr/bin/env python3
"""
Скрипт для сохранения данных графиков обучения в текстовом формате
"""

import pickle
import numpy as np
import os
import json
from datetime import datetime

def save_training_data_to_files():
    """Сохранить данные обучения в различных форматах для анализа"""
    
    # Найти последний файл истории
    history_files = [f for f in os.listdir('.') if f.startswith('training_history_') and f.endswith('.pkl')]
    
    if not history_files:
        print("❌ Файлы истории обучения не найдены!")
        return
    
    # Взять последний файл
    latest_file = max(history_files, key=lambda x: os.path.getmtime(x))
    print(f"📁 Обрабатываю файл: {latest_file}")
    
    # Загрузить данные
    with open(latest_file, 'rb') as f:
        history = pickle.load(f)
    
    epochs_count = len(history['loss'])
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. JSON формат - для программной обработки
    json_data = {}
    for key, values in history.items():
        json_data[key] = [float(v) for v in values]  # Конвертируем в float для JSON
    
    json_filename = f"training_data_{epochs_count}epochs_{timestamp}.json"
    with open(json_filename, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    print(f"💾 JSON данные сохранены: {json_filename}")
    
    # 2. CSV формат - для Excel/анализа
    csv_filename = f"training_data_{epochs_count}epochs_{timestamp}.csv"
    with open(csv_filename, 'w', encoding='utf-8') as f:
        # Заголовки
        headers = list(history.keys())
        f.write("epoch," + ",".join(headers) + "\n")
        
        # Данные по эпохам
        for i in range(epochs_count):
            row = [str(i+1)]  # Номер эпохи
            for key in headers:
                row.append(str(history[key][i]))
            f.write(",".join(row) + "\n")
    print(f"📊 CSV данные сохранены: {csv_filename}")
    
    # 3. Текстовый отчет - для чтения человеком
    txt_filename = f"training_report_{epochs_count}epochs_{timestamp}.txt"
    with open(txt_filename, 'w', encoding='utf-8') as f:
        f.write(f"ОТЧЕТ ПО ОБУЧЕНИЮ EEG МОДЕЛИ\n")
        f.write(f"="*50 + "\n")
        f.write(f"Дата: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Количество эпох: {epochs_count}\n")
        f.write(f"Исходный файл: {latest_file}\n\n")
        
        # Статистика по метрикам
        f.write("ФИНАЛЬНЫЕ ЗНАЧЕНИЯ МЕТРИК:\n")
        f.write("-" * 30 + "\n")
        for key, values in history.items():
            final_val = values[-1]
            best_val = max(values) if 'loss' not in key else min(values)
            best_epoch = (values.index(best_val) + 1) if 'loss' not in key else (values.index(best_val) + 1)
            
            f.write(f"{key:20s}: {final_val:.6f} (финал)\n")
            f.write(f"{'':<20s}: {best_val:.6f} (лучший, эпоха {best_epoch})\n\n")
        
        # Динамика обучения
        f.write("ДИНАМИКА ОБУЧЕНИЯ (каждые 10% эпох):\n")
        f.write("-" * 40 + "\n")
        step = max(1, epochs_count // 10)
        f.write(f"{'Эпоха':>6s} {'Loss':>8s} {'Val_Loss':>10s} {'Accuracy':>10s} {'Val_Acc':>10s}")
        if 'f1_score' in history:
            f.write(f" {'F1_Score':>10s}")
        f.write("\n")
        
        for i in range(0, epochs_count, step):
            f.write(f"{i+1:>6d} {history['loss'][i]:>8.4f} {history['val_loss'][i]:>10.4f}")
            f.write(f" {history['accuracy'][i]:>10.4f} {history['val_accuracy'][i]:>10.4f}")
            if 'f1_score' in history:
                f.write(f" {history['f1_score'][i]:>10.4f}")
            f.write("\n")
        
        # Последняя эпоха
        i = epochs_count - 1
        f.write(f"{i+1:>6d} {history['loss'][i]:>8.4f} {history['val_loss'][i]:>10.4f}")
        f.write(f" {history['accuracy'][i]:>10.4f} {history['val_accuracy'][i]:>10.4f}")
        if 'f1_score' in history:
            f.write(f" {history['f1_score'][i]:>10.4f}")
        f.write("\n")
        
    print(f"📄 Текстовый отчет сохранен: {txt_filename}")
    
    # 4. Python скрипт для построения графиков
    plot_script = f"plot_training_{epochs_count}epochs_{timestamp}.py"
    with open(plot_script, 'w', encoding='utf-8') as f:
        f.write('''#!/usr/bin/env python3
"""
Автоматически сгенерированный скрипт для построения графиков обучения
"""

import matplotlib.pyplot as plt
import json

# Загрузить данные
with open('{}', 'r', encoding='utf-8') as f:
    data = json.load(f)

epochs = list(range(1, len(data['loss']) + 1))

# Создать графики
plt.figure(figsize=(15, 10))

# График функции потерь
plt.subplot(2, 3, 1)
plt.plot(epochs, data['loss'], label='Training Loss', color='red')
plt.plot(epochs, data['val_loss'], label='Validation Loss', color='blue')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, alpha=0.3)

# График точности
plt.subplot(2, 3, 2)
plt.plot(epochs, data['accuracy'], label='Training Accuracy', color='green')
plt.plot(epochs, data['val_accuracy'], label='Validation Accuracy', color='orange')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True, alpha=0.3)

# График Precision
plt.subplot(2, 3, 3)
plt.plot(epochs, data['precision'], label='Training Precision', color='purple')
plt.plot(epochs, data['val_precision'], label='Validation Precision', color='brown')
plt.title('Model Precision')
plt.xlabel('Epoch')
plt.ylabel('Precision')
plt.legend()
plt.grid(True, alpha=0.3)

# График Recall
plt.subplot(2, 3, 4)
plt.plot(epochs, data['recall'], label='Training Recall', color='pink')
plt.plot(epochs, data['val_recall'], label='Validation Recall', color='gray')
plt.title('Model Recall')
plt.xlabel('Epoch')
plt.ylabel('Recall')
plt.legend()
plt.grid(True, alpha=0.3)

# График F1 Score
plt.subplot(2, 3, 5)
if 'f1_score' in data:
    plt.plot(epochs, data['f1_score'], label='Training F1', color='cyan')
    plt.plot(epochs, data['val_f1_score'], label='Validation F1', color='magenta')
    plt.title('F1 Score')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.grid(True, alpha=0.3)

# Комбинированный график валидационных метрик
plt.subplot(2, 3, 6)
plt.plot(epochs, data['val_accuracy'], label='Val Accuracy')
plt.plot(epochs, data['val_precision'], label='Val Precision') 
plt.plot(epochs, data['val_recall'], label='Val Recall')
if 'val_f1_score' in data:
    plt.plot(epochs, data['val_f1_score'], label='Val F1 Score')
plt.title('Validation Metrics')
plt.xlabel('Epoch')
plt.ylabel('Score')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_plots_{}_epochs.png', dpi=300, bbox_inches='tight')
plt.show()

print("Графики сохранены как: training_plots_{}_epochs.png")
'''.format(json_filename, epochs_count, epochs_count))
    
    print(f"🐍 Python скрипт для графиков: {plot_script}")
    
    print(f"\n✅ Данные сохранены в {len([json_filename, csv_filename, txt_filename, plot_script])} форматах")
    print("📈 Для построения графиков запустите:")
    print(f"   python {plot_script}")

if __name__ == "__main__":
    save_training_data_to_files()