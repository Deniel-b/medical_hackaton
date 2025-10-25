#!/usr/bin/env python3
"""
EEG Model Inference

Использует обученную модель для предсказания классов движений по EEG данным.
"""

import numpy as np
import pickle
import tensorflow as tf
from pathlib import Path
import mne
import warnings
warnings.filterwarnings('ignore')

class EEGModelInference:
    def __init__(self, model_path="best_models/final_eeg_model.h5", 
                 preprocessing_path="best_models/preprocessing_objects.pkl"):
        
        print("=== EEG Model Inference ===")
        
        # Загрузить модель
        self.model = tf.keras.models.load_model(model_path)
        print(f"Модель загружена из: {model_path}")
        
        # Загрузить объекты препроцессинга
        with open(preprocessing_path, 'rb') as f:
            preprocessing_objects = pickle.load(f)
        
        self.scaler = preprocessing_objects['scaler']
        self.label_encoder = preprocessing_objects['label_encoder']
        self.classes = preprocessing_objects['classes']
        self.sequence_length = preprocessing_objects['sequence_length']
        self.n_features = preprocessing_objects['n_features']
        
        print(f"Классы: {self.classes}")
        print(f"Длина последовательности: {self.sequence_length}")
        print(f"Количество признаков: {self.n_features}")
        
        # Целевые каналы (те же, что в препроцессинге)
        self.target_channels = [
            'C3', 'Cz', 'C4', 'FCz', 'Fz', 'F3', 'F4', 
            'FC3', 'FC4', 'F7', 'F8', 'T7', 'T4', 'TP8', 
            'T5', 'PO7', 'PO3', 'PO4', 'PO8', 'P3', 'P4', 'Pz'
        ]
    
    def extract_features_from_epoch(self, epoch_data):
        """Извлечь признаки из одной эпохи (та же логика, что в препроцессинге)"""
        features = []
        
        for channel_data in epoch_data:
            if len(channel_data) == 0:
                features.extend([0.0, 0.0])
                continue
            
            # Признак 1: Среднее значение
            mean_val = np.mean(channel_data)
            
            # Признак 2: Медиана или стандартное отклонение
            if len(features) % 4 == 0:
                stat_val = np.median(channel_data)
            else:
                stat_val = np.std(channel_data)
            
            features.extend([mean_val, stat_val])
        
        # Дополнить до нужного количества признаков
        while len(features) < self.n_features:
            features.append(0.0)
        
        features = features[:self.n_features]
        return np.array(features)
    
    def process_edf_file(self, edf_path):
        """Обработать EDF файл и извлечь последовательности"""
        print(f"Обработка файла: {edf_path}")
        
        # Загрузить EDF файл
        raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
        data = raw.get_data()
        channel_names = raw.ch_names
        sfreq = raw.info['sfreq']
        annotations = raw.annotations
        
        print(f"Каналы: {len(channel_names)}, Частота: {sfreq} Гц, Аннотации: {len(annotations)}")
        
        # Извлечь данные нужных каналов
        channels_matrix = []
        for target_ch in self.target_channels:
            found = False
            for i, ch_name in enumerate(channel_names):
                if target_ch.upper() == ch_name.upper():
                    channels_matrix.append(data[i])
                    found = True
                    break
            if not found:
                channels_matrix.append(np.zeros(data.shape[1]))
        
        channels_matrix = np.array(channels_matrix)
        
        # Извлечь эпохи из аннотаций
        epochs_features = []
        epochs_info = []
        
        for i, (onset, duration, description) in enumerate(
            zip(annotations.onset, annotations.duration, annotations.description)
        ):
            start_sample = max(0, int(onset * sfreq))
            if duration <= 0:
                duration = 1.0
            length_samples = int(duration * sfreq)
            end_sample = start_sample + length_samples
            
            if end_sample > channels_matrix.shape[1]:
                continue
            
            epoch_data = channels_matrix[:, start_sample:end_sample]
            
            if epoch_data.shape[1] > 0:
                features = self.extract_features_from_epoch(epoch_data)
                epochs_features.append(features)
                epochs_info.append({
                    'onset': onset,
                    'duration': duration,
                    'description': description,
                    'epoch_index': i
                })
        
        print(f"Извлечено эпох: {len(epochs_features)}")
        return np.array(epochs_features), epochs_info
    
    def create_sequences_for_prediction(self, epochs_features):
        """Создать последовательности для предсказания"""
        sequences = []
        sequence_indices = []
        
        # Создать последовательности скользящим окном
        for i in range(len(epochs_features) - self.sequence_length + 1):
            sequence = []
            for j in range(self.sequence_length):
                sequence.append(epochs_features[i + j])
            
            sequences.append(np.array(sequence))
            sequence_indices.append(i + self.sequence_length - 1)  # Индекс последней эпохи
        
        return np.array(sequences), sequence_indices
    
    def predict_sequences(self, sequences):
        """Предсказать классы для последовательностей"""
        if len(sequences) == 0:
            return [], []
        
        # Нормализовать данные
        sequences_reshaped = sequences.reshape(-1, self.n_features)
        sequences_scaled = self.scaler.transform(sequences_reshaped)
        sequences_final = sequences_scaled.reshape(sequences.shape)
        
        # Получить предсказания
        predictions_proba = self.model.predict(sequences_final, verbose=0)
        predictions = np.argmax(predictions_proba, axis=1)
        
        # Преобразовать в названия классов
        predicted_classes = self.label_encoder.inverse_transform(predictions)
        confidence = np.max(predictions_proba, axis=1)
        
        return predicted_classes, confidence
    
    def predict_from_edf(self, edf_path):
        """Предсказать классы движений из EDF файла"""
        # Обработать файл
        epochs_features, epochs_info = self.process_edf_file(edf_path)
        
        if len(epochs_features) < self.sequence_length:
            print(f"Недостаточно эпох для создания последовательностей (нужно минимум {self.sequence_length})")
            return []
        
        # Создать последовательности
        sequences, sequence_indices = self.create_sequences_for_prediction(epochs_features)
        
        # Получить предсказания
        predicted_classes, confidence = self.predict_sequences(sequences)
        
        # Собрать результаты
        results = []
        for i, (pred_class, conf, seq_idx) in enumerate(zip(predicted_classes, confidence, sequence_indices)):
            epoch_info = epochs_info[seq_idx]
            results.append({
                'sequence_index': i,
                'epoch_index': seq_idx,
                'predicted_class': pred_class,
                'confidence': conf,
                'original_annotation': epoch_info['description'],
                'onset': epoch_info['onset'],
                'duration': epoch_info['duration']
            })
        
        return results
    
    def print_predictions(self, results):
        """Вывести результаты предсказаний"""
        print(f"\n=== Результаты предсказаний ===")
        print(f"Всего последовательностей: {len(results)}")
        
        # Статистика по классам
        class_counts = {}
        for result in results:
            pred_class = result['predicted_class']
            class_counts[pred_class] = class_counts.get(pred_class, 0) + 1
        
        print("\nРаспределение предсказанных классов:")
        for class_name, count in sorted(class_counts.items()):
            print(f"  {class_name}: {count}")
        
        print(f"\nДетальные результаты:")
        for i, result in enumerate(results[:10]):  # Показать первые 10
            print(f"  {i+1:2d}. Время: {result['onset']:6.2f}с | "
                  f"Предсказание: {result['predicted_class']:8s} | "
                  f"Уверенность: {result['confidence']:.3f} | "
                  f"Исходная аннотация: {result['original_annotation']}")
        
        if len(results) > 10:
            print(f"  ... и еще {len(results) - 10} результатов")

def main():
    """Пример использования"""
    # Создать инференс объект
    inference = EEGModelInference()
    
    # Указать путь к EDF файлу для тестирования
    test_edf_path = "data/Russian/sub-1/sub-1.edf"
    
    if not Path(test_edf_path).exists():
        print(f"Файл {test_edf_path} не найден!")
        print("Укажите корректный путь к EDF файлу")
        return
    
    # Получить предсказания
    results = inference.predict_from_edf(test_edf_path)
    
    # Вывести результаты
    inference.print_predictions(results)

if __name__ == "__main__":
    main()