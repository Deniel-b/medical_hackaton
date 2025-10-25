#!/usr/bin/env python3
"""
EEG Data Preprocessing Pipeline for Neural Network Training

Обрабатывает EDF файлы из папки data/Russian/sub-N и создает
датасет с признаками для классификации движений.
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import mne
from sklearn.preprocessing import StandardScaler
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

class EEGPreprocessor:
    def __init__(self, data_path="data/Russian", sequence_length=15, n_features=20):
        self.data_path = Path(data_path)
        self.sequence_length = sequence_length  # 15 эпох
        self.n_features = n_features  # 20 признаков
        
        # Нужные каналы (22 канала как было)
        self.target_channels = [
            'C3', 'Cz', 'C4',           # Центральные каналы
            'FCz', 'Fz', 'F3', 'F4',   # Фронтальные каналы  
            'FC3', 'FC4', 'F7', 'F8',  # Фронто-центральные
            'T7', 'T4', 'TP8', 'T5',   # Височные каналы
            'PO7', 'PO3', 'PO4', 'PO8', # Парието-окципитальные
            'P3', 'P4', 'Pz'           # Париетальные каналы
        ]
        
        # Маппинг классов
        self.class_mapping = {
            'Up': ['UP1', 'UP2', 'UP', 'up'],
            'Down': ['DOWN1', 'DOWN2', 'DOWN', 'down'],
            'Left': ['LEFT1', 'LEFT2', 'LEFT', 'left'],
            'Right': ['RIGHT1', 'RIGHT2', 'RIGHT', 'right'],
            'Back': ['BACK1', 'BACK2', 'BACK', 'back'],
            'Forward': ['FORWARD1', 'FORWARD2', 'FORWARD', 'forward'],
            'None': ['REST', 'NONE', 'none', 'rest', 'baseline']
        }
        
        # Создать обратный маппинг
        self.annotation_to_class = {}
        for target_class, annotations in self.class_mapping.items():
            for ann in annotations:
                self.annotation_to_class[ann.upper()] = target_class
        
        print(f"Инициализирован препроцессор:")
        print(f"  Путь к данным: {self.data_path}")
        print(f"  Длина последовательности: {self.sequence_length} эпох")
        print(f"  Количество признаков: {self.n_features}")
        print(f"  Целевые каналы: {len(self.target_channels)}")
        print(f"  Классы: {list(self.class_mapping.keys())}")
    
    def find_edf_files(self):
        """Найти все EDF файлы в папках sub-N"""
        edf_files = []
        for subject_dir in self.data_path.glob("sub-*"):
            if subject_dir.is_dir():
                for edf_file in subject_dir.glob("*.edf"):
                    edf_files.append(edf_file)
                    print(f"Найден файл: {edf_file}")
        
        print(f"Всего найдено EDF файлов: {len(edf_files)}")
        return edf_files
    
    def load_edf_file(self, edf_path):
        """Загрузить EDF файл и извлечь данные"""
        try:
            print(f"Загрузка файла: {edf_path}")
            raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
            
            # Получить информацию о файле
            data = raw.get_data()
            channel_names = raw.ch_names
            sfreq = raw.info['sfreq']
            annotations = raw.annotations
            
            print(f"  Каналы: {len(channel_names)}, Частота: {sfreq} Гц")
            print(f"  Аннотации: {len(annotations)}")
            
            return {
                'data': data,
                'channel_names': channel_names,
                'sfreq': sfreq,
                'annotations': annotations,
                'file_path': edf_path
            }
        except Exception as e:
            print(f"Ошибка загрузки {edf_path}: {e}")
            return None
    
    def extract_channel_data(self, eeg_data, channel_names):
        """Извлечь данные нужных каналов"""
        available_channels = {}
        channel_indices = {}
        
        # Найти индексы доступных каналов
        for i, ch_name in enumerate(channel_names):
            for target_ch in self.target_channels:
                if target_ch.upper() == ch_name.upper():
                    available_channels[target_ch] = eeg_data[i]
                    channel_indices[target_ch] = i
                    break
        
        print(f"  Найдено каналов из целевого списка: {len(available_channels)}/{len(self.target_channels)}")
        print(f"  Доступные каналы: {list(available_channels.keys())}")
        
        return available_channels, channel_indices
    
    def map_annotation_to_class(self, annotation):
        """Преобразовать аннотацию в целевой класс"""
        ann_upper = annotation.upper()
        
        # Проверить точное совпадение
        if ann_upper in self.annotation_to_class:
            return self.annotation_to_class[ann_upper]
        
        # Проверить частичное совпадение
        for target_class, annotations in self.class_mapping.items():
            for ann in annotations:
                if ann.upper() in ann_upper:
                    return target_class
        
        # Если не найдено, возвращаем None
        return 'None'
    
    def extract_epochs_from_annotations(self, eeg_data, annotations, sfreq):
        """Извлечь эпохи на основе аннотаций"""
        epochs_data = []
        epochs_labels = []
        
        for i, (onset, duration, description) in enumerate(
            zip(annotations.onset, annotations.duration, annotations.description)
        ):
            # Преобразовать аннотацию в класс
            target_class = self.map_annotation_to_class(description)
            
            # Вычислить временные границы
            start_sample = max(0, int(onset * sfreq))
            if duration <= 0:
                duration = 1.0
            length_samples = int(duration * sfreq)
            end_sample = start_sample + length_samples
            
            # Проверить границы
            if end_sample > eeg_data.shape[1]:
                continue
            
            # Извлечь данные эпохи
            epoch_data = eeg_data[:, start_sample:end_sample]
            
            if epoch_data.shape[1] > 0:
                epochs_data.append(epoch_data)
                epochs_labels.append(target_class)
                print(f"    Эпоха {i}: {description} -> {target_class}, длина: {epoch_data.shape[1]} сэмплов")
        
        return epochs_data, epochs_labels
    
    def extract_features_from_epoch(self, epoch_data):
        """Извлечь признаки из одной эпохи"""
        features = []
        
        # Для каждого канала извлекаем признаки
        for channel_data in epoch_data:
            if len(channel_data) == 0:
                # Если канал пустой, добавляем нули
                features.extend([0.0, 0.0])
                continue
            
            # Признак 1: Среднее значение
            mean_val = np.mean(channel_data)
            
            # Признак 2: Медиана (для некоторых каналов) или стандартное отклонение
            if len(features) % 4 == 0:  # Каждый 4-й канал - медиана
                stat_val = np.median(channel_data)
            else:  # Остальные - стандартное отклонение
                stat_val = np.std(channel_data)
            
            features.extend([mean_val, stat_val])
        
        # Если признаков меньше 20, дополняем нулями
        while len(features) < self.n_features:
            features.append(0.0)
        
        # Если признаков больше 20, обрезаем
        features = features[:self.n_features]
        
        return np.array(features)
    
    def create_sequences(self, all_epochs_features, all_epochs_labels):
        """Создать последовательности из признаков эпох"""
        sequences = []
        sequence_labels = []
        
        print(f"Создание последовательностей из {len(all_epochs_features)} эпох...")
        
        # Создаем последовательности скользящим окном
        for i in range(len(all_epochs_features) - self.sequence_length + 1):
            # Берем последовательность из sequence_length эпох
            sequence_features = []
            for j in range(self.sequence_length):
                sequence_features.append(all_epochs_features[i + j])
            
            # Метка последовательности = метка последней эпохи
            sequence_label = all_epochs_labels[i + self.sequence_length - 1]
            
            sequences.append(np.array(sequence_features))
            sequence_labels.append(sequence_label)
        
        print(f"Создано последовательностей: {len(sequences)}")
        return np.array(sequences), np.array(sequence_labels)
    
    def process_all_files(self):
        """Обработать все EDF файлы и создать датасет"""
        edf_files = self.find_edf_files()
        
        all_epochs_features = []
        all_epochs_labels = []
        file_info = []
        
        for edf_file in edf_files:
            print(f"\n=== Обработка файла: {edf_file.name} ===")
            
            # Загрузить файл
            eeg_data_dict = self.load_edf_file(edf_file)
            if eeg_data_dict is None:
                continue
            
            # Извлечь данные нужных каналов
            channel_data, channel_indices = self.extract_channel_data(
                eeg_data_dict['data'], eeg_data_dict['channel_names']
            )
            
            if len(channel_data) == 0:
                print("  Пропуск файла: не найдено целевых каналов")
                continue
            
            # Создать матрицу данных каналов (заполнить отсутствующие каналы нулями)
            channels_matrix = []
            for target_ch in self.target_channels:
                if target_ch in channel_data:
                    channels_matrix.append(channel_data[target_ch])
                else:
                    # Заполнить отсутствующий канал нулями
                    channels_matrix.append(np.zeros(eeg_data_dict['data'].shape[1]))
            
            channels_matrix = np.array(channels_matrix)
            
            # Извлечь эпохи
            epochs_data, epochs_labels = self.extract_epochs_from_annotations(
                channels_matrix, eeg_data_dict['annotations'], eeg_data_dict['sfreq']
            )
            
            # Извлечь признаки из каждой эпохи
            for epoch_data, epoch_label in zip(epochs_data, epochs_labels):
                features = self.extract_features_from_epoch(epoch_data)
                all_epochs_features.append(features)
                all_epochs_labels.append(epoch_label)
                file_info.append(edf_file.name)
        
        print("\n=== Общая статистика ===")
        print(f"Всего эпох обработано: {len(all_epochs_features)}")
        
        # Статистика по классам
        class_counts = defaultdict(int)
        for label in all_epochs_labels:
            class_counts[label] += 1
        
        print("Распределение классов:")
        for class_name, count in class_counts.items():
            print(f"  {class_name}: {count} эпох")
        
        # Создать последовательности
        sequences, sequence_labels = self.create_sequences(all_epochs_features, all_epochs_labels)
        
        return {
            'sequences': sequences,
            'labels': sequence_labels,
            'class_counts': dict(class_counts),
            'file_info': file_info,
            'feature_names': [f'feature_{i}' for i in range(self.n_features)]
        }
    
    def save_processed_data(self, processed_data, output_path="processed_eeg_data.pkl"):
        """Сохранить обработанные данные"""
        print(f"\nСохранение обработанных данных в {output_path}")
        
        with open(output_path, 'wb') as f:
            pickle.dump(processed_data, f)
        
        print("Данные сохранены:")
        print(f"  Форма последовательностей: {processed_data['sequences'].shape}")
        print(f"  Количество меток: {len(processed_data['labels'])}")
        print(f"  Размер файла: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")

def main():
    """Основная функция препроцессинга"""
    print("=== EEG Data Preprocessing Pipeline ===")
    
    # Создать препроцессор
    preprocessor = EEGPreprocessor(
        data_path="data/Russian",
        sequence_length=15,
        n_features=20
    )
    
    # Обработать все файлы
    processed_data = preprocessor.process_all_files()
    
    # Сохранить результаты
    preprocessor.save_processed_data(processed_data)
    
    print("\n=== Препроцессинг завершен ===")
    print("Итоговый датасет:")
    print(f"  Последовательности: {processed_data['sequences'].shape}")
    print(f"  Метки: {processed_data['labels'].shape}")
    print(f"  Классы: {list(processed_data['class_counts'].keys())}")

if __name__ == "__main__":
    main()