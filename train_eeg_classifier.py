#!/usr/bin/env python3
"""
EEG Neural Network Classifier Training

Обучает нейронную сеть TensorFlow для классификации движений по EEG данным.
Использует предобработанные данные из eeg_preprocessing.py
"""

import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os

class EEGNeuralNetworkTrainer:
    def __init__(self, processed_data_path="processed_eeg_data.pkl"):
        self.processed_data_path = processed_data_path
        self.sequence_length = 15
        self.n_features = 20
        self.classes = ['Up', 'Down', 'Left', 'Right', 'Back', 'Forward', 'None']
        self.n_classes = len(self.classes)
        
        # Создать папки для моделей
        self.models_dir = Path("best_models")
        self.models_dir.mkdir(exist_ok=True)
        
        print("=== EEG Neural Network Classifier ===")
        print(f"Длина последовательности: {self.sequence_length}")
        print(f"Количество признаков: {self.n_features}")
        print(f"Классы: {self.classes}")
        print(f"Количество классов: {self.n_classes}")
    
    def load_processed_data(self):
        """Загрузить предобработанные данные"""
        print(f"Загрузка данных из {self.processed_data_path}")
        
        if not os.path.exists(self.processed_data_path):
            raise FileNotFoundError(
                f"Файл {self.processed_data_path} не найден. "
                "Сначала запустите eeg_preprocessing.py"
            )
        
        with open(self.processed_data_path, 'rb') as f:
            data = pickle.load(f)
        
        print("Данные загружены:")
        print(f"  Форма последовательностей: {data['sequences'].shape}")
        print(f"  Количество меток: {len(data['labels'])}")
        print(f"  Распределение классов: {data['class_counts']}")
        
        return data
    
    def prepare_data_for_training(self, data):
        """Подготовить данные для обучения с аугментацией"""
        sequences = data['sequences']  # (n_samples, sequence_length, n_features)
        labels = data['labels']
        
        print("\nПодготовка данных для обучения...")
        print(f"Исходная форма данных: {sequences.shape}")
        
        # Аугментация данных для улучшения обобщения
        print("Применение аугментации данных...")
        augmented_sequences = []
        augmented_labels = []
        
        for seq, label in zip(sequences, labels):
            # Оригинальные данные
            augmented_sequences.append(seq)
            augmented_labels.append(label)
            
            # Только для малочисленных классов (не None)
            if label != 'None':
                # Добавляем шум
                noise_factor = 0.05
                noisy_seq = seq + np.random.normal(0, noise_factor, seq.shape)
                augmented_sequences.append(noisy_seq)
                augmented_labels.append(label)
                
                # Временной сдвиг (циклический)
                if len(seq) > 1:
                    shifted_seq = np.roll(seq, shift=1, axis=0)
                    augmented_sequences.append(shifted_seq)
                    augmented_labels.append(label)
        
        sequences = np.array(augmented_sequences)
        labels = np.array(augmented_labels)
        
        print(f"После аугментации: {sequences.shape}")
        
        # Кодировать метки
        label_encoder = LabelEncoder()
        encoded_labels = label_encoder.fit_transform(labels)
        
        # Преобразовать в категориальные метки
        categorical_labels = to_categorical(encoded_labels, num_classes=self.n_classes)
        
        print(f"Кодировщик классов: {dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))}")
        
        # Разделить на train/validation/test
        X_temp, X_test, y_temp, y_test = train_test_split(
            sequences, categorical_labels, 
            test_size=0.2, 
            random_state=42, 
            stratify=encoded_labels
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, 
            test_size=0.25,  # 0.25 * 0.8 = 0.2 от общего датасета
            random_state=42, 
            stratify=np.argmax(y_temp, axis=1)
        )
        
        print("Разделение данных:")
        print(f"  Обучение: {X_train.shape[0]} образцов")
        print(f"  Валидация: {X_val.shape[0]} образцов") 
        print(f"  Тест: {X_test.shape[0]} образцов")
        
        # Нормализация признаков
        # Reshape для нормализации: (n_samples * sequence_length, n_features)
        X_train_reshaped = X_train.reshape(-1, self.n_features)
        X_val_reshaped = X_val.reshape(-1, self.n_features)
        X_test_reshaped = X_test.reshape(-1, self.n_features)
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_reshaped)
        X_val_scaled = scaler.transform(X_val_reshaped)
        X_test_scaled = scaler.transform(X_test_reshaped)
        
        # Reshape обратно: (n_samples, sequence_length, n_features)
        X_train_final = X_train_scaled.reshape(X_train.shape)
        X_val_final = X_val_scaled.reshape(X_val.shape)
        X_test_final = X_test_scaled.reshape(X_test.shape)
        
        print("Нормализация признаков завершена")
        
        return {
            'X_train': X_train_final,
            'X_val': X_val_final,
            'X_test': X_test_final,
            'y_train': y_train,
            'y_val': y_val,
            'y_test': y_test,
            'label_encoder': label_encoder,
            'scaler': scaler
        }
    
    def create_model(self):
        """Создать улучшенную нейронную сеть с продвинутыми функциями активации"""
        print("Создание улучшенной модели нейронной сети...")
        
        # Входная форма: (sequence_length, n_features)
        input_shape = (self.sequence_length, self.n_features)
        
        model = Sequential([
            # Flatten входных данных для Dense слоев
            tf.keras.layers.Flatten(input_shape=input_shape),
            
            # Значительно увеличенная архитектура
            Dense(32, activation='relu'),
            Dropout(0.3),
            
            Dense(64, activation=tf.keras.layers.PReLU()),
            tf.keras.layers.BatchNormalization(),
            Dropout(0.2),
            
            Dense(128, activation=tf.keras.layers.PReLU()),
            tf.keras.layers.BatchNormalization(),
            Dropout(0.3),
            
            Dense(256, activation=tf.keras.layers.PReLU()),
            tf.keras.layers.BatchNormalization(),
            Dropout(0.3),
            
            Dense(512, activation=tf.keras.layers.PReLU()),  # Больше нейронов
            tf.keras.layers.BatchNormalization(),
            Dropout(0.4),
            
            Dense(256, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            Dropout(0.3),
            
            Dense(128, activation=tf.keras.layers.PReLU()),
            tf.keras.layers.BatchNormalization(),
            Dropout(0.2),
            
            Dense(64, activation=tf.keras.layers.PReLU()),
            Dropout(0.1),
            
            Dense(32, activation=tf.keras.layers.LeakyReLU()),
            
            Dense(16, activation=tf.keras.layers.PReLU()),
            Dropout(0.1),
            
            # Последний слой для классификации
            Dense(self.n_classes, activation='softmax')
        ])
        
        # Улучшенная компиляция модели
        model.compile(
            optimizer='nadam',  # Улучшенный оптимизатор
            loss='categorical_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        print("Архитектура улучшенной модели:")
        model.summary()
        
        return model
    
    def train_model(self, model, train_data):
        """Обучить модель с улучшенными параметрами и балансировкой классов"""
        print("\nНачало обучения улучшенной модели...")
        
        # Вычисляем веса классов для балансировки
        y_train_labels = np.argmax(train_data['y_train'], axis=1)
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(y_train_labels),
            y=y_train_labels
        )
        class_weight_dict = dict(enumerate(class_weights))
        
        # Уменьшаем вес класса "None" (индекс 4) для лучшей балансировки
        if 4 in class_weight_dict:  # "None" класс
            class_weight_dict[4] *= 0.7  # Снижаем важность "None"
            
        print(f"Веса классов для балансировки: {class_weight_dict}")
        
        # Снижение learning rate при застое
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_f1_score',
            factor=0.3,  # Более агрессивное снижение
            patience=20,
            min_lr=1e-6,
            mode='max',
            verbose=1
        )
        
        # Добавляем метрику F1 score
        model.compile(
            optimizer=tf.keras.optimizers.Nadam(learning_rate=0.002),  # Немного больше learning rate
            loss='categorical_crossentropy',
            metrics=['accuracy', 'precision', 'recall', tf.keras.metrics.F1Score(average='macro')]
        )
        
        # Обучение с максимальным количеством эпох и балансировкой классов
        history = model.fit(
            train_data['X_train'], train_data['y_train'],
            validation_data=(train_data['X_val'], train_data['y_val']),
            batch_size=16,  # Меньший batch size для лучшего обучения
            epochs=1000,    # 1000 эпох как запрошено
            verbose=1,
            class_weight=class_weight_dict,  # Балансировка классов
            callbacks=[reduce_lr]  # ТОЛЬКО reduce_lr, БЕЗ early stopping
        )
        
        print("Обучение завершено!")
        
        # Сохраняем модель ТОЛЬКО В КОНЦЕ
        print("Сохранение финальной модели...")
        model.save(str(self.models_dir / 'final_eeg_model.h5'))
        print("Модель сохранена как final_eeg_model.h5")
        
        # Сохраняем историю обучения
        epochs_completed = len(history.history['loss'])
        print(f"Фактически выполнено эпох: {epochs_completed}")
        
        # Сохранение истории в файл
        history_file = f'training_history_{epochs_completed}_epochs.pkl'
        with open(history_file, 'wb') as f:
            pickle.dump(history.history, f)
        print(f"История обучения сохранена в: {history_file}")
        
        return history
    
    def evaluate_model(self, model, train_data):
        """Оценить модель и показать примеры"""
        print("\n=== Оценка модели ===")
        
        # Предсказания на тестовой выборке
        y_pred_proba = model.predict(train_data['X_test'])
        y_pred = np.argmax(y_pred_proba, axis=1)
        y_true = np.argmax(train_data['y_test'], axis=1)
        
        # Преобразовать обратно в названия классов
        label_encoder = train_data['label_encoder']
        y_true_labels = label_encoder.inverse_transform(y_true)
        y_pred_labels = label_encoder.inverse_transform(y_pred)
        
        # F1 Score (макро и микро)
        f1_macro = f1_score(y_true, y_pred, average='macro')
        f1_micro = f1_score(y_true, y_pred, average='micro')
        f1_weighted = f1_score(y_true, y_pred, average='weighted')
        
        print(f"F1 Score (macro): {f1_macro:.4f}")
        print(f"F1 Score (micro): {f1_micro:.4f}")
        print(f"F1 Score (weighted): {f1_weighted:.4f}")
        
        # Детальный отчет по классам
        print("\nОтчет по классам:")
        print(classification_report(y_true_labels, y_pred_labels))
        
        # Показать примеры предсказаний
        print("\n=== ПРИМЕРЫ ТЕСТОВЫХ ДАННЫХ ===")
        print("Формат: [Реальный класс] -> [Предсказанный класс] (Уверенность)")
        print("-" * 60)
        
        # Найти примеры разных типов предсказаний
        examples_shown = 0
        max_examples = 10
        
        for i in range(len(y_true_labels)):
            if examples_shown >= max_examples:
                break
                
            true_label = y_true_labels[i]
            pred_label = y_pred_labels[i]
            confidence = np.max(y_pred_proba[i])
            
            # Показать данные образца
            sample_data = train_data['X_test'][i]
            
            print(f"\nПример {examples_shown + 1}:")
            print(f"  Реальный класс:     {true_label}")
            print(f"  Предсказанный:      {pred_label}")
            print(f"  Уверенность:        {confidence:.3f}")
            print(f"  Правильно:          {'✅' if true_label == pred_label else '❌'}")
            print(f"  Форма данных:       {sample_data.shape}")
            print(f"  Признаки (первые 5): {sample_data.flatten()[:5]}")
            print(f"  Среднее значение:    {np.mean(sample_data):.3f}")
            print(f"  Стд. отклонение:     {np.std(sample_data):.3f}")
            
            examples_shown += 1
        
        # Показать статистику по правильности предсказаний
        correct_predictions = np.sum(y_true == y_pred)
        total_predictions = len(y_true)
        accuracy = correct_predictions / total_predictions
        
        print("\n=== ОБЩАЯ СТАТИСТИКА ===")
        print(f"Всего тестовых образцов: {total_predictions}")
        print(f"Правильных предсказаний: {correct_predictions}")
        print(f"Точность: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        # Показать распределение ошибок по классам
        print("\n=== АНАЛИЗ ОШИБОК ПО КЛАССАМ ===")
        for class_name in self.classes:
            class_mask = y_true_labels == class_name
            if np.sum(class_mask) > 0:
                class_accuracy = np.sum((y_true_labels == y_pred_labels) & class_mask) / np.sum(class_mask)
                class_count = np.sum(class_mask)
                print(f"{class_name:>8}: {class_accuracy:.3f} ({class_count} образцов)")
        
        # Матрица ошибок
        cm = confusion_matrix(y_true_labels, y_pred_labels, labels=self.classes)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.classes, yticklabels=self.classes)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=300)
        plt.show()
        
        return {
            'f1_macro': f1_macro,
            'f1_micro': f1_micro,
            'f1_weighted': f1_weighted,
            'y_true': y_true_labels,
            'y_pred': y_pred_labels,
            'confusion_matrix': cm,
            'accuracy': accuracy
        }
    
    def plot_training_history(self, history):
        """Построить графики обучения с полной информацией"""
        epochs_trained = len(history.history['loss'])
        print(f"\nСоздание графиков обучения для {epochs_trained} эпох...")
        
        plt.figure(figsize=(20, 12))
        
        # График функции потерь
        plt.subplot(2, 3, 1)
        plt.plot(history.history['loss'], label='Training Loss', linewidth=2)
        plt.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
        plt.title(f'Model Loss ({epochs_trained} epochs)', fontsize=14)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # График точности
        plt.subplot(2, 3, 2)
        plt.plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
        plt.title(f'Model Accuracy ({epochs_trained} epochs)', fontsize=14)
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # График precision
        plt.subplot(2, 3, 3)
        plt.plot(history.history['precision'], label='Training Precision', linewidth=2)
        plt.plot(history.history['val_precision'], label='Validation Precision', linewidth=2)
        plt.title(f'Model Precision ({epochs_trained} epochs)', fontsize=14)
        plt.xlabel('Epoch')
        plt.ylabel('Precision')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # График recall
        plt.subplot(2, 3, 4)
        plt.plot(history.history['recall'], label='Training Recall', linewidth=2)
        plt.plot(history.history['val_recall'], label='Validation Recall', linewidth=2)
        plt.title(f'Model Recall ({epochs_trained} epochs)', fontsize=14)
        plt.xlabel('Epoch')
        plt.ylabel('Recall')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # График F1 Score
        plt.subplot(2, 3, 5)
        if 'f1_score' in history.history:
            plt.plot(history.history['f1_score'], label='Training F1', linewidth=2)
            plt.plot(history.history['val_f1_score'], label='Validation F1', linewidth=2)
            plt.title(f'F1 Score ({epochs_trained} epochs)', fontsize=14)
            plt.xlabel('Epoch')
            plt.ylabel('F1 Score')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        # Learning Rate (если есть)
        plt.subplot(2, 3, 6)
        if 'lr' in history.history:
            plt.plot(history.history['lr'], label='Learning Rate', linewidth=2)
            plt.title(f'Learning Rate ({epochs_trained} epochs)', fontsize=14)
            plt.xlabel('Epoch')
            plt.ylabel('Learning Rate')
            plt.yscale('log')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Сохраняем с указанием количества эпох в названии
        filename = f'training_history_{epochs_trained}_epochs.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"График сохранен как: {filename}")
        
        # Также сохраняем стандартное название для совместимости
        plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def save_training_artifacts(self, model, train_data, evaluation_results, history):
        """Сохранить артефакты обучения"""
        print("\nСохранение артефактов обучения...")
        
        # Сохранить модель
        model.save(str(self.models_dir / 'final_eeg_model.h5'))
        
        # Сохранить скейлер и кодировщик
        with open(str(self.models_dir / 'preprocessing_objects.pkl'), 'wb') as f:
            pickle.dump({
                'scaler': train_data['scaler'],
                'label_encoder': train_data['label_encoder'],
                'classes': self.classes,
                'sequence_length': self.sequence_length,
                'n_features': self.n_features
            }, f)
        
        # Сохранить результаты оценки
        with open(str(self.models_dir / 'evaluation_results.pkl'), 'wb') as f:
            pickle.dump(evaluation_results, f)
        
        # Сохранить историю обучения
        epochs_trained = len(history.history['loss'])
        with open(str(self.models_dir / 'training_history.pkl'), 'wb') as f:
            pickle.dump(history.history, f)
        
        # Сохранить подробную информацию об обучении
        info_file = self.models_dir / 'training_info.txt'
        with open(info_file, 'w', encoding='utf-8') as f:
            f.write(f"Информация об обучении модели EEG\n")
            f.write(f"="*50 + "\n")
            f.write(f"Количество эпох: {epochs_trained}\n")
            f.write(f"Итоговая точность: {evaluation_results.get('accuracy', 'N/A')}\n")
            f.write(f"F1 Score (macro): {evaluation_results.get('f1_macro', 'N/A')}\n")
            f.write(f"F1 Score (micro): {evaluation_results.get('f1_micro', 'N/A')}\n")
            f.write(f"F1 Score (weighted): {evaluation_results.get('f1_weighted', 'N/A')}\n")
            f.write(f"\nФинальные значения метрик:\n")
            f.write(f"Training Loss: {history.history['loss'][-1]:.4f}\n")
            f.write(f"Validation Loss: {history.history['val_loss'][-1]:.4f}\n")
            f.write(f"Training Accuracy: {history.history['accuracy'][-1]:.4f}\n")
            f.write(f"Validation Accuracy: {history.history['val_accuracy'][-1]:.4f}\n")
        
        print(f"Артефакты сохранены в папку: {self.models_dir}")
        print(f"График с {epochs_trained} эпохами сохранен как: training_history_{epochs_trained}_epochs.png")
    
    def run_training_pipeline(self):
        """Запустить полный пайплайн обучения"""
        print("=== Запуск пайплайна обучения ===")
        
        # 1. Загрузить данные
        data = self.load_processed_data()
        
        # 2. Подготовить данные
        train_data = self.prepare_data_for_training(data)
        
        # 3. Создать модель
        model = self.create_model()
        
        # 4. Обучить модель
        history = self.train_model(model, train_data)
        
        # 5. Оценить модель
        evaluation_results = self.evaluate_model(model, train_data)
        
        # 6. Построить графики
        self.plot_training_history(history)
        
        # 7. Сохранить артефакты
        self.save_training_artifacts(model, train_data, evaluation_results, history)
        
        print("\n=== Обучение завершено успешно! ===")
        print(f"Лучший F1 Score (macro): {evaluation_results['f1_macro']:.4f}")
        
        return model, history, evaluation_results

def main():
    """Основная функция"""
    # Создать тренер
    trainer = EEGNeuralNetworkTrainer()
    
    # Запустить обучение
    model, history, results = trainer.run_training_pipeline()
    
    print("Готово! Модель обучена и сохранена.")

if __name__ == "__main__":
    main()