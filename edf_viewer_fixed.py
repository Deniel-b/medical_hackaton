#!/usr/bin/env python3
"""
EDF Desktop Viewer - Настольное приложение для просмотра EDF файлов
с интерактивными matplotlib графиками
"""

import sys
import os
import argparse
from pathlib import Path

import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import mne

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTabWidget, QLabel, QSpinBox, QComboBox, QPushButton, QFileDialog,
    QTextEdit, QGroupBox, QGridLayout, QMessageBox, QProgressBar,
    QFrame, QCheckBox, QRadioButton, QButtonGroup, QScrollArea
)
from PyQt5.QtCore import QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QFont

def apply_kalman_filter(data, process_variance=1e-4, measurement_variance=1e-2):
    """Применить фильтр Калмана для шумоподавления"""
    n_samples = len(data)
    
    # Инициализация массивов
    x_hat = np.zeros(n_samples)  # Оценка состояния
    P = np.ones(n_samples)       # Ковариация ошибки
    
    # Шум процесса и измерения (адаптивный на основе дисперсии сигнала)
    Q = process_variance
    R = measurement_variance * (np.var(data) + 1e-6)  # Адаптивный шум измерения
    
    # Начальные значения
    x_hat[0] = data[0]
    P[0] = R
    
    for k in range(1, n_samples):
        # Шаг предсказания
        x_hat_minus = x_hat[k-1]
        P_minus = P[k-1] + Q
        
        # Шаг коррекции
        K = P_minus / (P_minus + R)  # Коэффициент Калмана
        x_hat[k] = x_hat_minus + K * (data[k] - x_hat_minus)
        P[k] = (1 - K) * P_minus
    
    return x_hat

def create_bipolar_montage(raw_data, channel_names):
    """Создать биполярный монтаж double-banana с полными парами электродов"""
    print(f"Доступные каналы: {channel_names}")
    
    # Полные пары биполярного монтажа double-banana из оригинального скрипта
    montage_pairs = [
        ("Fp1", "F3"), ("F3", "C3"), ("C3", "P3"), ("P3", "O1"),
        ("Fp2", "F4"), ("F4", "C4"), ("C4", "P4"), ("P4", "O2"),
        ("F7", "T3"), ("T3", "T5"), ("T5", "O1"),
        ("F8", "T4"), ("T4", "T6"), ("T6", "O2"),
        ("Fp1", "Fpz"), ("Fpz", "Fp2"),
        ("F7", "F3"), ("F3", "Cz"), ("Cz", "Pz"), ("Pz", "O1"),
        ("F8", "F4"), ("F4", "Cz"), ("Cz", "Pz"), ("Pz", "O2"),
        ("T3", "C3"), ("C3", "Cz"), ("Cz", "C4"), ("C4", "T4"),
        ("P3", "Pz"), ("Pz", "P4")
    ]
    
    # Создать индексное отображение каналов
    channel_index = {name: idx for idx, name in enumerate(channel_names)}
    print(f"Индексы каналов: {list(channel_index.keys())}")
    
    bipolar_data = []
    bipolar_names = []
    missing_pairs = []
    
    for ch1, ch2 in montage_pairs:
        if ch1 in channel_index and ch2 in channel_index:
            idx1 = channel_index[ch1]
            idx2 = channel_index[ch2]
            bipolar_signal = raw_data[idx1] - raw_data[idx2]
            bipolar_data.append(bipolar_signal)
            bipolar_names.append(f"{ch1}-{ch2}")
            print(f"Создана биполярная пара: {ch1}-{ch2}")
        else:
            missing_pairs.append(f"{ch1}-{ch2}")
    
    if missing_pairs:
        print(f"Предупреждение: Отсутствуют каналы для пар: {', '.join(missing_pairs)}")
    
    if not bipolar_data:
        # Резервный вариант - простые соседние пары, если стандартные пары не найдены
        print("Создание резервных биполярных пар из соседних каналов...")
        for i in range(min(len(channel_names)-1, 20)):
            bipolar_signal = raw_data[i] - raw_data[i+1]
            bipolar_data.append(bipolar_signal)
            bipolar_names.append(f"{channel_names[i]}-{channel_names[i+1]}")
    
    print(f"Создано {len(bipolar_data)} биполярных каналов: {bipolar_names}")
    return np.array(bipolar_data), bipolar_names

class EEGDataProcessor:
    def __init__(self, edf_data):
        print("=== ИНИЦИАЛИЗАЦИЯ ПРОЦЕССОРА ДАННЫХ ===")
        self.edf_data = edf_data
        self.data = edf_data['data']
        self.channel_names = edf_data['channel_names']
        self.sfreq = edf_data['sfreq']
        self.annotations = edf_data['annotations']
        
        print(f"Данные загружены: {self.data.shape}")
        print(f"Каналы: {len(self.channel_names)}")
        print(f"Частота дискретизации: {self.sfreq} Гц")
        print(f"Аннотации: {len(self.annotations)}")
        
        # Создать биполярный монтаж
        self.bipolar_data, self.bipolar_names = create_bipolar_montage(
            self.data, self.channel_names
        )
        
        # Временная ось
        self.time_axis = np.arange(self.data.shape[1]) / self.sfreq
        
        # Обработать аннотации
        self.annotation_groups = self._process_annotations()
        print("=== ПРОЦЕССОР ДАННЫХ ГОТОВ ===")
    
    def _process_annotations(self):
        """Обработать аннотации и сгруппировать по описанию - точно как в оригинальном скрипте"""
        print("Обработка аннотаций...")
        groups = {}
        
        # Обработать каждую аннотацию как отдельную эпоху (как в оригинальном скрипте)
        for i, (onset, duration, description) in enumerate(
            zip(self.annotations.onset, self.annotations.duration, self.annotations.description)
        ):
            # Вычислить начальный и конечный сэмпл для эпохи
            start_sample = max(0, int(round(onset * self.sfreq)))
            
            # Если длительность <= 0, установить минимальную длительность (как в оригинале)
            if duration <= 0:
                duration = 1.0
            
            length_samples = int(round(duration * self.sfreq))
            end_sample = start_sample + length_samples
            
            # Проверить границы данных
            max_samples = self.data.shape[1]
            if start_sample >= max_samples or end_sample > max_samples:
                print(f"Пропуск аннотации {i}: границы {start_sample}-{end_sample} за пределами данных (0-{max_samples})")
                continue
            
            if description not in groups:
                groups[description] = []
            
            # Сохранить информацию об эпохе
            groups[description].append({
                'onset': onset,
                'duration': duration,
                'start_sample': start_sample,
                'end_sample': end_sample,
                'length_samples': length_samples,
                'index': i
            })
            
            print(f"Эпоха {i} ({description}): сэмплы {start_sample}-{end_sample}, длительность {duration:.3f}с")
        
        print("Группы аннотаций:")
        for ann_type, epochs in groups.items():
            print(f"  {ann_type}: {len(epochs)} эпох")
        
        return groups

class LoadEDFThread(QThread):
    """Поток для загрузки EDF файлов в фоне"""
    finished = pyqtSignal(object)
    error = pyqtSignal(str)
    progress = pyqtSignal(str)
    
    def __init__(self, edf_path):
        super().__init__()
        self.edf_path = edf_path
    
    def run(self):
        try:
            self.progress.emit("Загрузка EDF файла...")
            
            # Загрузить EDF файл с помощью MNE
            raw = mne.io.read_raw_edf(self.edf_path, preload=True, verbose=False)
            
            self.progress.emit("Обработка данных...")
            
            # Получить данные и информацию о каналах
            data = raw.get_data()
            channel_names = raw.ch_names
            sfreq = raw.info['sfreq']
            annotations = raw.annotations
            
            edf_data = {
                'data': data,
                'channel_names': channel_names,
                'sfreq': sfreq,
                'annotations': annotations
            }
            
            self.progress.emit("Создание процессора ЭЭГ...")
            processor = EEGDataProcessor(edf_data)
            
            self.progress.emit("Готово!")
            self.finished.emit(processor)
            
        except Exception as e:
            self.error.emit(f"Ошибка загрузки EDF файла: {str(e)}")

class InteractivePlotWidget(QWidget):
    """Виджет для отображения интерактивных matplotlib графиков"""
    
    def __init__(self, title="График"):
        super().__init__()
        self.processor = None
        self.title = title
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout()
        
        # Элементы управления
        controls_frame = QFrame()
        controls_frame.setFrameStyle(QFrame.Box)
        controls_layout = QGridLayout(controls_frame)
        
        # Максимальное количество каналов
        controls_layout.addWidget(QLabel("Макс. каналов:"), 0, 0)
        self.max_channels_spin = QSpinBox()
        self.max_channels_spin.setRange(1, 50)
        self.max_channels_spin.setValue(15)
        self.max_channels_spin.valueChanged.connect(self.update_plot)
        controls_layout.addWidget(self.max_channels_spin, 0, 1)
        
        # Временное окно
        controls_layout.addWidget(QLabel("Врем. окно (с):"), 0, 2)
        self.time_window_spin = QSpinBox()
        self.time_window_spin.setRange(1, 300)
        self.time_window_spin.setValue(30)
        self.time_window_spin.valueChanged.connect(self.update_plot)
        controls_layout.addWidget(self.time_window_spin, 0, 3)
        
        # Переключатель фильтра Калмана
        self.kalman_checkbox = QCheckBox("Фильтр Калмана")
        self.kalman_checkbox.setChecked(True)
        self.kalman_checkbox.stateChanged.connect(self.update_plot)
        controls_layout.addWidget(self.kalman_checkbox, 0, 4)
        
        # Тип аннотации (будет включен при загрузке данных)
        controls_layout.addWidget(QLabel("Тип аннотации:"), 1, 0)
        self.annotation_combo = QComboBox()
        self.annotation_combo.setEnabled(False)
        self.annotation_combo.currentTextChanged.connect(self.update_plot)
        controls_layout.addWidget(self.annotation_combo, 1, 1, 1, 2)
        
        # Максимальное количество эпох
        controls_layout.addWidget(QLabel("Макс. эпох:"), 1, 3)
        self.max_epochs_spin = QSpinBox()
        self.max_epochs_spin.setRange(1, 50)
        self.max_epochs_spin.setValue(10)
        self.max_epochs_spin.valueChanged.connect(self.update_plot)
        controls_layout.addWidget(self.max_epochs_spin, 1, 4)
        
        layout.addWidget(controls_frame)
        
        # Matplotlib фигура и канвас
        self.figure = Figure(figsize=(16, 10))
        self.canvas = FigureCanvas(self.figure)
        
        # Панель инструментов matplotlib
        self.toolbar = NavigationToolbar(self.canvas, self)
        
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        
        self.setLayout(layout)
    
    def set_processor(self, processor):
        """Установить процессор данных ЭЭГ"""
        print(f"Установка процессора для {self.title}")
        self.processor = processor
        
        # Обновить комбобокс аннотаций
        if processor.annotation_groups:
            self.annotation_combo.clear()
            self.annotation_combo.addItems(list(processor.annotation_groups.keys()))
            self.annotation_combo.setEnabled(True)
        
        self.update_plot()
    
    def update_plot(self):
        """Обновить текущий график"""
        # Будет переопределено в подклассах
        pass

class RawDataPlotWidget(InteractivePlotWidget):
    """Виджет для графиков исходных данных"""
    
    def __init__(self):
        super().__init__("Исходные данные ЭЭГ")
    
    def update_plot(self):
        if not self.processor:
            return
        
        print("=== ОБНОВЛЕНИЕ ГРАФИКА ИСХОДНЫХ ДАННЫХ ===")
        max_channels = self.max_channels_spin.value()
        time_window = self.time_window_spin.value()
        use_kalman = self.kalman_checkbox.isChecked()
        
        print(f"Параметры: макс_каналов={max_channels}, врем_окно={time_window}, калман={use_kalman}")
        
        # Очистить фигуру
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        
        n_channels = min(len(self.processor.channel_names), max_channels)
        
        # Временное окно (первые X секунд или указанное)
        end_sample = min(int(time_window * self.processor.sfreq), self.processor.data.shape[1])
        time_subset = self.processor.time_axis[:end_sample]
        
        print(f"Будет построено {n_channels} каналов, {len(time_subset)} образцов")
        
        # Вычислить вертикальные смещения для укладки каналов
        offsets = np.linspace(0, (n_channels - 1) * 100, n_channels)[::-1]
        
        lines_plotted = 0
        for i in range(n_channels):
            # Получить данные канала
            channel_data = self.processor.data[i, :end_sample]
            
            # Применить фильтр Калмана, если включен
            if use_kalman:
                filtered_data = apply_kalman_filter(channel_data)
                print(f"Канал {i} ({self.processor.channel_names[i]}): фильтрован Калманом")
            else:
                filtered_data = channel_data
                print(f"Канал {i} ({self.processor.channel_names[i]}): исходные данные")
            
            # Нормализовать и масштабировать как в оригинальном скрипте
            signal_std = np.std(filtered_data)
            if signal_std > 0:
                normalized_signal = (filtered_data / signal_std) * 20
            else:
                normalized_signal = filtered_data
            
            # Добавить смещение для укладки
            y_data = normalized_signal + offsets[i]
            
            # Построить линию
            ax.plot(time_subset, y_data, color='black', linewidth=0.8, 
                   label=self.processor.channel_names[i])
            lines_plotted += 1
        
        # Настроить график
        ax.set_xlabel("Время (секунды)")
        ax.set_ylabel("Каналы")
        ax.set_yticks(offsets)
        ax.set_yticklabels(self.processor.channel_names[:n_channels])
        filter_status = "с фильтром Калмана" if use_kalman else "исходные данные"
        ax.set_title(f"Исходные данные ЭЭГ ({filter_status})")
        ax.grid(True, alpha=0.3)
        
        print(f"График создан с {lines_plotted} линиями")
        
        # Обновить канвас
        self.canvas.draw()
        print("=== ГРАФИК ИСХОДНЫХ ДАННЫХ ЗАВЕРШЕН ===")

class BipolarPlotWidget(InteractivePlotWidget):
    """Виджет для графиков биполярного монтажа"""
    
    def __init__(self):
        super().__init__("Биполярный монтаж")
    
    def update_plot(self):
        if not self.processor:
            return
        
        print("=== ОБНОВЛЕНИЕ БИПОЛЯРНОГО ГРАФИКА ===")
        max_channels = self.max_channels_spin.value()
        time_window = self.time_window_spin.value()
        use_kalman = self.kalman_checkbox.isChecked()
        
        if len(self.processor.bipolar_data) == 0:
            print("ОШИБКА: Нет данных биполярного монтажа!")
            return
        
        # Очистить фигуру
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        
        n_channels = min(len(self.processor.bipolar_names), max_channels)
        
        # Временное окно
        end_sample = min(int(time_window * self.processor.sfreq), self.processor.bipolar_data.shape[1])
        time_subset = self.processor.time_axis[:end_sample]
        
        print(f"Будет построено {n_channels} биполярных каналов")
        
        # Вычислить вертикальные смещения для укладки каналов
        offsets = np.linspace(0, (n_channels - 1) * 100, n_channels)[::-1]
        
        lines_plotted = 0
        for i in range(n_channels):
            # Получить биполярные данные канала
            channel_data = self.processor.bipolar_data[i, :end_sample]
            
            # Применить фильтр Калмана, если включен
            if use_kalman:
                filtered_data = apply_kalman_filter(channel_data)
            else:
                filtered_data = channel_data
            
            # Нормализовать и масштабировать как в оригинальном скрипте
            signal_std = np.std(filtered_data)
            if signal_std > 0:
                normalized_signal = (filtered_data / signal_std) * 20
            else:
                normalized_signal = filtered_data
            
            # Добавить смещение для укладки
            y_data = normalized_signal + offsets[i]
            
            # Построить линию
            ax.plot(time_subset, y_data, color='black', linewidth=0.8, 
                   label=self.processor.bipolar_names[i])
            lines_plotted += 1
        
        # Настроить график
        ax.set_xlabel("Время (секунды)")
        ax.set_ylabel("Биполярные каналы")
        ax.set_yticks(offsets)
        ax.set_yticklabels(self.processor.bipolar_names[:n_channels])
        filter_status = "с фильтром Калмана" if use_kalman else "исходные данные"
        ax.set_title(f"Биполярный монтаж Double-Banana ({filter_status})")
        ax.grid(True, alpha=0.3)
        
        print(f"Биполярный график создан с {lines_plotted} линиями")
        
        # Обновить канвас
        self.canvas.draw()
        print("=== БИПОЛЯРНЫЙ ГРАФИК ЗАВЕРШЕН ===")

class StackedEpochsPlotWidget(InteractivePlotWidget):
    """Виджет для графиков сложенных эпох"""
    
    def __init__(self):
        super().__init__("Сложенные эпохи")
        self.current_condition = None
        self.condition_buttons = {}
        
    def init_ui(self):
        """Переопределить инициализацию UI для добавления переключателей условий"""
        layout = QVBoxLayout()
        
        # Элементы управления
        controls_frame = QFrame()
        controls_frame.setFrameStyle(QFrame.Box)
        controls_layout = QGridLayout(controls_frame)
        
        # Максимальное количество каналов
        controls_layout.addWidget(QLabel("Макс. каналов:"), 0, 0)
        self.max_channels_spin = QSpinBox()
        self.max_channels_spin.setRange(1, 50)
        self.max_channels_spin.setValue(15)
        self.max_channels_spin.valueChanged.connect(self.update_plot)
        controls_layout.addWidget(self.max_channels_spin, 0, 1)
        
        # Максимальное количество эпох
        controls_layout.addWidget(QLabel("Макс. эпох:"), 0, 2)
        self.max_epochs_spin = QSpinBox()
        self.max_epochs_spin.setRange(1, 50)
        self.max_epochs_spin.setValue(10)
        self.max_epochs_spin.valueChanged.connect(self.update_plot)
        controls_layout.addWidget(self.max_epochs_spin, 0, 3)
        
        # Переключатель фильтра Калмана
        self.kalman_checkbox = QCheckBox("Фильтр Калмана")
        self.kalman_checkbox.setChecked(True)
        self.kalman_checkbox.stateChanged.connect(self.update_plot)
        controls_layout.addWidget(self.kalman_checkbox, 0, 4)
        
        layout.addWidget(controls_frame)
        
        # Добавить область для переключателей условий
        self.conditions_frame = QGroupBox("Выбор условия:")
        self.conditions_layout = QHBoxLayout(self.conditions_frame)
        layout.addWidget(self.conditions_frame)
        
        # Matplotlib фигура и канвас
        self.figure = Figure(figsize=(16, 10))
        self.canvas = FigureCanvas(self.figure)
        
        # Панель инструментов matplotlib
        self.toolbar = NavigationToolbar(self.canvas, self)
        
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        
        self.setLayout(layout)
    
    def set_processor(self, processor):
        """Установить процессор данных ЭЭГ и создать переключатели условий"""
        print(f"Установка процессора для {self.title}")
        self.processor = processor
        
        # Создать переключатели для каждого условия
        self.create_condition_buttons()
        
        # Выбрать первое условие по умолчанию
        if self.condition_buttons:
            first_condition = list(self.condition_buttons.keys())[0]
            self.condition_buttons[first_condition].setChecked(True)
            self.current_condition = first_condition
        
        self.update_plot()
    
    def create_condition_buttons(self):
        """Создать переключатели для всех доступных условий"""
        # Очистить предыдущие кнопки
        for button in self.condition_buttons.values():
            button.deleteLater()
        self.condition_buttons.clear()
        
        # Создать новые кнопки для каждого условия
        for condition in sorted(self.processor.annotation_groups.keys()):
            button = QCheckBox(f"{condition} ({len(self.processor.annotation_groups[condition])} эпох)")
            button.clicked.connect(lambda checked, cond=condition: self.on_condition_selected(cond, checked))
            self.condition_buttons[condition] = button
            self.conditions_layout.addWidget(button)
    
    def on_condition_selected(self, condition, checked):
        """Обработать выбор условия"""
        if checked:
            # Снять галочки с других кнопок (только одно условие за раз)
            for cond, button in self.condition_buttons.items():
                if cond != condition:
                    button.setChecked(False)
            
            self.current_condition = condition
            print(f"Выбрано условие: {condition}")
            self.update_plot()
        else:
            # Если снята галочка с текущего условия
            if condition == self.current_condition:
                self.current_condition = None
                self.figure.clear()
                self.canvas.draw()
    
    def update_plot(self):
        if not self.processor or not self.current_condition:
            return
        
        print(f"=== ОБНОВЛЕНИЕ ГРАФИКА СЛОЖЕННЫХ ЭПОХ для условия {self.current_condition} ===")
        max_channels = self.max_channels_spin.value()
        max_epochs = self.max_epochs_spin.value()
        use_kalman = self.kalman_checkbox.isChecked()
        
        if self.current_condition not in self.processor.annotation_groups:
            print(f"Условие '{self.current_condition}' не найдено")
            return
        
        epochs_info = self.processor.annotation_groups[self.current_condition][:max_epochs]
        n_epochs = len(epochs_info)
        
        print(f"Найдено {n_epochs} эпох для условия '{self.current_condition}'")
        
        if n_epochs == 0:
            print("Нет эпох для этого условия")
            return
        
        # Извлечь эпохи ТОЧНО как в оригинальном скрипте
        epoch_data_list = []
        epoch_labels = []
        samples_per_epoch_list = []
        
        for i, epoch in enumerate(epochs_info):
            start_sample = epoch['start_sample']
            end_sample = epoch['end_sample']
            length_samples = epoch['length_samples']
            
            print(f"Обработка эпохи {i}: сэмплы {start_sample}-{end_sample}, длина {length_samples}")
            
            # Проверить границы
            if start_sample < 0 or end_sample > self.processor.bipolar_data.shape[1]:
                print(f"Пропуск эпохи {i}: неверные границы")
                continue
            
            if length_samples <= 0:
                print(f"Пропуск эпохи {i}: нулевая длина")
                continue
            
            # Извлечь данные эпохи из биполярных данных
            epoch_data = self.processor.bipolar_data[:, start_sample:end_sample]
            
            if epoch_data.shape[1] > 0:  # Убедиться, что есть данные
                epoch_data_list.append(epoch_data)
                epoch_labels.append(f"{self.current_condition} #{i+1}")
                samples_per_epoch_list.append(epoch_data.shape[1])
                print(f"Добавлена эпоха {i} с формой {epoch_data.shape}")
            else:
                print(f"Пропуск эпохи {i}: пустые данные")
        
        if not epoch_data_list:
            print("Не найдено действительных данных эпох")
            return
        
        print(f"Будет объединено {len(epoch_data_list)} эпох")
        
        # Сложить эпохи горизонтально (объединить по времени) - ТОЧНО как в оригинале
        stacked_data = np.concatenate(epoch_data_list, axis=1)
        print(f"Сложенные данные: форма {stacked_data.shape}")
        
        # Создать временную ось для сложенных данных
        total_samples = stacked_data.shape[1]
        time_axis = np.arange(total_samples) / self.processor.sfreq
        
        # Ограничить каналы
        n_channels = min(len(self.processor.bipolar_names), max_channels)
        if n_channels == 0:
            print("Нет биполярных каналов")
            return
        
        # Очистить фигуру
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        
        # Вычислить вертикальные смещения и нормализацию ТОЧНО как в оригинале
        offsets = np.linspace(0, (n_channels - 1) * 100, n_channels)[::-1]
        
        # Применить фильтрацию Калмана к сложенным данным
        filtered_stacked = np.zeros_like(stacked_data[:n_channels])
        for ch_idx in range(n_channels):
            if use_kalman:
                filtered_stacked[ch_idx] = apply_kalman_filter(stacked_data[ch_idx])
            else:
                filtered_stacked[ch_idx] = stacked_data[ch_idx]
        
        # Нормализовать и отобразить каналы ТОЧНО как в оригинале
        lines_plotted = 0
        stds = np.std(filtered_stacked, axis=1, keepdims=True)
        stds[stds == 0] = 1.0  # Предотвратить деление на ноль
        
        for ch_idx in range(n_channels):
            # Нормализация как в оригинальном скрипте
            normalized_signal = (filtered_stacked[ch_idx] / stds[ch_idx, 0]) * 20
            
            # Добавить смещение для укладки
            y_data = normalized_signal + offsets[ch_idx]
            
            # Построить линию
            ax.plot(time_axis, y_data, color='black', linewidth=0.6)
            lines_plotted += 1
        
        # Добавить разметку эпох точно как в оригинальном скрипте
        current_sample = 0
        for i, (label, samples_in_epoch) in enumerate(zip(epoch_labels, samples_per_epoch_list)):
            # Вычислить центр эпохи для размещения текста
            center_time = (current_sample + samples_in_epoch / 2) / self.processor.sfreq
            
            # Добавить текст с названием эпохи
            ax.text(
                center_time,
                offsets[0] + 20,
                label,
                fontsize=8,
                ha="center",
                va="bottom",
                rotation=90,
            )
            
            # Добавить вертикальную линию границы эпохи (кроме последней)
            if i < len(epoch_labels) - 1:
                boundary_time = (current_sample + samples_in_epoch) / self.processor.sfreq
                ax.axvline(boundary_time, color="gray", linestyle="--", linewidth=0.6, alpha=0.5)
            
            current_sample += samples_in_epoch
        
        # Настроить график с заголовком условия
        ax.set_xlabel("Время (секунды)")
        ax.set_ylabel("Каналы")
        ax.set_yticks(offsets)
        ax.set_yticklabels([self.processor.bipolar_names[i] for i in range(n_channels)])
        ax.set_title(f"Condition: {self.current_condition}")  # Заголовок с названием условия
        ax.grid(False)
        
        self.figure.tight_layout()
        self.canvas.draw()
        
        print(f"График построен: {lines_plotted} каналов, {len(epoch_labels)} эпох")

class PowerSpectrumPlotWidget(InteractivePlotWidget):
    """Виджет для графиков спектра мощности"""
    
    def __init__(self):
        super().__init__("Спектр мощности")
    
    def update_plot(self):
        if not self.processor:
            return
        
        print("=== ОБНОВЛЕНИЕ ГРАФИКА СПЕКТРА МОЩНОСТИ ===")
        max_channels = self.max_channels_spin.value()
        
        from scipy import signal
        
        n_channels = min(len(self.processor.channel_names), max_channels)
        
        # Очистить фигуру
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        
        print(f"Будет построен спектр мощности для {n_channels} каналов")
        
        for i in range(n_channels):
            # Вычислить спектральную плотность мощности
            freqs, psd = signal.welch(
                self.processor.data[i], 
                fs=self.processor.sfreq, 
                nperseg=min(2048, self.processor.data.shape[1]//4)
            )
            
            # Сосредоточиться на диапазоне частот ЭЭГ (0.5-50 Гц)
            freq_mask = (freqs >= 0.5) & (freqs <= 50)
            
            ax.plot(freqs[freq_mask], 10 * np.log10(psd[freq_mask]),  # Преобразовать в дБ
                   linewidth=2, label=self.processor.channel_names[i])
        
        # Настроить график
        ax.set_xlabel("Частота (Гц)")
        ax.set_ylabel("Мощность (дБ)")
        ax.set_title("Спектральная плотность мощности")
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        print("График спектра мощности завершен")
        
        # Обновить канвас
        self.figure.tight_layout()
        self.canvas.draw()
        print("=== ГРАФИК СПЕКТРА МОЩНОСТИ ЗАВЕРШЕН ===")

class MainWindow(QMainWindow):
    def __init__(self, default_edf_path=None):
        super().__init__()
        self.default_edf_path = default_edf_path
        self.processor = None
        self.load_thread = None
        self.init_ui()
        
        # Автоматически загрузить файл по умолчанию, если предоставлен
        if self.default_edf_path and os.path.exists(self.default_edf_path):
            QTimer.singleShot(100, self.load_default_file)
        else:
            # Показать сообщение о том, что нужно выбрать файл
            QTimer.singleShot(500, self.show_file_selection_hint)
    
    def init_ui(self):
        self.setWindowTitle("Просмотрщик EDF файлов")
        self.setGeometry(100, 100, 1600, 1000)
        
        # Центральный виджет
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Основная компоновка
        main_layout = QVBoxLayout(central_widget)
        
        # Верхние элементы управления
        top_controls = QHBoxLayout()
        
        # Выбор файла
        file_group = QGroupBox("Файл")
        file_layout = QHBoxLayout(file_group)
        
        self.file_label = QLabel("Файл не загружен")
        self.file_label.setWordWrap(True)
        file_layout.addWidget(self.file_label)
        
        self.load_button = QPushButton("Загрузить EDF файл")
        self.load_button.clicked.connect(self.load_file)
        file_layout.addWidget(self.load_button)
        
        top_controls.addWidget(file_group)
        
        # Информационная панель
        info_group = QGroupBox("Информация о файле")
        info_layout = QVBoxLayout(info_group)
        
        self.info_text = QTextEdit()
        self.info_text.setMaximumHeight(120)
        self.info_text.setReadOnly(True)
        info_layout.addWidget(self.info_text)
        
        top_controls.addWidget(info_group)
        
        # Полоса прогресса
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        top_controls.addWidget(self.progress_bar)
        
        main_layout.addLayout(top_controls)
        
        # Виджет вкладок для различных графиков
        self.tab_widget = QTabWidget()
        
        # Создать виджеты графиков
        self.raw_data_widget = RawDataPlotWidget()
        self.bipolar_widget = BipolarPlotWidget()
        self.stacked_epochs_widget = StackedEpochsPlotWidget()
        self.power_spectrum_widget = PowerSpectrumPlotWidget()
        
        # Добавить вкладки с русскими названиями
        self.tab_widget.addTab(self.raw_data_widget, "Исходные данные")
        self.tab_widget.addTab(self.bipolar_widget, "Биполярный монтаж")
        self.tab_widget.addTab(self.stacked_epochs_widget, "Сложенные эпохи")
        self.tab_widget.addTab(self.power_spectrum_widget, "Спектр мощности")
        
        main_layout.addWidget(self.tab_widget)
        
        # Строка состояния
        self.statusBar().showMessage("Готов")
        
        # Добавить объяснительный текст
        self.add_help_text()
    
    def add_help_text(self):
        """Добавить объяснительный текст о том, что показывают графики"""
        help_group = QGroupBox("Справка")
        help_layout = QVBoxLayout(help_group)
        
        help_text = QTextEdit()
        help_text.setMaximumHeight(80)
        help_text.setReadOnly(True)
        help_text.setHtml("""
        <b>Исходные данные:</b> Сырые сигналы ЭЭГ с каждого электрода<br>
        <b>Биполярный монтаж:</b> Разности между парами электродов (улучшает локализацию)<br>
        <b>Сложенные эпохи:</b> Временные сегменты сигнала, объединенные по типу события<br>
        <b>Спектр мощности:</b> Частотный анализ сигналов (показывает ритмы мозга)
        """)
        help_layout.addWidget(help_text)
        
        # Добавить справку в нижнюю часть окна
        self.centralWidget().layout().addWidget(help_group)
    
    def load_default_file(self):
        """Загрузить файл EDF по умолчанию"""
        if self.default_edf_path:
            self.load_edf_file(self.default_edf_path)
    
    def show_file_selection_hint(self):
        """Показать подсказку о выборе файла"""
        self.statusBar().showMessage("Выберите EDF файл для начала работы")
        self.file_label.setText("Файл не выбран - нажмите 'Выбрать файл'")
    
    def load_file(self):
        """Открыть диалог выбора файла и загрузить выбранный EDF файл"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Выберите EDF файл",
            str(Path(self.default_edf_path).parent) if self.default_edf_path else "",
            "EDF файлы (*.edf);;Все файлы (*)"
        )
        
        if file_path:
            self.load_edf_file(file_path)
    
    def load_edf_file(self, file_path):
        """Загрузить EDF файл в фоновом потоке"""
        if self.load_thread and self.load_thread.isRunning():
            return
        
        self.file_label.setText(f"Загрузка: {os.path.basename(file_path)}")
        self.load_button.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Неопределенный прогресс
        
        # Создать и запустить поток загрузки
        self.load_thread = LoadEDFThread(file_path)
        self.load_thread.finished.connect(self.on_data_loaded)
        self.load_thread.error.connect(self.on_load_error)
        self.load_thread.progress.connect(self.on_load_progress)
        self.load_thread.start()
    
    def on_data_loaded(self, processor):
        """Обработать успешную загрузку данных"""
        print("=== ДАННЫЕ УСПЕШНО ЗАГРУЖЕНЫ ===")
        self.processor = processor
        
        # Обновить информацию о файле
        self.file_label.setText(f"Загружен: {os.path.basename(self.load_thread.edf_path)}")
        
        info_text = f"""
Каналы: {len(processor.channel_names)}
Частота дискретизации: {processor.sfreq} Гц
Продолжительность: {processor.data.shape[1] / processor.sfreq:.2f} секунд
Аннотации: {len(processor.annotations)}
Биполярные каналы: {len(processor.bipolar_names)}

Доступные типы аннотаций:
"""
        for ann_type, epochs in processor.annotation_groups.items():
            info_text += f"• {ann_type}: {len(epochs)} эпох\n"
        
        self.info_text.setText(info_text)
        
        # Установить процессор для всех виджетов
        print("Установка процессора для RawDataPlotWidget")
        self.raw_data_widget.set_processor(processor)
        print("Установка процессора для BipolarPlotWidget")
        self.bipolar_widget.set_processor(processor)
        print("Установка процессора для StackedEpochsPlotWidget")
        self.stacked_epochs_widget.set_processor(processor)
        print("Установка процессора для PowerSpectrumPlotWidget")
        self.power_spectrum_widget.set_processor(processor)
        print("=== ВСЕ ВИДЖЕТЫ ОБНОВЛЕНЫ ===")
        
        # Очистка
        self.load_button.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.statusBar().showMessage("EDF файл успешно загружен")
    
    def on_load_error(self, error_msg):
        """Обработать ошибки загрузки"""
        QMessageBox.critical(self, "Ошибка", error_msg)
        self.file_label.setText("Файл не загружен")
        self.load_button.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.statusBar().showMessage("Ошибка загрузки файла")
    
    def on_load_progress(self, message):
        """Обработать обновления прогресса"""
        self.statusBar().showMessage(message)
    
    def closeEvent(self, event):
        """Очистить при закрытии приложения"""
        # Остановить любые работающие потоки
        if self.load_thread and self.load_thread.isRunning():
            self.load_thread.terminate()
            self.load_thread.wait()
        
        event.accept()

def main():
    parser = argparse.ArgumentParser(description='Просмотрщик EDF файлов - Приложение PyQt5')
    parser.add_argument('edf_file', nargs='?', 
                       default=None,  # Убираем путь по умолчанию для exe
                       help='Путь к EDF файлу')
    
    args = parser.parse_args()
    
    app = QApplication(sys.argv)
    app.setApplicationName("Просмотрщик EDF файлов")
    
    # Установить шрифт приложения
    font = QFont()
    font.setPointSize(9)
    app.setFont(font)
    
    # Создать и показать главное окно
    main_window = MainWindow(args.edf_file)
    main_window.show()
    
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()