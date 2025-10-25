#!/usr/bin/env python3
"""
EDF Desktop Viewer - PyQt5 desktop application for viewing EDF files
with tabbed interface and embedded Plotly interactive graphs
"""

import sys
import os
import argparse
from pathlib import Path
import tempfile

import numpy as np
import plotly.graph_objects as go
import plotly.offline as pyo
import mne

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTabWidget, QLabel, QSpinBox, QComboBox, QPushButton, QFileDialog,
    QTextEdit, QGroupBox, QGridLayout, QMessageBox, QProgressBar,
    QFrame
)
from PyQt5.QtCore import QThread, pyqtSignal, QTimer, QUrl
from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.QtGui import QFont

def apply_kalman_filter(data, process_variance=1e-4, measurement_variance=1e-2):
    """Apply Kalman filter for noise reduction with improved parameters"""
    n_samples = len(data)
    
    # Initialize arrays
    x_hat = np.zeros(n_samples)  # State estimate
    P = np.ones(n_samples)       # Error covariance
    
    # Process and measurement noise (adaptive based on signal variance)
    Q = process_variance
    R = measurement_variance * (np.var(data) + 1e-6)  # Adaptive measurement noise
    
    # Initial values
    x_hat[0] = data[0]
    P[0] = R
    
    for k in range(1, n_samples):
        # Predict step
        x_hat_minus = x_hat[k-1]
        P_minus = P[k-1] + Q
        
        # Update step
        K = P_minus / (P_minus + R)  # Kalman gain
        x_hat[k] = x_hat_minus + K * (data[k] - x_hat_minus)
        P[k] = (1 - K) * P_minus
    
    return x_hat

def create_bipolar_montage(raw_data, channel_names):
    """Create double-banana bipolar montage with full electrode pairs"""
    # Complete double-banana montage pairs from original script
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
    
    # Create channel index mapping
    channel_index = {name: idx for idx, name in enumerate(channel_names)}
    
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
        else:
            missing_pairs.append(f"{ch1}-{ch2}")
    
    if missing_pairs:
        print(f"Warning: Missing channels for pairs: {', '.join(missing_pairs)}")
    
    if not bipolar_data:
        # Fallback to simple adjacent pairs if no standard pairs found
        for i in range(min(len(channel_names)-1, 10)):
            bipolar_signal = raw_data[i] - raw_data[i+1]
            bipolar_data.append(bipolar_signal)
            bipolar_names.append(f"{channel_names[i]}-{channel_names[i+1]}")
    
    return np.array(bipolar_data), bipolar_names

class EEGDataProcessor:
    def __init__(self, edf_data):
        self.edf_data = edf_data
        self.data = edf_data['data']
        self.channel_names = edf_data['channel_names']
        self.sfreq = edf_data['sfreq']
        self.annotations = edf_data['annotations']
        
        # Create bipolar montage
        self.bipolar_data, self.bipolar_names = create_bipolar_montage(
            self.data, self.channel_names
        )
        
        # Time axis
        self.time_axis = np.arange(self.data.shape[1]) / self.sfreq
        
        # Process annotations
        self.annotation_groups = self._process_annotations()
    
    def _process_annotations(self):
        """Process annotations and group by description"""
        groups = {}
        
        for i, (onset, duration, description) in enumerate(
            zip(self.annotations.onset, self.annotations.duration, self.annotations.description)
        ):
            if description not in groups:
                groups[description] = []
            
            # Find the time window for this annotation
            start_sample = int(onset * self.sfreq)
            end_sample = int((onset + duration) * self.sfreq)
            
            groups[description].append({
                'onset': onset,
                'duration': duration,
                'start_sample': start_sample,
                'end_sample': end_sample,
                'index': i
            })
        
        return groups
    
    def create_raw_data_plot(self, max_channels=15, time_window=30):
        """Create plot for raw EEG data with proper scaling"""
        print("=== RAW DATA PLOT DEBUG ===")
        print(f"Total channels available: {len(self.channel_names)}")
        print(f"Channel names: {self.channel_names[:5]}...")  # First 5 channels
        print(f"Data shape: {self.data.shape}")
        print(f"Max channels requested: {max_channels}")
        print(f"Time window: {time_window} seconds")
        
        n_channels = min(len(self.channel_names), max_channels)
        print(f"Will plot {n_channels} channels")
        
        # Time window (first X seconds or specified)
        end_sample = min(int(time_window * self.sfreq), self.data.shape[1])
        time_subset = self.time_axis[:end_sample]
        print(f"Time subset: {len(time_subset)} samples from 0 to {time_subset[-1]:.2f} seconds")
        
        # Create figure with proper scaling like original script
        fig = go.Figure()
        
        # Calculate vertical offsets for stacking channels
        offsets = np.linspace(0, (n_channels - 1) * 100, n_channels)[::-1]
        print(f"Channel offsets: {offsets[:3]}...")  # First 3 offsets
        
        for i in range(n_channels):
            # Get raw data for this channel
            raw_channel_data = self.data[i, :end_sample]
            print(f"Channel {i} ({self.channel_names[i]}): min={np.min(raw_channel_data):.6f}, max={np.max(raw_channel_data):.6f}, std={np.std(raw_channel_data):.6f}")
            
            # Apply Kalman filter
            filtered_data = apply_kalman_filter(raw_channel_data)
            print(f"After Kalman filter: min={np.min(filtered_data):.6f}, max={np.max(filtered_data):.6f}, std={np.std(filtered_data):.6f}")
            
            # Normalize and scale like in original script
            signal_std = np.std(filtered_data)
            if signal_std > 0:
                normalized_signal = (filtered_data / signal_std) * 20
                print(f"After normalization (*20): min={np.min(normalized_signal):.2f}, max={np.max(normalized_signal):.2f}")
            else:
                normalized_signal = filtered_data
                print(f"No normalization (std=0): min={np.min(normalized_signal):.6f}, max={np.max(normalized_signal):.6f}")
            
            # Add offset for stacking
            y_data = normalized_signal + offsets[i]
            print(f"After offset (+{offsets[i]}): min={np.min(y_data):.2f}, max={np.max(y_data):.2f}")
            
            fig.add_trace(
                go.Scatter(
                    x=time_subset,
                    y=y_data,
                    mode='lines',
                    name=self.channel_names[i],
                    line=dict(width=0.8, color='black'),
                    showlegend=False
                )
            )
            
            if i >= 2:  # Only show debug for first 3 channels
                break
        
        print(f"Created figure with {len(fig.data)} traces")
        print("=== END RAW DATA PLOT DEBUG ===")
        
        fig.update_layout(
            height=max(600, n_channels * 40),
            title="Raw EEG Data (Kalman Filtered)",
            xaxis_title="Time (seconds)",
            yaxis_title="Channels",
            yaxis=dict(
                tickvals=offsets,
                ticktext=self.channel_names[:n_channels],
                showgrid=False
            ),
            xaxis=dict(showgrid=True, gridcolor='lightgray'),
            showlegend=False,
            plot_bgcolor='white'
        )
        
        return fig
    
    def create_bipolar_plot(self, max_channels=15, time_window=30):
        """Create plot for bipolar montage data with proper scaling"""
        if len(self.bipolar_data) == 0:
            fig = go.Figure()
            fig.add_annotation(
                text="No bipolar montage channels available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig
        
        n_channels = min(len(self.bipolar_names), max_channels)
        
        # Time window
        end_sample = min(int(time_window * self.sfreq), self.bipolar_data.shape[1])
        time_subset = self.time_axis[:end_sample]
        
        # Create figure with proper scaling
        fig = go.Figure()
        
        # Calculate vertical offsets for stacking channels
        offsets = np.linspace(0, (n_channels - 1) * 100, n_channels)[::-1]
        
        for i in range(n_channels):
            # Apply Kalman filter
            filtered_data = apply_kalman_filter(self.bipolar_data[i, :end_sample])
            
            # Normalize and scale like in original script
            signal_std = np.std(filtered_data)
            if signal_std > 0:
                normalized_signal = (filtered_data / signal_std) * 20
            else:
                normalized_signal = filtered_data
            
            # Add offset for stacking
            y_data = normalized_signal + offsets[i]
            
            fig.add_trace(
                go.Scatter(
                    x=time_subset,
                    y=y_data,
                    mode='lines',
                    name=self.bipolar_names[i],
                    line=dict(width=0.8, color='black'),
                    showlegend=False
                )
            )
        
        fig.update_layout(
            height=max(600, n_channels * 40),
            title="Bipolar Montage (Double-Banana)",
            xaxis_title="Time (seconds)",
            yaxis_title="Channels",
            yaxis=dict(
                tickvals=offsets,
                ticktext=self.bipolar_names[:n_channels],
                showgrid=False
            ),
            xaxis=dict(showgrid=True, gridcolor='lightgray'),
            showlegend=False,
            plot_bgcolor='white'
        )
        
        return fig
    
    def create_stacked_epochs_plot(self, annotation_type, max_epochs=10, max_channels=15):
        """Create stacked epochs plot exactly like in original script"""
        if annotation_type not in self.annotation_groups:
            fig = go.Figure()
            fig.add_annotation(
                text=f"No annotations of type '{annotation_type}' found",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig
        
        epochs_info = self.annotation_groups[annotation_type][:max_epochs]
        n_epochs = len(epochs_info)
        
        if n_epochs == 0:
            fig = go.Figure()
            fig.add_annotation(
                text="No epochs available for this annotation type",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig
        
        # Extract epochs and stack them exactly like original script
        epoch_data_list = []
        epoch_labels = []
        
        for i, epoch in enumerate(epochs_info):
            start_sample = max(0, epoch['start_sample'])
            end_sample = min(self.bipolar_data.shape[1], epoch['end_sample'])
            
            if end_sample > start_sample:
                epoch_data = self.bipolar_data[:, start_sample:end_sample]
                epoch_data_list.append(epoch_data)
                epoch_labels.append(f"{annotation_type} #{i+1}")
        
        if not epoch_data_list:
            fig = go.Figure()
            fig.add_annotation(
                text="No valid epoch data found",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig
        
        # Stack epochs horizontally (concatenate in time) - EXACTLY like original
        stacked_data = np.concatenate(epoch_data_list, axis=1)
        
        # Create time axis for stacked data
        total_samples = stacked_data.shape[1]
        time_axis = np.arange(total_samples) / self.sfreq
        
        # Limit channels
        n_channels = min(len(self.bipolar_names), max_channels)
        if n_channels == 0:
            fig = go.Figure()
            fig.add_annotation(
                text="No bipolar channels available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig
        
        # Apply Kalman filtering to stacked data
        filtered_stacked = np.zeros_like(stacked_data[:n_channels])
        for ch_idx in range(n_channels):
            filtered_stacked[ch_idx] = apply_kalman_filter(stacked_data[ch_idx])
        
        # Create figure with proper scaling like original script
        fig = go.Figure()
        
        # Calculate vertical offsets and normalization EXACTLY like original
        offsets = np.linspace(0, (n_channels - 1) * 100, n_channels)[::-1]
        
        # Calculate standard deviations for normalization
        stds = np.std(filtered_stacked, axis=1, keepdims=True)
        stds[stds == 0] = 1.0  # Prevent division by zero
        
        # Scale and offset each channel
        for ch_idx in range(n_channels):
            # Normalize and scale EXACTLY like original script
            normalized_signal = (filtered_stacked[ch_idx] / stds[ch_idx, 0]) * 20
            y_data = normalized_signal + offsets[ch_idx]
            
            fig.add_trace(
                go.Scatter(
                    x=time_axis,
                    y=y_data,
                    mode='lines',
                    name=self.bipolar_names[ch_idx],
                    line=dict(width=0.6, color='black'),
                    showlegend=False
                )
            )
        
        # Add epoch boundaries and labels EXACTLY like original
        current_time = 0
        samples_per_epoch = [ep.shape[1] for ep in epoch_data_list]
        epoch_duration_list = [samples / self.sfreq for samples in samples_per_epoch]
        
        for i, (duration, label) in enumerate(zip(epoch_duration_list, epoch_labels)):
            # Add epoch label at the top
            center_time = current_time + duration/2
            fig.add_annotation(
                x=center_time,
                y=offsets[0] + 20,  # Above the top channel
                text=label,
                showarrow=False,
                textangle=90,
                font=dict(size=8),
                yref="y"
            )
            
            # Add boundary line (except for the last epoch)
            if i < len(epoch_duration_list) - 1:
                current_time += duration
                fig.add_vline(
                    x=current_time,
                    line=dict(color="gray", width=0.6, dash="dash"),
                    opacity=0.5
                )
            else:
                current_time += duration
        
        fig.update_layout(
            height=max(600, n_channels * 30),
            title=f"Stacked Epochs: {annotation_type} ({n_epochs} epochs)",
            xaxis_title="Time (seconds)",
            yaxis_title="Channels",
            yaxis=dict(
                tickvals=offsets,
                ticktext=self.bipolar_names[:n_channels],
                showgrid=False
            ),
            xaxis=dict(showgrid=True, gridcolor='lightgray'),
            showlegend=False,
            plot_bgcolor='white'
        )
        
        return fig
    
    def create_power_spectrum_plot(self, max_channels=10):
        """Create power spectrum plot"""
        from scipy import signal
        
        n_channels = min(len(self.channel_names), max_channels)
        
        fig = go.Figure()
        
        for i in range(n_channels):
            # Calculate power spectral density
            freqs, psd = signal.welch(
                self.data[i], 
                fs=self.sfreq, 
                nperseg=min(2048, self.data.shape[1]//4)
            )
            
            # Focus on EEG frequency range (0.5-50 Hz)
            freq_mask = (freqs >= 0.5) & (freqs <= 50)
            
            fig.add_trace(
                go.Scatter(
                    x=freqs[freq_mask],
                    y=10 * np.log10(psd[freq_mask]),  # Convert to dB
                    mode='lines',
                    name=self.channel_names[i],
                    line=dict(width=2)
                )
            )
        
        fig.update_layout(
            title="Power Spectral Density",
            xaxis_title="Frequency (Hz)",
            yaxis_title="Power (dB)",
            height=600
        )
        
        return fig

class LoadEDFThread(QThread):
    """Thread for loading EDF files in background"""
    finished = pyqtSignal(object)
    error = pyqtSignal(str)
    progress = pyqtSignal(str)
    
    def __init__(self, edf_path):
        super().__init__()
        self.edf_path = edf_path
    
    def run(self):
        try:
            self.progress.emit("Loading EDF file...")
            
            # Load EDF file using MNE
            raw = mne.io.read_raw_edf(self.edf_path, preload=True, verbose=False)
            
            self.progress.emit("Processing data...")
            
            # Get data and channel information
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
            
            self.progress.emit("Creating EEG processor...")
            processor = EEGDataProcessor(edf_data)
            
            self.progress.emit("Ready!")
            self.finished.emit(processor)
            
        except Exception as e:
            self.error.emit(f"Error loading EDF file: {str(e)}")

class EEGPlotWidget(QWidget):
    """Widget for displaying EEG plots with controls"""
    
    def __init__(self):
        super().__init__()
        self.processor = None
        self.temp_files = []  # Keep track of temp files
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout()
        
        # Controls
        controls_frame = QFrame()
        controls_frame.setFrameStyle(QFrame.Box)
        controls_layout = QGridLayout(controls_frame)
        
        # Max channels control
        controls_layout.addWidget(QLabel("Max Channels:"), 0, 0)
        self.max_channels_spin = QSpinBox()
        self.max_channels_spin.setRange(1, 50)
        self.max_channels_spin.setValue(15)
        self.max_channels_spin.valueChanged.connect(self.update_plot)
        controls_layout.addWidget(self.max_channels_spin, 0, 1)
        
        # Time window control
        controls_layout.addWidget(QLabel("Time Window (s):"), 0, 2)
        self.time_window_spin = QSpinBox()
        self.time_window_spin.setRange(1, 300)
        self.time_window_spin.setValue(30)
        self.time_window_spin.valueChanged.connect(self.update_plot)
        controls_layout.addWidget(self.time_window_spin, 0, 3)
        
        # Annotation type control (will be enabled when data is loaded)
        controls_layout.addWidget(QLabel("Annotation Type:"), 1, 0)
        self.annotation_combo = QComboBox()
        self.annotation_combo.setEnabled(False)
        self.annotation_combo.currentTextChanged.connect(self.update_plot)
        controls_layout.addWidget(self.annotation_combo, 1, 1, 1, 2)
        
        # Max epochs control
        controls_layout.addWidget(QLabel("Max Epochs:"), 1, 3)
        self.max_epochs_spin = QSpinBox()
        self.max_epochs_spin.setRange(1, 50)
        self.max_epochs_spin.setValue(10)
        self.max_epochs_spin.valueChanged.connect(self.update_plot)
        controls_layout.addWidget(self.max_epochs_spin, 1, 4)
        
        layout.addWidget(controls_frame)
        
        # Web view for Plotly
        self.web_view = QWebEngineView()
        layout.addWidget(self.web_view)
        
        self.setLayout(layout)
    
    def set_processor(self, processor):
        """Set the EEG data processor"""
        self.processor = processor
        
        # Update annotation combo
        if processor.annotation_groups:
            self.annotation_combo.clear()
            self.annotation_combo.addItems(list(processor.annotation_groups.keys()))
            self.annotation_combo.setEnabled(True)
        
        self.update_plot()
    
    def update_plot(self):
        """Update the current plot"""
        # This will be overridden by subclasses
        pass
    
    def cleanup_temp_files(self):
        """Clean up temporary HTML files"""
        for temp_file in self.temp_files:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            except Exception:
                pass
        self.temp_files.clear()

class RawDataPlotWidget(EEGPlotWidget):
    """Widget for raw data plots"""
    
    def update_plot(self):
        if not self.processor:
            print("ERROR: No processor available for Raw Data plot")
            return
        
        print("=== UPDATING RAW DATA PLOT ===")
        max_channels = self.max_channels_spin.value()
        time_window = self.time_window_spin.value()
        print(f"Parameters: max_channels={max_channels}, time_window={time_window}")
        
        fig = self.processor.create_raw_data_plot(max_channels, time_window)
        print(f"Figure created with {len(fig.data)} traces")
        
        # Save to temporary HTML file with embedded Plotly.js
        with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as f:
            self.temp_files.append(f.name)
            print(f"Saving plot to: {f.name}")
            
            # Create HTML with embedded Plotly.js
            config = {
                'displayModeBar': True,
                'displaylogo': False,
                'modeBarButtonsToRemove': ['pan2d', 'lasso2d']
            }
            
            pyo.plot(fig, filename=f.name, auto_open=False, include_plotlyjs=True, config=config)
            print("Plot saved successfully")
            
            self.web_view.load(QUrl.fromLocalFile(f.name))
            print("=== RAW DATA PLOT UPDATE COMPLETE ===")

class BipolarPlotWidget(EEGPlotWidget):
    """Widget for bipolar montage plots"""
    
    def update_plot(self):
        if not self.processor:
            print("ERROR: No processor available for Bipolar plot")
            return
        
        print("=== UPDATING BIPOLAR PLOT ===")
        max_channels = self.max_channels_spin.value()
        time_window = self.time_window_spin.value()
        print(f"Parameters: max_channels={max_channels}, time_window={time_window}")
        
        fig = self.processor.create_bipolar_plot(max_channels, time_window)
        print(f"Bipolar figure created with {len(fig.data)} traces")
        
        # Save to temporary HTML file with embedded Plotly.js
        with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as f:
            self.temp_files.append(f.name)
            print(f"Saving bipolar plot to: {f.name}")
            
            config = {
                'displayModeBar': True,
                'displaylogo': False,
                'modeBarButtonsToRemove': ['pan2d', 'lasso2d']
            }
            
            pyo.plot(fig, filename=f.name, auto_open=False, include_plotlyjs=True, config=config)
            self.web_view.load(QUrl.fromLocalFile(f.name))
            print("=== BIPOLAR PLOT UPDATE COMPLETE ===")

class StackedEpochsPlotWidget(EEGPlotWidget):
    """Widget for stacked epochs plots"""
    
    def update_plot(self):
        if not self.processor:
            print("ERROR: No processor available for Stacked Epochs plot")
            return
        
        print("=== UPDATING STACKED EPOCHS PLOT ===")
        max_channels = self.max_channels_spin.value()
        max_epochs = self.max_epochs_spin.value()
        annotation_type = self.annotation_combo.currentText()
        print(f"Parameters: max_channels={max_channels}, max_epochs={max_epochs}, annotation_type={annotation_type}")
        
        if not annotation_type:
            print("ERROR: No annotation type selected")
            return
        
        fig = self.processor.create_stacked_epochs_plot(annotation_type, max_epochs, max_channels)
        print(f"Stacked epochs figure created with {len(fig.data)} traces")
        
        # Save to temporary HTML file with embedded Plotly.js
        with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as f:
            self.temp_files.append(f.name)
            print(f"Saving stacked epochs plot to: {f.name}")
            
            config = {
                'displayModeBar': True,
                'displaylogo': False,
                'modeBarButtonsToRemove': ['pan2d', 'lasso2d']
            }
            
            pyo.plot(fig, filename=f.name, auto_open=False, include_plotlyjs=True, config=config)
            self.web_view.load(QUrl.fromLocalFile(f.name))
            print("=== STACKED EPOCHS PLOT UPDATE COMPLETE ===")

class PowerSpectrumPlotWidget(EEGPlotWidget):
    """Widget for power spectrum plots"""
    
    def update_plot(self):
        if not self.processor:
            print("ERROR: No processor available for Power Spectrum plot")
            return
        
        print("=== UPDATING POWER SPECTRUM PLOT ===")
        max_channels = self.max_channels_spin.value()
        print(f"Parameters: max_channels={max_channels}")
        
        fig = self.processor.create_power_spectrum_plot(max_channels)
        print(f"Power spectrum figure created with {len(fig.data)} traces")
        
        # Save to temporary HTML file with embedded Plotly.js
        with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as f:
            self.temp_files.append(f.name)
            print(f"Saving power spectrum plot to: {f.name}")
            
            config = {
                'displayModeBar': True,
                'displaylogo': False,
                'modeBarButtonsToRemove': ['pan2d', 'lasso2d']
            }
            
            pyo.plot(fig, filename=f.name, auto_open=False, include_plotlyjs=True, config=config)
            self.web_view.load(QUrl.fromLocalFile(f.name))
            print("=== POWER SPECTRUM PLOT UPDATE COMPLETE ===")

class MainWindow(QMainWindow):
    def __init__(self, default_edf_path=None):
        super().__init__()
        self.default_edf_path = default_edf_path
        self.processor = None
        self.load_thread = None
        self.init_ui()
        
        # Auto-load default file if provided
        if self.default_edf_path and os.path.exists(self.default_edf_path):
            QTimer.singleShot(100, self.load_default_file)
    
    def init_ui(self):
        self.setWindowTitle("EDF Desktop Viewer")
        self.setGeometry(100, 100, 1400, 900)
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QVBoxLayout(central_widget)
        
        # Top controls
        top_controls = QHBoxLayout()
        
        # File selection
        file_group = QGroupBox("File")
        file_layout = QHBoxLayout(file_group)
        
        self.file_label = QLabel("No file loaded")
        self.file_label.setWordWrap(True)
        file_layout.addWidget(self.file_label)
        
        self.load_button = QPushButton("Load EDF File")
        self.load_button.clicked.connect(self.load_file)
        file_layout.addWidget(self.load_button)
        
        top_controls.addWidget(file_group)
        
        # Info panel
        info_group = QGroupBox("File Information")
        info_layout = QVBoxLayout(info_group)
        
        self.info_text = QTextEdit()
        self.info_text.setMaximumHeight(100)
        self.info_text.setReadOnly(True)
        info_layout.addWidget(self.info_text)
        
        top_controls.addWidget(info_group)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        top_controls.addWidget(self.progress_bar)
        
        main_layout.addLayout(top_controls)
        
        # Tab widget for different plots
        self.tab_widget = QTabWidget()
        
        # Create plot widgets
        self.raw_data_widget = RawDataPlotWidget()
        self.bipolar_widget = BipolarPlotWidget()
        self.stacked_epochs_widget = StackedEpochsPlotWidget()
        self.power_spectrum_widget = PowerSpectrumPlotWidget()
        
        # Add tabs
        self.tab_widget.addTab(self.raw_data_widget, "Raw Data")
        self.tab_widget.addTab(self.bipolar_widget, "Bipolar Montage")
        self.tab_widget.addTab(self.stacked_epochs_widget, "Stacked Epochs")
        self.tab_widget.addTab(self.power_spectrum_widget, "Power Spectrum")
        
        main_layout.addWidget(self.tab_widget)
        
        # Status bar
        self.statusBar().showMessage("Ready")
    
    def load_default_file(self):
        """Load the default EDF file"""
        if self.default_edf_path:
            self.load_edf_file(self.default_edf_path)
    
    def load_file(self):
        """Open file dialog and load selected EDF file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select EDF File",
            str(Path(self.default_edf_path).parent) if self.default_edf_path else "",
            "EDF Files (*.edf);;All Files (*)"
        )
        
        if file_path:
            self.load_edf_file(file_path)
    
    def load_edf_file(self, file_path):
        """Load EDF file in background thread"""
        if self.load_thread and self.load_thread.isRunning():
            return
        
        self.file_label.setText(f"Loading: {os.path.basename(file_path)}")
        self.load_button.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Indeterminate progress
        
        # Create and start loading thread
        self.load_thread = LoadEDFThread(file_path)
        self.load_thread.finished.connect(self.on_data_loaded)
        self.load_thread.error.connect(self.on_load_error)
        self.load_thread.progress.connect(self.on_load_progress)
        self.load_thread.start()
    
    def on_data_loaded(self, processor):
        """Handle successful data loading"""
        self.processor = processor
        
        # Update file info
        self.file_label.setText(f"Loaded: {os.path.basename(self.load_thread.edf_path)}")
        
        info_text = f"""
Channels: {len(processor.channel_names)}
Sampling Rate: {processor.sfreq} Hz
Duration: {processor.data.shape[1] / processor.sfreq:.2f} seconds
Annotations: {len(processor.annotations)}
Bipolar Channels: {len(processor.bipolar_names)}

Available Annotation Types:
"""
        for ann_type, epochs in processor.annotation_groups.items():
            info_text += f"• {ann_type}: {len(epochs)} epochs\n"
        
        self.info_text.setText(info_text)
        
        # Set processor for all widgets
        self.raw_data_widget.set_processor(processor)
        self.bipolar_widget.set_processor(processor)
        self.stacked_epochs_widget.set_processor(processor)
        self.power_spectrum_widget.set_processor(processor)
        
        # Cleanup
        self.load_button.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.statusBar().showMessage("EDF file loaded successfully")
    
    def on_load_error(self, error_msg):
        """Handle loading errors"""
        QMessageBox.critical(self, "Error", error_msg)
        self.file_label.setText("No file loaded")
        self.load_button.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.statusBar().showMessage("Error loading file")
    
    def on_load_progress(self, message):
        """Handle progress updates"""
        self.statusBar().showMessage(message)
    
    def closeEvent(self, event):
        """Clean up when closing the application"""
        # Stop any running threads
        if self.load_thread and self.load_thread.isRunning():
            self.load_thread.terminate()
            self.load_thread.wait()
        
        # Clean up temporary files
        for widget in [self.raw_data_widget, self.bipolar_widget, 
                      self.stacked_epochs_widget, self.power_spectrum_widget]:
            widget.cleanup_temp_files()
        
        event.accept()

def main():
    parser = argparse.ArgumentParser(description='EDF Desktop Viewer - PyQt5 Application')
    parser.add_argument('edf_file', nargs='?', 
                       default=r'data\Russian\sub-1\sub-1.edf',
                       help='Path to EDF file (default: data\\Russian\\sub-1\\sub-1.edf)')
    
    args = parser.parse_args()
    
    app = QApplication(sys.argv)
    app.setApplicationName("EDF Desktop Viewer")
    
    # Set application font
    font = QFont()
    font.setPointSize(9)
    app.setFont(font)
    
    # Create and show main window
    main_window = MainWindow(args.edf_file)
    main_window.show()
    
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()