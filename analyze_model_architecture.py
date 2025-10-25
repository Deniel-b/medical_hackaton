from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Flatten
from tensorflow.keras.layers import PReLU, LeakyReLU

def create_model():
    """Создание модели точно как в train_eeg_classifier.py"""
    model = Sequential([
        # Входной слой для сглаживания
        Flatten(input_shape=(15, 20)),  # 15 эпох × 20 признаков = 300
        
        # Входной слой
        Dense(32, activation='relu'),
        Dropout(0.3),
        BatchNormalization(),
        
        # Скрытые слои с продвинутыми активациями
        Dense(64, activation=PReLU()),
        Dropout(0.3),
        BatchNormalization(),
        
        Dense(128, activation=PReLU()),
        Dropout(0.3),
        BatchNormalization(),
        
        Dense(256, activation=PReLU()),
        Dropout(0.3),
        BatchNormalization(),
        
        Dense(512, activation=PReLU()),        # Максимальная ширина
        Dropout(0.3),
        BatchNormalization(),
        
        Dense(256, activation='relu'),
        BatchNormalization(),
        
        Dense(128, activation=PReLU()),
        Dropout(0.1),
        BatchNormalization(),
        
        Dense(64, activation=PReLU()),
        Dropout(0.1),
        
        Dense(32, activation=LeakyReLU(alpha=0.1)),
        
        Dense(16, activation=PReLU()),
        Dropout(0.1),
        
        # Выходной слой
        Dense(7, activation='softmax')         # 7 классов движений
    ])
    
    return model

def print_architecture():
    """Выводит детальную архитектуру модели"""
    
    print("🧠 Создание модели...")
    model = create_model()
    model.compile(optimizer='nadam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    print("\n" + "="*80)
    print("         EEG MOVEMENT CLASSIFICATION NEURAL NETWORK ARCHITECTURE")
    print("="*80)
    
    model.summary()
    
    print("\n" + "="*80)
    print("                              LAYER ANALYSIS")
    print("="*80)
    
    total_params = model.count_params()
    
    print(f"\n📊 ОБЩАЯ СТАТИСТИКА:")
    print(f"   • Общее количество параметров: {total_params:,}")
    print(f"   • Количество слоев: {len(model.layers)}")
    print(f"   • Входная размерность: (15 эпох × 20 признаков) = 300")
    print(f"   • Выходная размерность: 7 классов движений")
    
    print(f"\n🏗️ АРХИТЕКТУРНЫЙ ПАТТЕРН:")
    print(f"   • Тип: Deep Feedforward Neural Network")
    print(f"   • Форма: Diamond (Расширение → Максимум → Сужение)")
    print(f"   • Максимальная ширина: 512 нейронов")
    
    print(f"\n⚙️ ДЕТАЛИ СЛОЕВ:")
    layer_count = 0
    dense_count = 0
    
    for i, layer in enumerate(model.layers):
        layer_type = layer.__class__.__name__
        
        if layer_type == 'Dense':
            dense_count += 1
            neurons = layer.units
            
            if hasattr(layer, 'activation'):
                activation_name = layer.activation.__name__ if hasattr(layer.activation, '__name__') else str(layer.activation)
            else:
                activation_name = 'linear'
                
            params = layer.count_params()
            
            print(f"   {dense_count:2d}. Dense Layer: {neurons:3d} neurons, {activation_name:>10s}, {params:,} параметров")
        
        elif layer_type == 'Dropout':
            rate = layer.rate
            print(f"       └─ Dropout: rate={rate}")
            
        elif layer_type == 'BatchNormalization':
            params = layer.count_params()
            print(f"       └─ BatchNorm: {params} параметров")
            
        elif layer_type == 'Flatten':
            print(f"   {i+1:2d}. Flatten Layer: 300 features")
    
    print(f"\n🎯 КЛАССЫ ДВИЖЕНИЙ:")
    classes = ['Up', 'Down', 'Left', 'Right', 'Back', 'Forward', 'None']
    for i, class_name in enumerate(classes):
        print(f"   {i}: {class_name}")

def create_text_diagram():
    """Создание ASCII диаграммы архитектуры"""
    
    diagram = """
    
    ╔═══════════════════════════════════════════════════════════════════╗
    ║                EEG MOVEMENT CLASSIFICATION NEURAL NETWORK        ║
    ╚═══════════════════════════════════════════════════════════════════╝
    
    INPUT: EEG Data (15 epochs × 20 features = 300)
           │
           ▼
    ┌─────────────────────────────────────────────────────────────────┐
    │                      FLATTEN LAYER                             │
    │                     (300 features)                             │
    └─────────────────────────────────────────────────────────────────┘
           │
           ▼
    ┌─────────────────────────────────────────────────────────────────┐
    │                   DENSE LAYER 1: 32                            │
    │               ReLU activation                                   │
    │                    Dropout (0.3)                               │
    │                 Batch Normalization                            │
    └─────────────────────────────────────────────────────────────────┘
           │
           ▼
    ┌─────────────────────────────────────────────────────────────────┐
    │                   DENSE LAYER 2: 64                            │
    │               PReLU activation                                  │
    │                    Dropout (0.3)                               │
    │                 Batch Normalization                            │
    └─────────────────────────────────────────────────────────────────┘
           │
           ▼
    ┌─────────────────────────────────────────────────────────────────┐
    │                   DENSE LAYER 3: 128                           │
    │              PReLU activation                                   │
    │                    Dropout (0.3)                               │
    │                 Batch Normalization                            │
    └─────────────────────────────────────────────────────────────────┘
           │
           ▼
    ┌─────────────────────────────────────────────────────────────────┐
    │                   DENSE LAYER 4: 256                           │
    │              PReLU activation                                   │
    │                    Dropout (0.3)                               │
    │                 Batch Normalization                            │
    └─────────────────────────────────────────────────────────────────┘
           │
           ▼
    ┌─────────────────────────────────────────────────────────────────┐
    │               DENSE LAYER 5: 512 (MAX WIDTH)                   │
    │              PReLU activation                                   │
    │                    Dropout (0.3)                               │
    │                 Batch Normalization                            │
    └─────────────────────────────────────────────────────────────────┘
           │
           ▼
    ┌─────────────────────────────────────────────────────────────────┐
    │                   DENSE LAYER 6: 256                           │
    │               ReLU activation                                   │
    │                 Batch Normalization                            │
    └─────────────────────────────────────────────────────────────────┘
           │
           ▼
    ┌─────────────────────────────────────────────────────────────────┐
    │                   DENSE LAYER 7: 128                           │
    │              PReLU activation                                   │
    │                    Dropout (0.1)                               │
    │                 Batch Normalization                            │
    └─────────────────────────────────────────────────────────────────┘
           │
           ▼
    ┌─────────────────────────────────────────────────────────────────┐
    │                   DENSE LAYER 8: 64                            │
    │              PReLU activation                                   │
    │                    Dropout (0.1)                               │
    └─────────────────────────────────────────────────────────────────┘
           │
           ▼
    ┌─────────────────────────────────────────────────────────────────┐
    │                   DENSE LAYER 9: 32                            │
    │             LeakyReLU activation (α=0.1)                       │
    └─────────────────────────────────────────────────────────────────┘
           │
           ▼
    ┌─────────────────────────────────────────────────────────────────┐
    │                   DENSE LAYER 10: 16                           │
    │              PReLU activation                                   │
    │                    Dropout (0.1)                               │
    └─────────────────────────────────────────────────────────────────┘
           │
           ▼
    ┌─────────────────────────────────────────────────────────────────┐
    │                    OUTPUT LAYER: 7                             │
    │              Softmax activation                                 │
    │        (Up, Down, Left, Right, Back, Forward, None)            │
    └─────────────────────────────────────────────────────────────────┘
    
    Architecture Pattern: 32 → 64 → 128 → 256 → 512 → 256 → 128 → 64 → 32 → 16 → 7
    Total Parameters: ~365,255
    Network Type: Deep Feedforward (Diamond Shape)
    
    ╔═══════════════════════════════════════════════════════════════════╗
    ║  ACTIVATION FUNCTIONS:                                            ║
    ║  • ReLU: Standard rectified linear activation                     ║
    ║  • PReLU: Parametric ReLU (learnable slope for negative values)   ║
    ║  • LeakyReLU: ReLU with small slope for negative values          ║
    ║  • Softmax: Probability distribution for multi-class output      ║
    ╚═══════════════════════════════════════════════════════════════════╝
    """
    
    print(diagram)
    
    # Сохраняем в файл
    with open('eeg_architecture_diagram.txt', 'w', encoding='utf-8') as f:
        f.write(diagram)
    
    print("✅ Текстовая диаграмма сохранена как 'eeg_architecture_diagram.txt'")

if __name__ == "__main__":
    print("🧠 Анализ архитектуры нейронной сети EEG Classifier...")
    
    # Выводим детальную информацию
    print_architecture()
    
    print("\n" + "="*80)
    print("                             ASCII ДИАГРАММА")
    print("="*80)
    
    # Создаем текстовую диаграмму
    create_text_diagram()
    
    print("\n🎨 Анализ архитектуры завершен!")